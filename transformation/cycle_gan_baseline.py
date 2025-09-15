"""
CycleGAN (Unpaired Image-to-Image Translation)
---------------------------------------------
- X 도메인(일반 사진) ↔ Y 도메인(포켓몬 이미지) 간 변환 (unpaired)
- 2xGenerator(G_XY, G_YX) + 2PxatchGAN Discriminator(D_X, D_Y)
- Loss: LSGAN(MSE) + Cycle(L1) + Identity(L1)

Quick Start
1) 폴더 구성
   data/
     X/   # 일반 사진들 (AFHQ 등)
     Y/   # 포켓몬 이미지들 (PNG/JPG)

2) 설치 (가상환경)
   pip install torch torchvision pillow

3) 학습 (해상도 128 권장)
   python cycle_gan_clean.py \
     --mode train \
     --data_x ./data/X \
     --data_y ./data/Y \
     --out_dir runs/pokemon128 \
     --size 128 --epochs 30 --batch 4 --lr 2e-4

4) 추론 (X→Y)
   python cycle_gan_clean.py \
     --mode test \
     --gen_ckpt runs/pokemon128/checkpoints/G_XY_latest.pt \
     --in_dir ./data/X \
     --out_dir runs/pokemon128/infer \
     --size 128

샘플링 옵션:
   --max_x 500   # X(예: AFHQ)에서 랜덤 500장만 사용
   --max_y 500   # Y(포켓몬)도 랜덤 제한

Apple Silicon(M1/M2/M3): mps 자동 사용. 필요시
   export PYTORCH_ENABLE_MPS_FALLBACK=1
"""

# ===== add this near the top (before torchvision models are used) =====
import os, torch
CACHE_DIR = os.path.expanduser('~/Desktop/Pokemon-style/torch_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR
try:
    import torch.hub as hub
    hub.set_dir(CACHE_DIR)
except Exception:
    pass
# =====================================================================
import glob
import csv
import math
import random
import argparse
from typing import List, Tuple, Optional

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np

# ------------------------------
# Device (CUDA / MPS / CPU)
# ------------------------------
DEVICE = (
    "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
)
print(f"[Device] {DEVICE}")

# ------------------------------
# Utils
# ------------------------------
def seed_all(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def weights_init(m: nn.Module):
    cn = m.__class__.__name__
    if 'Conv' in cn or 'Linear' in cn:
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
    if 'InstanceNorm' in cn:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def denorm(x: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [0,1]
    return (x.clamp_(-1, 1) * 0.5 + 0.5)

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------------
# Dataset (Unpaired) + Random Subsample
# ------------------------------
class UnpairedImageDataset(Dataset):
    def __init__(self, x_root: str, y_root: str, size: int = 128, max_x: int = 0, max_y: int = 0):
        self.x_paths = self._collect(x_root)
        self.y_paths = self._collect(y_root)

        if max_x > 0 and len(self.x_paths) > max_x:
            random.shuffle(self.x_paths)
            self.x_paths = self.x_paths[:max_x]
        if max_y > 0 and len(self.y_paths) > max_y:
            random.shuffle(self.y_paths)
            self.y_paths = self.y_paths[:max_y]

        if not self.x_paths:
            raise RuntimeError(f"No images in X: {x_root}")
        if not self.y_paths:
            raise RuntimeError(f"No images in Y: {y_root}")

        self.size = size
        self.tf = transforms.Compose([
            transforms.Resize(size + 30),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # 평가용 변환(중앙크롭, resize-fixed)
        self.eval_tf = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    def _collect(self, root: str) -> List[str]:
        exts = (".png", ".jpg", ".jpeg", ".webp")
        paths: List[str] = []
        if os.path.isdir(root):
            for e in exts:
                paths += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)
        else:
            if any(root.lower().endswith(x) for x in exts):
                paths = [root]
        return sorted(paths)

    def __len__(self):
        return max(len(self.x_paths), len(self.y_paths))

    def __getitem__(self, idx: int):
        x_path = self.x_paths[idx % len(self.x_paths)]
        y_path = self.y_paths[random.randrange(len(self.y_paths))]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        return self.tf(x), self.tf(y)

    # 평가용 작은 배치 생성 (고정 시드로 재현)
    def sample_eval_batch(self, n: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(1234)
        xs = []
        ys = []
        for i in range(n):
            x_path = self.x_paths[rng.randrange(len(self.x_paths))]
            y_path = self.y_paths[rng.randrange(len(self.y_paths))]
            x = Image.open(x_path).convert('RGB')
            y = Image.open(y_path).convert('RGB')
            xs.append(self.eval_tf(x))
            ys.append(self.eval_tf(y))
        return torch.stack(xs, 0), torch.stack(ys, 0)

# ------------------------------
# Models: ResNet Generator & PatchGAN Discriminator
# ------------------------------
class ResnetBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)

class ResnetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, 1, 0),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),

            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf*4)]
        model += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   3, 2, 1, output_padding=1), nn.InstanceNorm2d(ngf),   nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_ch, 7, 1, 0),
            nn.Tanh(),
        ]
        self.net = nn.Sequential(*model)
    def forward(self, x):
        return self.net(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1), nn.InstanceNorm2d(ndf*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1), nn.InstanceNorm2d(ndf*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*4, ndf*8, 4, 1, 1), nn.InstanceNorm2d(ndf*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf*8, 1, 4, 1, 1),
        )
    def forward(self, x):
        return self.net(x)

# ------------------------------
# Image Replay Buffer (stabilize D)
# ------------------------------
class ImagePool:
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images: List[torch.Tensor] = []
    def query(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0:
            return imgs
        out_imgs = []
        for img in imgs:
            img = img.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                out_imgs.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randrange(self.pool_size)
                    tmp = self.images[idx].clone()
                    self.images[idx] = img
                    out_imgs.append(tmp)
                else:
                    out_imgs.append(img)
        return torch.cat(out_imgs, dim=0)

# ------------------------------
# Metrics: FID & LPIPS (VGG16-feat)
# ------------------------------
class InceptionV3Feat(nn.Module):
    """InceptionV3 2048-d feature extractor for FID."""
    def __init__(self):
        super().__init__()
        try:
            # torchvision >=0.13
            weights = getattr(models, "Inception_V3_Weights", None)
            if weights is not None:
                net = models.inception_v3(weights=weights.IMAGENET1K_V1, aux_logits=True)
                net.fc = nn.Identity()
                self.net = net.eval().to(DEVICE)
            else:
                net = models.inception_v3(pretrained=True, aux_logits=False)
            net.fc = nn.Identity()
            self.net = net.eval().to(DEVICE)
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.ok = True
        except Exception as e:
            print(f"[WARN] InceptionV3 weights not available: {e}")
            self.ok = False

        # Inception input normalization
        self.pre = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
        ])

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.ok:
            return None
        x = denorm(imgs)  # [0,1]
        x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        out = self.net(x)
        # ✅ torchvision 버전에 따라 InceptionOutputs가 나올 수 있음
        if hasattr(out, "logits"):  # torchvision>=0.13
            out = out.logits
        return out  # [N,2048] (fc=Identity라 풀링 특징이 나옴)

def _cov_mean(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(feats, axis=0)
    feats_centered = feats - mu
    cov = feats_centered.T @ feats_centered / (feats.shape[0] - 1)
    return cov, mu

def _sqrtm(mat: np.ndarray) -> np.ndarray:
    """Robust matrix square root for (generally non-symmetric) matrix."""
    try:
        from scipy.linalg import sqrtm
        s = sqrtm(mat)
        # 수치적으로 아주 작은 허수부 제거
        if np.iscomplexobj(s):
            s = s.real
        return s
    except Exception:
        # scipy가 없거나 실패하면, 대칭화 후 PSD 방식으로 근사
        sym = (mat + mat.T) / 2.0
        vals, vecs = np.linalg.eigh(sym)
        vals = np.clip(vals, 0.0, None)
        return (vecs * np.sqrt(vals)) @ vecs.T

def fid_score(feats_fake: np.ndarray, feats_real: np.ndarray, eps: float = 1e-6) -> float:
    """Fréchet Inception Distance. 수치안정화 포함."""
    cov1, mu1 = _cov_mean(feats_fake)
    cov2, mu2 = _cov_mean(feats_real)

    # ✅ 대각 안정화
    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    diff = mu1 - mu2
    covmean = _sqrtm(cov1 @ cov2)
    tr_covmean = np.trace(covmean)

    fid = diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * tr_covmean
    # ✅ 음수로 튀는 수치오차 방지
    return float(max(fid, 0.0))

class VGG16LPIPS(nn.Module):
    """LPIPS-like perceptual distance using VGG16 features (uncalibrated)."""
    def __init__(self):
        super().__init__()
        try:
            weights = getattr(models, "VGG16_Weights", None)
            if weights is not None:
                vgg = models.vgg16(weights=weights.IMAGENET1K_V1).features
            else:
                vgg = models.vgg16(pretrained=True).features
            # Slice into blocks
            self.blocks = nn.ModuleList([
                vgg[:4],    # relu1_2
                vgg[4:9],   # relu2_2
                vgg[9:16],  # relu3_3
                vgg[16:23], # relu4_3
            ])
            for p in self.parameters():
                p.requires_grad_(False)
            # ✅ 추가: 평가모드 + DEVICE로 올리기
            self.blocks = self.blocks.eval().to(DEVICE)

            self.ok = True
        except Exception as e:
            print(f"[WARN] VGG16 weights not available: {e}")
            self.ok = False

        # Expect input in [-1,1]; convert to ImageNet norm
        self.to_imagenet = transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225],
        )

    def _norm(self, x):
        x = denorm(x)  # [0,1]
        return self.to_imagenet(x)

    @torch.no_grad()
    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.ok:
            return None
        # ✅ 추가: 블록의 디바이스로 맞추기
        dev = next(self.blocks.parameters()).device
        x = x.to(dev)
        y = y.to(dev)

        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        y = F.interpolate(y, size=(224,224), mode="bilinear", align_corners=False)
        x = self._norm(x); y = self._norm(y)
        dist = 0.0
        feat_x = x
        feat_y = y
        for blk in self.blocks:
            feat_x = blk(feat_x)
            feat_y = blk(feat_y)
            nx = F.normalize(feat_x, p=2, dim=1)
            ny = F.normalize(feat_y, p=2, dim=1)
            dist += F.mse_loss(nx, ny, reduction='mean')
        return dist

@torch.no_grad()
def evaluate_metrics(G_XY: nn.Module,
                     ds: UnpairedImageDataset,
                     eval_n: int,
                     incep: InceptionV3Feat,
                     lpips_vgg: VGG16LPIPS) -> Tuple[float, float]:
    """Return (FID, LPIPS) over a small fixed eval batch."""
    G_XY.eval()
    xs, ys_real = ds.sample_eval_batch(n=eval_n)
    xs = xs.to(DEVICE)
    ys_real = ys_real.to(DEVICE)
    ys_fake = G_XY(xs)

    # FID
    fid_val = float("nan")
    if incep is not None and incep.ok:
        f_fake = incep(ys_fake)
        f_real = incep(ys_real)
        if (f_fake is not None) and (f_real is not None):
            f_fake = f_fake.detach().cpu().numpy()
            f_real = f_real.detach().cpu().numpy()
            fid_val = fid_score(f_fake, f_real)

    # LPIPS (lower is better)
    lpips_val = float("nan")
    if lpips_vgg is not None and lpips_vgg.ok:
        lp_val = lpips_vgg(ys_fake, ys_real)
        if lp_val is not None:
            lpips_val = float(lp_val.item())
    return fid_val, lpips_val

# ------------------------------
# Train / Test
# ------------------------------
def make_loader(ds: UnpairedImageDataset, batch: int, workers: int):
    return DataLoader(
        ds, batch_size=batch, shuffle=True,
        num_workers=workers, drop_last=True,
        pin_memory=(DEVICE == 'cuda'),
    )

def _write_metrics_header(csv_path: str):
    is_new = not os.path.exists(csv_path)
    if is_new:
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["epoch","g_total","d_x","d_y","cyc","idt","fid","lpips","lr_g","lr_d"])

def _append_metrics(csv_path: str, row: List):
    with open(csv_path, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(row)

def _maybe_plot_metrics(csv_path: str, out_dir: str, ema_alpha: float = 0.6):
    """
    두 장으로 분리 출력:
      - losses.png : G_total, D_X, D_Y, cyc, idt
      - scores.png : FID, LPIPS  (EMA 스무딩)
    NaN/빈칸 안전 처리.
    """
    try:
        import matplotlib.pyplot as plt
        import csv, math, os
        import numpy as np

        ep, g, dx, dy, cyc, idt, fid, lp = [], [], [], [], [], [], [], []
        with open(csv_path, 'r') as f:
            r = csv.DictReader(f)
            for row in r:
                ep.append(int(row["epoch"]))
                g.append(float(row["g_total"]))
                dx.append(float(row["d_x"]))
                dy.append(float(row["d_y"]))
                cyc.append(float(row["cyc"]))
                idt.append(float(row["idt"]))
                # FID/LPIPS는 빈칸일 수 있음
                fid.append(float(row["fid"]) if row["fid"] != "" else float("nan"))
                lp.append(float(row["lpips"]) if row["lpips"] != "" else float("nan"))

        def ema(arr, alpha=0.6):
            out = []
            m = None
            for v in arr:
                if math.isnan(v):
                    out.append(math.nan)
                    continue
                m = (alpha * v) + ((1 - alpha) * m) if m is not None else v
                out.append(m)
            return out

        # ---------- 1) 손실 그래프 ----------
        plt.figure()
        plt.plot(ep, g,  label="G_total")
        plt.plot(ep, dx, label="D_X")
        plt.plot(ep, dy, label="D_Y")
        plt.plot(ep, cyc, label="cyc")
        plt.plot(ep, idt, label="idt")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Losses")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "losses.png"))
        plt.close()

        # ---------- 2) 품질지표 그래프 (EMA 스무딩 + NaN 제거) ----------
        # 유효 값만 골라서 그릴 수 있게 마스킹
        def valid_xy(x, y):
            xv, yv = [], []
            for xi, yi in zip(x, y):
                if not math.isnan(yi):
                    xv.append(xi); yv.append(yi)
            return xv, yv

        ep_fid, fid_v = valid_xy(ep, ema(fid, ema_alpha))
        ep_lp,  lp_v  = valid_xy(ep, ema(lp, ema_alpha))

        plt.figure()
        if len(fid_v) > 0:
            plt.plot(ep_fid, fid_v, label=f"FID (EMA {ema_alpha})")
        if len(lp_v) > 0:
            plt.plot(ep_lp,  lp_v,  label=f"LPIPS (EMA {ema_alpha})")
        plt.xlabel("Epoch")
        plt.ylabel("Score (lower is better)")
        plt.title("Quality Scores")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "scores.png"))
        plt.close()
    except Exception as e:
        print(f"[INFO] metrics plot skipped: {e}")


def train(args):
    seed_all(args.seed)

    ds = UnpairedImageDataset(args.data_x, args.data_y, size=args.size,
                              max_x=args.max_x, max_y=args.max_y)
    dl = make_loader(ds, args.batch, args.workers)

    n_blocks = 6 if args.size <= 128 else 9
    G_XY = ResnetGenerator(n_blocks=n_blocks).to(DEVICE)
    G_YX = ResnetGenerator(n_blocks=n_blocks).to(DEVICE)
    D_X = PatchDiscriminator().to(DEVICE)
    D_Y = PatchDiscriminator().to(DEVICE)

    G_XY.apply(weights_init); G_YX.apply(weights_init)
    D_X.apply(weights_init); D_Y.apply(weights_init)

    adv_loss = nn.MSELoss()  # LSGAN
    l1 = nn.L1Loss()
    # ---- LR = 0.0002 (기본값 2e-4)
    g_opt = torch.optim.Adam(list(G_XY.parameters()) + list(G_YX.parameters()), lr=args.lr, betas=(0.5, 0.999))
    d_x_opt = torch.optim.Adam(D_X.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_y_opt = torch.optim.Adam(D_Y.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # linear decay after args.decay
    def lambda_rule(epoch):
        return 1.0 - max(0, epoch - args.decay) / float(max(1, args.epochs - args.decay))
    g_sch = torch.optim.lr_scheduler.LambdaLR(g_opt, lr_lambda=lambda_rule)
    dx_sch = torch.optim.lr_scheduler.LambdaLR(d_x_opt, lr_lambda=lambda_rule)
    dy_sch = torch.optim.lr_scheduler.LambdaLR(d_y_opt, lr_lambda=lambda_rule)

    pool_X = ImagePool(50)
    pool_Y = ImagePool(50)

    safe_mkdir(args.out_dir)
    samples_dir = os.path.join(args.out_dir, 'samples')
    ckpt_dir = os.path.join(args.out_dir, 'checkpoints')
    safe_mkdir(samples_dir)
    safe_mkdir(ckpt_dir)

    # Metrics logger
    csv_path = os.path.join(args.out_dir, "metrics.csv")
    _write_metrics_header(csv_path)

    # Metrics models
    incep = InceptionV3Feat()
    lpips_vgg = VGG16LPIPS()

    for epoch in range(1, args.epochs + 1):
        G_XY.train(); G_YX.train(); D_X.train(); D_Y.train()
        running = {"g":0.0,"dx":0.0,"dy":0.0,"cyc":0.0,"idt":0.0,"n":0}

        for i, (real_x, real_y) in enumerate(dl, 1):
            real_x = real_x.to(DEVICE)
            real_y = real_y.to(DEVICE)

            # ===== 1) Train Generators =====
            g_opt.zero_grad()
            fake_y = G_XY(real_x)
            rec_x  = G_YX(fake_y)

            fake_x = G_YX(real_y)
            rec_y  = G_XY(fake_x)

            # identity (helps color preservation)
            idt_y = G_XY(real_y)
            idt_x = G_YX(real_x)

            # D outputs & dynamic labels (no hard-coded size)
            pred_fake_y = D_Y(fake_y)
            pred_fake_x = D_X(fake_x)
            valid_y = torch.ones_like(pred_fake_y)
            valid_x = torch.ones_like(pred_fake_x)

            adv_XY = adv_loss(pred_fake_y, valid_y)
            adv_YX = adv_loss(pred_fake_x, valid_x)

            cyc = l1(rec_x, real_x) + l1(rec_y, real_y)
            idt = l1(idt_x, real_x) + l1(idt_y, real_y)

            g_total = adv_XY + adv_YX + args.lambda_cyc * cyc + args.lambda_idt * idt
            g_total.backward()
            g_opt.step()

            # ===== 2) Train D_X =====
            d_x_opt.zero_grad()
            pred_real_x = D_X(real_x)
            valid_dx = torch.ones_like(pred_real_x)

            fake_x_pool = pool_X.query(fake_x.detach())
            pred_fake_x_pool = D_X(fake_x_pool)
            fake_dx = torch.zeros_like(pred_fake_x_pool)

            loss_dx_real = adv_loss(pred_real_x, valid_dx)
            loss_dx_fake = adv_loss(pred_fake_x_pool, fake_dx)
            loss_dx = 0.5 * (loss_dx_real + loss_dx_fake)
            loss_dx.backward()
            d_x_opt.step()

            # ===== 3) Train D_Y =====
            d_y_opt.zero_grad()
            pred_real_y = D_Y(real_y)
            valid_dy = torch.ones_like(pred_real_y)

            fake_y_pool = pool_Y.query(fake_y.detach())
            pred_fake_y_pool = D_Y(fake_y_pool)
            fake_dy = torch.zeros_like(pred_fake_y_pool)

            loss_dy_real = adv_loss(pred_real_y, valid_dy)
            loss_dy_fake = adv_loss(pred_fake_y_pool, fake_dy)
            loss_dy = 0.5 * (loss_dy_real + loss_dy_fake)
            loss_dy.backward()
            d_y_opt.step()

            running["g"]  += g_total.item()
            running["dx"] += loss_dx.item()
            running["dy"] += loss_dy.item()
            running["cyc"] += cyc.item()
            running["idt"] += idt.item()
            running["n"]  += 1

            if i % args.print_every == 0:
                print(f"[E{epoch:03d}/{args.epochs}] [B{i:04d}/{len(dl)}] "
                      f"G:{g_total.item():.3f} D_X:{loss_dx.item():.3f} D_Y:{loss_dy.item():.3f} "
                      f"cyc:{cyc.item():.3f} id:{idt.item():.3f}")

        g_sch.step(); dx_sch.step(); dy_sch.step()

        # ---- Evaluate FID & LPIPS on small fixed batch (every epoch)
        with torch.no_grad():
            fid_val, lpips_val = evaluate_metrics(G_XY, ds, eval_n=args.eval_num, incep=incep, lpips_vgg=lpips_vgg)

        # ---- Save sample grid every N epochs (configurable)
        save_every = getattr(args, "sample_every", 1)
        if (epoch % save_every) == 0:
            try:
                with torch.no_grad():
                    G_XY.eval(); G_YX.eval()

                    # 작은 샘플 4+4장
                    sample_x = [ds.x_paths[k % len(ds.x_paths)] for k in range(4)]
                    sample_y = [ds.y_paths[k % len(ds.y_paths)] for k in range(4)]

                    def load_eval_batch(paths):
                        ims = []
                        for p in paths:
                            # 안전하게 파일 핸들 닫기
                            with Image.open(p) as img:
                                img = img.convert('RGB')
                                ims.append(ds.eval_tf(img))
                        batch = torch.stack(ims, 0).to(DEVICE)
                        return batch

                    rx = load_eval_batch(sample_x)
                    ry = load_eval_batch(sample_y)

                    fy = G_XY(rx); fx = G_YX(ry)
                    rcx = G_YX(fy); rcy = G_XY(fx)

                    grid = make_grid(
                        torch.cat([rx, fy, rcx, ry, fx, rcy], 0).detach().cpu(),  # ✅ CPU로 이동
                        nrow=4, normalize=True, value_range=(-1,1)
                    )
                    out_path = os.path.join(samples_dir, f'epoch_{epoch:03d}.png')
                    save_image(grid, out_path)
                    print(f"[SAMPLE] saved -> {out_path}")
            except Exception as e:
                print(f"[WARN] sample saving failed at epoch {epoch}: {e}")

        # ---- Save checkpoints (every epoch)
        torch.save(G_XY.state_dict(), os.path.join(ckpt_dir, f'G_XY_epoch{epoch:03d}.pt'))
        torch.save(G_YX.state_dict(), os.path.join(ckpt_dir, f'G_YX_epoch{epoch:03d}.pt'))
        torch.save(D_X.state_dict(),  os.path.join(ckpt_dir, f'D_X_epoch{epoch:03d}.pt'))
        torch.save(D_Y.state_dict(),  os.path.join(ckpt_dir, f'D_Y_epoch{epoch:03d}.pt'))
        # Also update latest
        torch.save(G_XY.state_dict(), os.path.join(ckpt_dir,'G_XY_latest.pt'))
        torch.save(G_YX.state_dict(), os.path.join(ckpt_dir,'G_YX_latest.pt'))
        torch.save(D_X.state_dict(),  os.path.join(ckpt_dir,'D_X_latest.pt'))
        torch.save(D_Y.state_dict(),  os.path.join(ckpt_dir,'D_Y_latest.pt'))

                # ---- Log metrics
        n = max(1, running["n"])
        g_avg  = running["g"]/n
        dx_avg = running["dx"]/n
        dy_avg = running["dy"]/n
        cyc_avg= running["cyc"]/n
        idt_avg= running["idt"]/n
        current_lr = g_opt.param_groups[0]["lr"]
        _append_metrics(csv_path, [
            epoch, f"{g_avg:.6f}", f"{dx_avg:.6f}", f"{dy_avg:.6f}",
            f"{cyc_avg:.6f}", f"{idt_avg:.6f}",
            f"{fid_val:.6f}" if not math.isnan(fid_val) else "",
            f"{lpips_val:.6f}" if not math.isnan(lpips_val) else "",
            f"{current_lr:.8f}", f"{current_lr:.8f}"
        ])
        _maybe_plot_metrics(csv_path, args.out_dir, ema_alpha=0.6)

        # ✅ 콘솔 요약 출력 (NaN 안전 처리)
        fid_str = "NaN" if math.isnan(fid_val) else f"{fid_val:.2f}"
        lpips_str = "NaN" if math.isnan(lpips_val) else f"{lpips_val:.3f}"
        print(f"[E{epoch:03d}] avg G:{g_avg:.3f} DX:{dx_avg:.3f} DY:{dy_avg:.3f} "
              f"cyc:{cyc_avg:.3f} idt:{idt_avg:.3f} | FID:{fid_str} LPIPS:{lpips_str} | lr:{current_lr:.6f}")

@torch.no_grad()
def test(args):
    size = args.size
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    n_blocks = 6 if size <= 128 else 9
    G_XY = ResnetGenerator(n_blocks=n_blocks).to(DEVICE)
    ckpt = torch.load(args.gen_ckpt, map_location='cpu')
    G_XY.load_state_dict(ckpt, strict=True)
    G_XY.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    exts = (".png",".jpg",".jpeg",".webp")
    paths = []
    if os.path.isdir(args.in_dir):
        for e in exts:
            paths += glob.glob(os.path.join(args.in_dir, f"**/*{e}"), recursive=True)
    else:
        if any(args.in_dir.lower().endswith(x) for x in exts):
            paths = [args.in_dir]
    if not paths:
        raise RuntimeError(f"No images in: {args.in_dir}")

    for p in paths:
        img = Image.open(p).convert('RGB')
        x = tf(img).unsqueeze(0).to(DEVICE)
        y = G_XY(x)
        y = denorm(y).cpu()
        out_path = os.path.join(args.out_dir, os.path.basename(p))
        save_image(y, out_path)
    print(f"Saved translated images -> {args.out_dir}")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, choices=['train','test'], required=True)
    # train
    ap.add_argument('--data_x', type=str, help='X domain root (e.g., AFHQ/photos)')
    ap.add_argument('--data_y', type=str, help='Y domain root (e.g., pokemon)')
    ap.add_argument('--out_dir', type=str, default='runs/cyclegan')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--decay', type=int, default=15, help='start of linear LR decay epoch')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-4)  # 0.0002
    ap.add_argument('--lambda_cyc', type=float, default=10.0)
    ap.add_argument('--lambda_idt', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0, help='DataLoader workers (macOS는 0 권장)')
    ap.add_argument('--print_every', type=int, default=50)
    ap.add_argument('--max_x', type=int, default=0, help='X domain 샘플 최대 수 (0=무제한)')
    ap.add_argument('--max_y', type=int, default=0, help='Y domain 샘플 최대 수 (0=무제한)')
    ap.add_argument('--eval_num', type=int, default=64, help='각 epoch 평가용 샘플 수 (작게 유지 권장)')
    # test
    ap.add_argument('--gen_ckpt', type=str, help='G_XY checkpoint path (for test)')
    ap.add_argument('--in_dir', type=str, help='input folder or image path (test mode)')
    args = ap.parse_args()

    if args.mode == 'train':
        if not args.data_x or not args.data_y:
            raise SystemExit('--data_x and --data_y are required in train mode')
        os.makedirs(args.out_dir, exist_ok=True)
        train(args)
    else:
        if not args.gen_ckpt or not args.in_dir or not args.out_dir:
            raise SystemExit('--gen_ckpt, --in_dir, --out_dir are required in test mode')
        test(args)

if __name__ == '__main__':
    main()
