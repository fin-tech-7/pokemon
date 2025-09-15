"""
ViT-CycleGAN (Unpaired Image-to-Image Translation, Pokémon style)
+ Metrics: FID, LPIPS, CSV logging, plots (scores.png, losses.png)
-----------------------------------------------------------------
- Generator: Vision Transformer (patchify → Transformer blocks → unpatchify/upsample)
- Discriminator: 70x70 PatchGAN (Conv)
- Loss: LSGAN (MSE) + Cycle (L1) + Identity (L1)
- Metrics:
    * FID (InceptionV3 pool3, 2048-d)
    * LPIPS (if 'lpips' package available; else VGG16 perceptual-distance fallback)

Quick Start (Apple Silicon M1/M2/M3 자동 mps 사용)
1) 데이터
   AFHQ 등 일반 사진을 X, 포켓몬 이미지를 Y 로 준비:
     ./AFHQ/**.{png,jpg,jpeg,webp}
     ./Pokemon_Dataset/**.{png,jpg,jpeg,webp}

2) 설치
   pip install torch torchvision pillow matplotlib
   # (선택) LPIPS:
   # pip install lpips

3) 학습 (빠른 확인용)
   python vit_cyclegan.py \
     --mode train \
     --data_x ./AFHQ \
     --data_y ./Pokemon_Dataset \
     --out_dir runs/vit_pokemon96_quick \
     --size 96 --epochs 5 --batch 4 --lr 5e-4 \
     --max_x 200 --max_y 200 \
     --metrics fid,lpips --eval_every 1 --eval_size 100

4) 추론 (X→Y) — 파일 또는 폴더
   python vit_cyclegan.py \
     --mode test \
     --gen_ckpt runs/vit_pokemon96_quick/checkpoints/G_XY_latest.pt \
     --in_dir ./my_photos \
     --out_dir runs/vit_pokemon96_quick/infer \
     --size 96

Tips
- 해상도는 size가 patch의 배수여야 함(기본 patch=8 → 64/96/128 권장).
- 품질↑: --embed 256 --depth 6 (느려짐), size=128 재학습.
- 재현성: --seed 고정.
"""

import os, glob, random, argparse, csv, math, time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.linalg  # for fid matrix sqrt

# ------------------------------
# Device (CUDA / MPS / CPU)
# ------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
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
        if getattr(m, "weight", None) is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if 'InstanceNorm' in cn:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp_(-1, 1) * 0.5 + 0.5)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# ------------------------------
# Dataset (Unpaired) + Random Subsample
# ------------------------------
class UnpairedImageDataset(Dataset):
    def __init__(self, x_root: str, y_root: str, size: int = 96, max_x: int = 0, max_y: int = 0):
        self.x_paths = self._collect(x_root)
        self.y_paths = self._collect(y_root)
        if max_x > 0 and len(self.x_paths) > max_x:
            random.shuffle(self.x_paths); self.x_paths = self.x_paths[:max_x]
        if max_y > 0 and len(self.y_paths) > max_y:
            random.shuffle(self.y_paths); self.y_paths = self.y_paths[:max_y]
        if not self.x_paths: raise RuntimeError(f"No images in X: {x_root}")
        if not self.y_paths: raise RuntimeError(f"No images in Y: {y_root}")
        self.tf = transforms.Compose([
            transforms.Resize(size + 30),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),
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

    def __len__(self): return max(len(self.x_paths), len(self.y_paths))
    def __getitem__(self, idx: int):
        x_path = self.x_paths[idx % len(self.x_paths)]
        y_path = self.y_paths[random.randrange(len(self.y_paths))]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        return self.tf(x), self.tf(y)

# ------------------------------
# ViT Generator
# ------------------------------
class SinusoidalPositional2D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, h: int, w: int, device=None):
        device = device or DEVICE
        y = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
        x = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)
        omega = torch.arange(self.dim//4, device=device).float()
        omega = 1. / (10000 ** (omega / (self.dim//4)))
        out = []
        for coord in [x, y]:
            coord = coord.float().unsqueeze(-1)
            angles = coord * omega
            out.extend([torch.sin(angles), torch.cos(angles)])
        pe = torch.cat(out, dim=-1)
        return pe.view(1, h*w, self.dim)

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=nhead, batch_first=True, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class ViTGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, embed=192, depth=4, nhead=6, patch=8):
        super().__init__()
        self.patch = patch
        self.embed = embed
        self.to_tokens = nn.Conv2d(in_ch, embed, kernel_size=patch, stride=patch)
        self.posenc = SinusoidalPositional2D(embed)
        self.blocks = nn.ModuleList([TransformerBlock(embed, nhead=nhead, mlp_ratio=4.0) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed)
        self.conv_refine = nn.Sequential(
            nn.Conv2d(embed, embed, 3, 1, 1), nn.InstanceNorm2d(embed), nn.GELU(),
            nn.Conv2d(embed, embed//2, 3, 1, 1), nn.InstanceNorm2d(embed//2), nn.GELU(),
            nn.Conv2d(embed//2, out_ch, 1, 1, 0),
            nn.Tanh(),
        )
    def forward(self, x):
        B, _, H, W = x.shape
        assert H % self.patch == 0 and W % self.patch == 0, "H/W must be divisible by patch size"
        h, w = H // self.patch, W // self.patch
        t = self.to_tokens(x)                 # [B, C, h, w]
        t = t.flatten(2).transpose(1, 2)      # [B, L, C]
        t = t + self.posenc(h, w, device=x.device)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t).transpose(1, 2).view(B, self.embed, h, w)
        t = F.interpolate(t, scale_factor=self.patch, mode='bilinear', align_corners=False)
        y = self.conv_refine(t)
        return y

# ------------------------------
# PatchGAN Discriminator
# ------------------------------
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
    def forward(self, x): return self.net(x)

# ------------------------------
# Image Pool
# ------------------------------
class ImagePool:
    def __init__(self, pool_size=50): self.pool_size = pool_size; self.images: List[torch.Tensor] = []
    def query(self, imgs: torch.Tensor) -> torch.Tensor:
        if self.pool_size == 0: return imgs
        out = []
        for img in imgs:
            img = img.detach().unsqueeze(0)
            if len(self.images) < self.pool_size:
                self.images.append(img); out.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randrange(self.pool_size); tmp = self.images[idx].clone()
                    self.images[idx] = img; out.append(tmp)
                else:
                    out.append(img)
        return torch.cat(out, dim=0)

# ------------------------------
# Metrics: FID / LPIPS
# ------------------------------
class InceptionFID(nn.Module):
    """InceptionV3 (pool3 2048-d) feature extractor for FID."""
    def __init__(self):
        super().__init__()
        self.resize = transforms.Resize((299, 299))
        self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        w = models.Inception_V3_Weights.IMAGENET1K_V1
        # aux_logits는 True로! (weights와 호환)
        net = models.inception_v3(weights=w, aux_logits=True, transform_input=False)
        net.fc = nn.Identity()
        # 보조 분기(AuxLogits)도 무시되도록 아이덴티티로 대체 (안전)
        if hasattr(net, "AuxLogits"):
            net.AuxLogits = nn.Identity()
        self.net = net.eval().to(DEVICE)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1], shape [B,3,H,W]
        x = self.resize(x)
        x = self.norm(x)
        return self.net(x)  # [B,2048]

def _matrix_sqrt_symmetric_cpu(M: torch.Tensor) -> torch.Tensor:
    """Matrix square root for a symmetric PSD matrix on CPU using eigh."""
    M = M.cpu()
    # force exact symmetry to reduce numerical noise
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = torch.linalg.eigh(M)  # CPU
    eigvals = torch.clamp(eigvals, min=0)
    sqrt_eig = torch.diag(torch.sqrt(eigvals))
    return eigvecs @ sqrt_eig @ eigvecs.T

def fid_from_activations(real_act: torch.Tensor, fake_act: torch.Tensor) -> float:
    """
    real_act/fake_act: [N, 2048]. Inception features on any device.
    All FID linear algebra is done on CPU to avoid MPS unsupported ops.
    Uses the symmetric 'sandwich' form: sqrt(C1^{1/2} C2 C1^{1/2}).
    """
    # Move to CPU numpy for stable covariance
    r_np = real_act.float().cpu().numpy()
    f_np = fake_act.float().cpu().numpy()

    mu1 = torch.from_numpy(r_np.mean(axis=0)).float()
    mu2 = torch.from_numpy(f_np.mean(axis=0)).float()
    diff = mu1 - mu2

    c1 = torch.from_numpy(np.cov(r_np, rowvar=False)).float()
    c2 = torch.from_numpy(np.cov(f_np, rowvar=False)).float()

    # Numerical stability
    eps = 1e-6
    c1 = c1 + torch.eye(c1.shape[0]) * eps
    c2 = c2 + torch.eye(c2.shape[0]) * eps

    # symmetric sandwich: A = sqrt(C1), S = A C2 A, sqrt_S = sqrt(S)
    A = _matrix_sqrt_symmetric_cpu(c1)
    S = A @ c2 @ A
    sqrt_S = torch.from_numpy(scipy.linalg.sqrtm((A@c2@A).cpu().numpy()).real).float()

    fid = diff.dot(diff).item() + torch.trace(c1 + c2 - 2.0 * sqrt_S).item()
    # clamp tiny negatives due to numerical error
    return float(max(0.0, fid))

class LPIPSWrapper(nn.Module):
    """LPIPS if available; else VGG16 perceptual distance fallback."""
    def __init__(self):
        super().__init__()
        self.use_lpips = False
        try:
            import lpips  # type: ignore
            self.lpips = lpips.LPIPS(net='vgg').to(DEVICE).eval()
            self.use_lpips = True
        except Exception:
            v = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(DEVICE)
            for p in v.parameters(): p.requires_grad_(False)
            self.vgg = v
            self.norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x,y in [0,1], shape [B,3,H,W]
        if self.use_lpips:
            # lpips expects [-1,1]
            def to_m1p1(t): return t*2-1
            return self.lpips(to_m1p1(x), to_m1p1(y)).squeeze()  # [B]
        # fallback: simple VGG-feature L1 across few layers
        xs = self._vgg_feats(self.norm(x))
        ys = self._vgg_feats(self.norm(y))
        d = 0
        for a, b in zip(xs, ys):
            d = d + torch.mean(torch.abs(a - b), dim=[1,2,3])  # [B]
        return d / len(xs)

    def _vgg_feats(self, t: torch.Tensor):
        feats = []
        x = t
        # collect relu2_2, relu3_3, relu4_3 roughly
        capture_ids = [8, 15, 22]
        for i, m in enumerate(self.vgg):
            x = m(x)
            if i in capture_ids:
                feats.append(x)
        return feats

class Evaluator:
    def __init__(self, y_paths: List[str], size: int, eval_size: int, metrics: List[str]):
        self.metrics = metrics
        self.eval_size = eval_size
        self.size = size
        self.tf_x = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
        ])
        self.tf_tensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.y_paths = y_paths

        self.fid = InceptionFID() if 'fid' in metrics else None
        self.lpips = LPIPSWrapper() if 'lpips' in metrics else None

    @torch.no_grad()
    def _load_imgs(self, paths: List[str]) -> torch.Tensor:
        batch = []
        for p in paths:
            try:
                img = Image.open(p).convert('RGB')
                img = self.tf_x(img)
                img = self.tf_tensor(img)
                batch.append(img)
            except Exception:
                continue
        if not batch:
            raise RuntimeError("No images found for evaluation batch.")
        return torch.stack(batch, 0).to(DEVICE)

    @torch.no_grad()
    def evaluate(self, G_XY: nn.Module, x_paths: List[str]) -> dict:
        # sample X and Y subsets of same length
        random.shuffle(x_paths)
        random.shuffle(self.y_paths)
        N = min(self.eval_size, len(x_paths), len(self.y_paths))
        sub_x = x_paths[:N]
        sub_y = self.y_paths[:N]

        X = self._load_imgs(sub_x)                  # [N,3,H,W] in [0,1]
        Y = self._load_imgs(sub_y)                  # [N,3,H,W] in [0,1]
        # normalize to [-1,1] for generator
        X_in = (X*2-1)
        G_XY.eval()
        fakeY = G_XY(X_in)                           # [-1,1]
        fakeY01 = (fakeY.clamp(-1,1)*0.5 + 0.5)      # [0,1]

        out = {}
        if self.fid is not None:
            r_act = self.fid.features(Y)
            f_act = self.fid.features(fakeY01)
            out['FID'] = fid_from_activations(r_act, f_act)
        if self.lpips is not None:
            out['LPIPS'] = self.lpips(fakeY01, Y).mean().item()
        return out

# ------------------------------
# Loaders
# ------------------------------
def make_loader(ds: UnpairedImageDataset, batch: int, workers: int):
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers, drop_last=True,
                      pin_memory=(DEVICE == 'cuda'))

# ------------------------------
# Train / Test
# ------------------------------
def plot_curves(csv_path: str, out_scores: str, out_losses: str):
    # read csv and draw
    if not os.path.exists(csv_path):
        return
    epochs, Gs, DXs, DYs, cycs, idts, fids, lpips = [], [], [], [], [], [], [], []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            epochs.append(int(r['epoch']))
            Gs.append(float(r['G']))
            DXs.append(float(r['D_X']))
            DYs.append(float(r['D_Y']))
            cycs.append(float(r['cyc']))
            idts.append(float(r['idt']))
            fids.append(float(r.get('FID', 'nan')) if r.get('FID','') else float('nan'))
            lpips.append(float(r.get('LPIPS', 'nan')) if r.get('LPIPS','') else float('nan'))

    # scores.png (FID & LPIPS)
    plt.figure(figsize=(7,5))
    if any(not math.isnan(v) for v in fids):
        plt.plot(epochs, [v if not math.isnan(v) else None for v in fids], label='FID')
    if any(not math.isnan(v) for v in lpips):
        plt.plot(epochs, [v if not math.isnan(v) else None for v in lpips], label='LPIPS')
    plt.xlabel('Epoch'); plt.ylabel('Score (lower is better)'); plt.title('FID / LPIPS')
    plt.legend(); plt.tight_layout(); plt.savefig(out_scores); plt.close()

    # losses.png
    plt.figure(figsize=(7,5))
    plt.plot(epochs, Gs, label='G total')
    plt.plot(epochs, DXs, label='D_X')
    plt.plot(epochs, DYs, label='D_Y')
    plt.plot(epochs, cycs, label='cycle')
    plt.plot(epochs, idts, label='identity')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves')
    plt.legend(); plt.tight_layout(); plt.savefig(out_losses); plt.close()

def append_csv_row(csv_path: str, row: dict):
    # create csv with header if not exists
    header = ['epoch','G','D_X','D_Y','cyc','idt','FID','LPIPS']
    create = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header)
        if create: w.writeheader()
        # fill missing keys with ''
        for k in header:
            row.setdefault(k, '')
        w.writerow(row)

def train(args):
    seed_all(args.seed)

    ds = UnpairedImageDataset(args.data_x, args.data_y, size=args.size, max_x=args.max_x, max_y=args.max_y)
    dl = make_loader(ds, args.batch, args.workers)

    G_XY = ViTGenerator(embed=args.embed, depth=args.depth, nhead=args.nhead, patch=args.patch).to(DEVICE)
    G_YX = ViTGenerator(embed=args.embed, depth=args.depth, nhead=args.nhead, patch=args.patch).to(DEVICE)
    D_X = PatchDiscriminator().to(DEVICE)
    D_Y = PatchDiscriminator().to(DEVICE)

    G_XY.apply(weights_init); G_YX.apply(weights_init); D_X.apply(weights_init); D_Y.apply(weights_init)

    adv_loss = nn.MSELoss()
    l1 = nn.L1Loss()
    g_opt  = torch.optim.Adam(list(G_XY.parameters()) + list(G_YX.parameters()), lr=args.lr, betas=(0.5, 0.999))
    dx_opt = torch.optim.Adam(D_X.parameters(), lr=args.lr, betas=(0.5, 0.999))
    dy_opt = torch.optim.Adam(D_Y.parameters(), lr=args.lr, betas=(0.5, 0.999))

    def lambda_rule(epoch): return 1.0 - max(0, epoch - args.decay) / float(max(1, args.epochs - args.decay))
    g_sch  = torch.optim.lr_scheduler.LambdaLR(g_opt,  lr_lambda=lambda_rule)
    dx_sch = torch.optim.lr_scheduler.LambdaLR(dx_opt, lr_lambda=lambda_rule)
    dy_sch = torch.optim.lr_scheduler.LambdaLR(dy_opt, lr_lambda=lambda_rule)

    pool_X, pool_Y = ImagePool(50), ImagePool(50)

    ensure_dir(os.path.join(args.out_dir, 'samples'))
    ensure_dir(os.path.join(args.out_dir, 'checkpoints'))

    # Evaluator prep
    metrics = [m.strip().lower() for m in args.metrics.split(',') if m.strip()] if args.metrics else []
    evaluator = Evaluator(y_paths=ds.y_paths, size=args.size, eval_size=args.eval_size, metrics=metrics) if metrics else None
    csv_path = os.path.join(args.out_dir, 'metrics.csv')
    scores_png = os.path.join(args.out_dir, 'scores.png')
    losses_png = os.path.join(args.out_dir, 'losses.png')

    for epoch in range(1, args.epochs+1):
        G_XY.train(); G_YX.train(); D_X.train(); D_Y.train()
        epoch_loss_G = 0.0; epoch_loss_Dx = 0.0; epoch_loss_Dy = 0.0; epoch_cyc = 0.0; epoch_idt = 0.0
        n_batches = 0

        for i, (real_x, real_y) in enumerate(dl, 1):
            real_x = real_x.to(DEVICE); real_y = real_y.to(DEVICE)

            # ---- 1) Generators ----
            g_opt.zero_grad()

            fake_y = G_XY(real_x); rec_x = G_YX(fake_y)
            fake_x = G_YX(real_y); rec_y = G_XY(fake_x)

            idt_y = G_XY(real_y); idt_x = G_YX(real_x)

            pred_fake_y = D_Y(fake_y); pred_fake_x = D_X(fake_x)
            valid_y = torch.ones_like(pred_fake_y); valid_x = torch.ones_like(pred_fake_x)

            adv_XY = adv_loss(pred_fake_y, valid_y)
            adv_YX = adv_loss(pred_fake_x, valid_x)
            cyc    = l1(rec_x, real_x) + l1(rec_y, real_y)
            idt    = l1(idt_x, real_x) + l1(idt_y, real_y)

            g_total = adv_XY + adv_YX + args.lambda_cyc * cyc + args.lambda_idt * idt
            g_total.backward(); g_opt.step()

            # ---- 2) D_X ----
            dx_opt.zero_grad()
            pred_real_x = D_X(real_x); valid_dx = torch.ones_like(pred_real_x)
            fake_x_pool = pool_X.query(fake_x.detach()); pred_fake_x_pool = D_X(fake_x_pool); fake_dx = torch.zeros_like(pred_fake_x_pool)
            loss_dx = 0.5 * (adv_loss(pred_real_x, valid_dx) + adv_loss(pred_fake_x_pool, fake_dx))
            loss_dx.backward(); dx_opt.step()

            # ---- 3) D_Y ----
            dy_opt.zero_grad()
            pred_real_y = D_Y(real_y); valid_dy = torch.ones_like(pred_real_y)
            fake_y_pool = pool_Y.query(fake_y.detach()); pred_fake_y_pool = D_Y(fake_y_pool); fake_dy = torch.zeros_like(pred_fake_y_pool)
            loss_dy = 0.5 * (adv_loss(pred_real_y, valid_dy) + adv_loss(pred_fake_y_pool, fake_dy))
            loss_dy.backward(); dy_opt.step()

            # accumulate
            epoch_loss_G  += g_total.item()
            epoch_loss_Dx += loss_dx.item()
            epoch_loss_Dy += loss_dy.item()
            epoch_cyc     += cyc.item()
            epoch_idt     += idt.item()
            n_batches     += 1

            if i % args.print_every == 0:
                print(f"[E{epoch:03d}/{args.epochs}] [B{i:04d}/{len(dl)}] "
                      f"G:{g_total.item():.3f} D_X:{loss_dx.item():.3f} D_Y:{loss_dy.item():.3f} "
                      f"cyc:{cyc.item():.3f} id:{idt.item():.3f}")

        g_sch.step(); dx_sch.step(); dy_sch.step()

        # Save samples
        with torch.no_grad():
            G_XY.eval(); G_YX.eval()
            sample_x = [ds.x_paths[k % len(ds.x_paths)] for k in range(4)]
            sample_y = [ds.y_paths[k % len(ds.y_paths)] for k in range(4)]
            tf = transforms.Compose([
                transforms.Resize(args.size),
                transforms.CenterCrop(args.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3),
            ])
            def batch(paths):
                ims = []
                for p in paths:
                    try: ims.append(tf(Image.open(p).convert('RGB')))
                    except: pass
                if not ims: ims = [torch.randn(3, args.size, args.size)]
                return torch.stack(ims,0).to(DEVICE)
            rx, ry = batch(sample_x), batch(sample_y)
            fy, fx = G_XY(rx), G_YX(ry)
            rcx, rcy = G_YX(fy), G_XY(fx)
            grid = make_grid(torch.cat([rx, fy, rcx, ry, fx, rcy], 0),
                             nrow=4, normalize=True, value_range=(-1,1))
            save_image(grid, os.path.join(args.out_dir, "samples", f"epoch_{epoch:03d}.png"))

        # ---- Evaluation + CSV/plots ----
        avg_row = {
            'epoch': epoch,
            'G': epoch_loss_G / max(1, n_batches),
            'D_X': epoch_loss_Dx / max(1, n_batches),
            'D_Y': epoch_loss_Dy / max(1, n_batches),
            'cyc': epoch_cyc / max(1, n_batches),
            'idt': epoch_idt / max(1, n_batches),
        }

        if evaluator and (epoch % max(1, args.eval_every) == 0):
            try:
                metrics_out = evaluator.evaluate(G_XY, ds.x_paths)
                for k, v in metrics_out.items():
                    avg_row[k.upper()] = float(v)
                print(f"[Eval@E{epoch}] " + " ".join([f"{k}:{v:.3f}" for k,v in metrics_out.items()]))
            except Exception as e:
                print(f"[Eval skipped @E{epoch}] {e}")

        append_csv_row(csv_path, avg_row)
        plot_curves(csv_path, scores_png, losses_png)

        # Save checkpoints
        torch.save(G_XY.state_dict(), os.path.join(args.out_dir,'checkpoints','G_XY_latest.pt'))
        torch.save(G_YX.state_dict(), os.path.join(args.out_dir,'checkpoints','G_YX_latest.pt'))
        torch.save(D_X.state_dict(),  os.path.join(args.out_dir,'checkpoints','D_X_latest.pt'))
        torch.save(D_Y.state_dict(),  os.path.join(args.out_dir,'checkpoints','D_Y_latest.pt'))

@torch.no_grad()
def test(args):
    size = args.size
    tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    G_XY = ViTGenerator(embed=args.embed, depth=args.depth, nhead=args.nhead, patch=args.patch).to(DEVICE)
    ckpt = torch.load(args.gen_ckpt, map_location='cpu')
    G_XY.load_state_dict(ckpt, strict=True); G_XY.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    exts = (".png",".jpg",".jpeg",".webp")
    paths: List[str] = []
    if os.path.isdir(args.in_dir):
        for e in exts: paths += glob.glob(os.path.join(args.in_dir, f"**/*{e}"), recursive=True)
    else:
        paths = [args.in_dir] if any(args.in_dir.lower().endswith(x) for x in exts) else []
    if not paths: raise RuntimeError(f"No images in: {args.in_dir}")

    for p in paths:
        img = Image.open(p).convert('RGB'); x = tf(img).unsqueeze(0).to(DEVICE)
        y = G_XY(x); y = denorm(y).cpu()
        out_path = os.path.join(args.out_dir, os.path.basename(p))
        save_image(y, out_path)
    print(f"Saved translated images -> {args.out_dir}")

# ------------------------------
# Main
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, choices=['train','test'], required=True)
    # data & io
    ap.add_argument('--data_x', type=str, help='X domain root (e.g., AFHQ/photos)')
    ap.add_argument('--data_y', type=str, help='Y domain root (e.g., pokemon)')
    ap.add_argument('--out_dir', type=str, default='runs/vit_cyclegan')
    ap.add_argument('--in_dir', type=str, help='input folder or file (test)')
    ap.add_argument('--gen_ckpt', type=str, help='G_XY checkpoint (test)')
    # training
    ap.add_argument('--size', type=int, default=96, help='image size (must be multiple of patch)')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--decay', type=int, default=3)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--lambda_cyc', type=float, default=10.0)
    ap.add_argument('--lambda_idt', type=float, default=5.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--print_every', type=int, default=50)
    ap.add_argument('--max_x', type=int, default=0, help='random subset for X (0=all)')
    ap.add_argument('--max_y', type=int, default=0, help='random subset for Y (0=all)')
    # ViT hyperparams
    ap.add_argument('--embed', type=int, default=192)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--nhead', type=int, default=6)
    ap.add_argument('--patch', type=int, default=8)
    # metrics
    ap.add_argument('--metrics', type=str, default='fid,lpips', help='comma-separated: fid,lpips or empty')
    ap.add_argument('--eval_every', type=int, default=1, help='evaluate every N epochs')
    ap.add_argument('--eval_size', type=int, default=100, help='#images for eval (X and Y each)')
    args = ap.parse_args()

    if args.mode == 'train':
        if not args.data_x or not args.data_y:
            raise SystemExit('--data_x and --data_y are required in train mode')
        if args.size % args.patch != 0:
            raise SystemExit('--size must be divisible by --patch (e.g., size=96, patch=8)')
        os.makedirs(args.out_dir, exist_ok=True)
        train(args)
    else:
        if not args.gen_ckpt or not args.in_dir or not args.out_dir:
            raise SystemExit('--gen_ckpt, --in_dir, --out_dir are required in test mode')
        if args.size % args.patch != 0:
            raise SystemExit('--size must be divisible by --patch (e.g., size=96, patch=8)')
        test(args)

if __name__ == '__main__':
    main()
