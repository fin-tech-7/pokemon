# -*- coding: utf-8 -*-
# filename: vit2wplus_stylegan_mps.py
"""
MPS-ONLY: ViT -> W+ 직접 회귀 → StyleGAN2-ADA 생성 (포켓몬풍)

필수:
  pip install timm torchvision pillow numpy tqdm
  (stylegan2-ada-pytorch 소스 폴더가 현재 경로에 존재해야 함)

예시 1-1) 셀프-인코더 사전학습 (G 고정, 가짜 (y,w+) 쌍으로 E 학습)
python vit2wplus_stylegan_mps.py \
  --mode train_encoder \
  --g_pkl ./checkpoints/pretrained.pkl \
  --out_pth ./checkpoints/tuning_ffhq2/vit2wplus_self.pth \
  --steps 20000 --batch 16 --lr 1e-4
  
예시 2-2) 추론 (현실 이미지 → ViT로 w+ 회귀 → G.synthesis)
python vit2wplus_stylegan_mps.py \
  --mode infer \
  --g_pkl ./checkpoints/pretrained.pkl \
  --enc_pth ./checkpoints/tuning_ffhq2/vit2wplus_self.pth \
  --in_dir ./Animal_Pose/keeps \
  --out_dir ./runs/poke_out/tuning_ffhq2/2 \
  --pad_square 1 --psi 1.0

예시 1) 셀프-인코더 사전학습 (G 고정, 가짜 (y,w+) 쌍으로 E 학습)
python vit2wplus_stylegan_mps.py \
  --mode train_encoder \
  --g_pkl ./checkpoints/network-snapshot-001098.pkl \
  --out_pth ./checkpoints/tuning/vit2wplus_self.pth \
  --steps 20000 --batch 16 --lr 1e-4

예시 2) 추론 (현실 이미지 → ViT로 w+ 회귀 → G.synthesis)
python vit2wplus_stylegan_mps.py \
  --mode infer \
  --g_pkl ./checkpoints/network-snapshot-001098.pkl \
  --enc_pth ./checkpoints/tuning/vit2wplus_self.pth \
  --in_dir ./Animal_Pose/keeps \
  --out_dir ./runs/poke_out/tuning/2 \
  --pad_square 1 --psi 1.0

주의:
- 이 스크립트는 MPS만 지원합니다. (assert로 강제)
- LPIPS/CPU 폴백/force_fp32 없음. 오로지 MPS + autocast(fp16)로만 동작.
"""

import os, sys, argparse, glob, random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils as vutils
from tqdm import tqdm
import timm  # ViT backbone

VIT_SIZE = 224

# --------- MPS 전용 ----------
def get_mps_device():
    assert getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(), \
        "MPS가 활성화되지 않았습니다. (macOS + Apple Silicon + PyTorch MPS 필요)"
    return torch.device("mps")

DEVICE = get_mps_device()
torch.set_float32_matmul_precision("high")

# --------- StyleGAN2-ADA ----------
sys.path.append("./stylegan2-ada-pytorch")
import dnnlib, legacy  # noqa: E402

# --------- Utils ----------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=DEVICE)[:, None, None]
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=DEVICE)[:, None, None]

def seed_all(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def safe_mkdir(p: str): os.makedirs(p, exist_ok=True)

def list_images(root: str, exts=(".png",".jpg",".jpeg",".webp",".bmp",".tif",".tiff")) -> List[str]:
    paths: List[str] = []
    for e in exts: paths += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)
    return sorted(paths)

def denorm(x: torch.Tensor) -> torch.Tensor:  # [-1,1] -> [0,1]
    return (x.clamp(-1,1) * 0.5 + 0.5)

def to_vit_norm(x01: torch.Tensor) -> torch.Tensor:  # [0,1] -> ImageNet norm
    return (x01 - IMAGENET_MEAN) / IMAGENET_STD

def lowpass(x: torch.Tensor, k: int = 4) -> torch.Tensor:
    h, w = x.shape[-2:]
    y = F.avg_pool2d(x, kernel_size=k, stride=k)
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)

def pad_square(img: Image.Image, size: int, fill=(255,255,255)) -> Image.Image:
    img = img.convert("RGB"); w, h = img.size; s = size / max(w, h)
    new = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(new, ((size - new.size[0])//2, (size - new.size[1])//2))
    return canvas

def center_resize_crop(img: Image.Image, size: int) -> Image.Image:
    tfm = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(size),
    ])
    return tfm(img.convert("RGB"))

def pil_to_tensor01(img: Image.Image) -> torch.Tensor:
    return transforms.ToTensor()(img)  # [0,1]

# --------- StyleGAN helpers ----------
@torch.no_grad()
def load_generator(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = legacy.load_network_pkl(f)
    G = data.get("G_ema", None) or data.get("G", None)
    if G is None: raise KeyError(f"G_ema/G not found in: {pkl_path}")
    return G.to(DEVICE).eval().requires_grad_(False)

def sg_shapes(G) -> Tuple[int,int,int]:
    num_ws = getattr(G, "num_ws", None) or G.synthesis.num_ws
    w_dim  = getattr(G, "w_dim",  None) or G.synthesis.w_dim
    res    = getattr(G, "img_resolution", None) or 512
    return int(num_ws), int(w_dim), int(res)

@torch.no_grad()
def sample_wplus(G, n: int, trunc_sigma: float = 0.7) -> torch.Tensor:
    num_ws, w_dim, _ = sg_shapes(G)
    w_avg = G.mapping.w_avg[None, None, :].to(DEVICE)
    return w_avg + torch.randn(n, num_ws, w_dim, device=DEVICE) * trunc_sigma

def _fix_ws(ws, G):
    num_ws = getattr(G, "num_ws", None) or G.synthesis.num_ws
    w_dim  = getattr(G, "w_dim",  None) or G.synthesis.w_dim
    if ws.ndim == 2: ws = ws.unsqueeze(1).repeat(1, num_ws, 1)
    if ws.shape[1] != num_ws and ws.shape[-1] == num_ws:
        ws = ws.transpose(1, 2).contiguous()
    return ws

def sg_synth_fp32(G, ws, noise_mode="const", psi: float = 1.0):
    """합성만 fp32로 강제 (MPS 안정화)."""
    ws = _fix_ws(ws, G).to(torch.float32, copy=False)
    with torch.autocast(device_type="mps", dtype=torch.float16, enabled=False):
        if psi == 1.0: out = G.synthesis(ws, noise_mode=noise_mode)
        else:          out = G(ws=ws, truncation_psi=psi, noise_mode=noise_mode)
    return out

# --------- ViT Encoder (ViT -> W+) ----------
class ViT2WPlus(nn.Module):
    def __init__(self, num_ws: int, w_dim: int,
                 backbone: str = "vit_base_patch16_224",
                 freeze_backbone: bool = True,
                 hidden: int = 2048,
                 vit_size: int = 224,
                 pretrained_backbone: bool = True):
        super().__init__()
        self.num_ws = num_ws; self.w_dim = w_dim; self.vit_size = vit_size
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained_backbone,
            num_classes=0, global_pool="avg", img_size=vit_size
        )
        emb = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(emb),
            nn.Linear(emb, hidden), nn.GELU(),
            nn.Linear(hidden, num_ws * w_dim)
        )
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False

    def forward(self, x01_normed: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x01_normed)          # [B, C]
        out   = self.head(feats).view(-1, self.num_ws, self.w_dim)
        return out

# --------- (y,w+) 가짜 배치 샘플러 ----------
def make_fake_batch(G, B: int,
                    psi_range=(0.7, 1.0),
                    p_stylemix: float = 0.9,
                    jitter_sigma: float = 0.0):
    device = DEVICE
    num_ws = getattr(G, "num_ws", None) or G.synthesis.num_ws
    z1 = torch.randn(B, G.z_dim, device=device)
    ws1 = G.mapping(z1, None)        # [B,num_ws,w_dim]
    ws = ws1.clone()

    # style-mix
    do_mix = torch.rand(B, device=device) < p_stylemix
    if do_mix.any():
        B2 = int(do_mix.sum().item())
        z2 = torch.randn(B2, G.z_dim, device=device)
        ws2 = G.mapping(z2, None)
        cuts = torch.randint(1, num_ws, (B2,), device=device)
        idxs = do_mix.nonzero(as_tuple=False).flatten()
        for k,i in enumerate(idxs.tolist()):
            c = int(cuts[k].item()); ws[i, c:, :] = ws2[k, c:, :]

    if jitter_sigma > 0: ws = ws + torch.randn_like(ws) * jitter_sigma

    # truncation
    w_avg = G.mapping.w_avg[None, None, :].to(device)
    psi = torch.empty(B, 1, 1, device=device).uniform_(*psi_range)
    ws = w_avg + psi * (ws - w_avg)

    with torch.no_grad():
        y = sg_synth_fp32(G, ws, noise_mode="const").clamp(-1, 1)
    return ws.detach(), y.detach()

def moments_loss(w_hat, w_tgt):
    x = w_hat.float(); t = w_tgt.float()
    mu_x = x.mean(dim=(0,1));  sd_x = x.std(dim=(0,1)) + 1e-6
    mu_t = t.mean(dim=(0,1));  sd_t = t.std(dim=(0,1)) + 1e-6
    return (mu_x - mu_t).abs().mean() + (sd_x.log() - sd_t.log()).abs().mean()

# --------- 학습 ----------
def train_encoder(
    g_pkl: str, out_pth: str,
    steps: int = 30000, batch: int = 16, lr: float = 1e-4,
    backbone: str = "vit_base_patch16_224", freeze_backbone: bool = True,
    loss_w: float = 1.0, loss_pix: float = 0.2,
    save_every: int = 10, seed: int = 42
):
    seed_all(seed)
    save_dir = os.path.dirname(out_pth) or "."
    save_stem = os.path.splitext(os.path.basename(out_pth))[0]
    os.makedirs(save_dir, exist_ok=True)

    G = load_generator(g_pkl)
    num_ws, w_dim, _ = sg_shapes(G)

    E = ViT2WPlus(num_ws, w_dim, backbone=backbone,
                  freeze_backbone=freeze_backbone, vit_size=VIT_SIZE).to(DEVICE)
    opt = torch.optim.AdamW(E.parameters(), lr=lr, betas=(0.9,0.99), weight_decay=1e-2)

    # 앞 레이어 약간 가중
    idx = torch.arange(num_ws, device=DEVICE, dtype=torch.float32)
    alpha = 0.13
    layer_w = 1.0 + alpha * (1.0 - idx / max(num_ws - 1, 1))
    layer_w = (layer_w / layer_w.mean()).view(1, num_ws, 1)

    pbar = tqdm(range(steps), desc="[Train Encoder:MPS]")
    for it in pbar:
        w_tgt, y = make_fake_batch(G, B=batch, psi_range=(0.7,1.0), p_stylemix=0.9, jitter_sigma=0.01)
        y01 = denorm(y)
        y01_resized = F.interpolate(y01, size=(VIT_SIZE, VIT_SIZE), mode="bilinear", align_corners=False)
        y_vit = to_vit_norm(y01_resized)

        with torch.autocast(device_type="mps", dtype=torch.float16):
            w_hat = E(y_vit)

        y_hat = sg_synth_fp32(G, w_hat, noise_mode="const").clamp(-1, 1)

        with torch.autocast(device_type="mps", dtype=torch.float16):
            w_err = (w_hat.float() - w_tgt.float()).pow(2)
            lw = (w_err.mean(dim=2) * layer_w.squeeze(-1)).mean() * loss_w
            lp = F.l1_loss(lowpass(y_hat.float()), lowpass(y.float())) * loss_pix
            lm = 0.123 * moments_loss(w_hat, w_tgt)
            loss = lw + lp + lm

        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        pbar.set_postfix(loss=float(loss.item()))

        if (it + 1) % save_every == 0 or (it + 1) == steps:
            step_id = it + 1
            ckpt_path = os.path.join(save_dir, f"{save_stem}_{step_id:06d}.pth")
            torch.save({"state_dict": E.state_dict(), "step": step_id}, ckpt_path)
            torch.save({"state_dict": E.state_dict(), "step": step_id}, out_pth)
            with torch.no_grad(): grid = denorm(y_hat[:8]).cpu()
            vutils.save_image(grid, os.path.join(save_dir, f"train_vis_{step_id:06d}.png"), nrow=4)
            ratio = float(w_hat.float().std() / (w_tgt.float().std() + 1e-6))
            print(f"[CKPT] {ckpt_path}  (latest -> {out_pth})  [std_ratio={ratio:.2f}]")

# --------- 추론 ----------
@torch.no_grad()
def infer(
    g_pkl: str, enc_pth: str, in_dir: str, out_dir: str,
    size: int = 224, pad_square_flag: int = 1, psi: float = 1.0,
    max_in: int = 0, seed: int = 42
):
    seed_all(seed); safe_mkdir(out_dir)
    size = VIT_SIZE  # 해상도 통일

    G = load_generator(g_pkl)
    num_ws, w_dim, res = sg_shapes(G)

    # 경고 억제: infer에서는 timm 프리트레인 로드 안 함
    E = ViT2WPlus(num_ws, w_dim,
                  backbone="vit_base_patch16_224",
                  freeze_backbone=True,
                  vit_size=VIT_SIZE,
                  pretrained_backbone=False).to(DEVICE)
    ckpt = torch.load(enc_pth, map_location=DEVICE)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    E.load_state_dict(state, strict=True); E.eval()

    paths = list_images(in_dir)
    if not paths: raise ValueError(f"No images in: {in_dir}")
    total = len(paths)
    if max_in > 0 and total > max_in:
        rng = random.Random(seed); paths = rng.sample(paths, k=max_in)

    print(f"[Info] G.res={res}, num_ws={num_ws}, w_dim={w_dim}, #images={len(paths)} (picked from {total})")

    for pth in paths:
        try:
            img = Image.open(pth).convert("RGB")
            im = pad_square(img, size=size) if pad_square_flag else center_resize_crop(img, size=size)
            x01 = pil_to_tensor01(im).unsqueeze(0).to(DEVICE)        # [1,3,224,224], [0,1]
            x_vit = to_vit_norm(x01)
        except Exception as e:
            print(f"[Skip] {pth} ({e})"); continue

        with torch.autocast(device_type="mps", dtype=torch.float16):
            w_hat = E(x_vit)                                         # [1,num_ws,w_dim]
        y = sg_synth_fp32(G, w_hat, noise_mode="const", psi=psi).clamp(-1, 1)
        out = denorm(y[0]).cpu()

        base = os.path.splitext(os.path.basename(pth))[0]
        vutils.save_image(out, os.path.join(out_dir, f"{base}_pokemon.png"))
        print(f"[OK] saved -> {os.path.join(out_dir, f'{base}_pokemon.png')}")

# --------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, required=True, choices=["train_encoder","infer"])
    ap.add_argument("--g_pkl", type=str, required=True)
    ap.add_argument("--out_pth", type=str, help="train_encoder 저장 경로")
    ap.add_argument("--enc_pth", type=str, help="infer 인코더 가중치 경로")
    ap.add_argument("--in_dir", type=str, help="infer 입력 폴더")
    ap.add_argument("--out_dir", type=str, help="infer 출력 폴더")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--backbone", type=str, default="vit_base_patch16_224")
    ap.add_argument("--freeze_backbone", type=int, default=1)
    ap.add_argument("--loss_w", type=float, default=1.0)
    ap.add_argument("--loss_pix", type=float, default=0.2)
    ap.add_argument("--save_every", type=int, default=10)
    ap.add_argument("--pad_square", type=int, default=1)
    ap.add_argument("--psi", type=float, default=1.0)
    ap.add_argument("--max_in", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--vit_size", type=int, default=224)
    args = ap.parse_args()

    # 전역 VIT_SIZE 동기화
    global VIT_SIZE
    VIT_SIZE = args.vit_size

    _ = get_mps_device()  # MPS 강제

    if args.mode == "train_encoder":
        assert args.out_pth, "--out_pth required"
        train_encoder(
            g_pkl=args.g_pkl, out_pth=args.out_pth,
            steps=args.steps, batch=args.batch, lr=args.lr,
            backbone=args.backbone, freeze_backbone=bool(args.freeze_backbone),
            loss_w=args.loss_w, loss_pix=args.loss_pix,
            save_every=args.save_every, seed=args.seed
        )
    else:
        assert args.enc_pth and args.in_dir and args.out_dir, "--enc_pth, --in_dir, --out_dir required"
        infer(
            g_pkl=args.g_pkl, enc_pth=args.enc_pth,
            in_dir=args.in_dir, out_dir=args.out_dir,
            size=args.size, pad_square_flag=args.pad_square,
            psi=args.psi, max_in=args.max_in, seed=args.seed
        )

if __name__ == "__main__":
    main()
