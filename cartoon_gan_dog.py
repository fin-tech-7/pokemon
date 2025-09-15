import os, glob, csv, random, argparse
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image, make_grid
from PIL import Image
import numpy as np
from scipy import linalg

# ------------------------------
# Device
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

def denorm(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp_(-1, 1) * 0.5 + 0.5)

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

# ------------------------------
# Dataset
# ------------------------------
class ImageDataset(Dataset):
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
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5]),
        ])

    def _collect(self, root: str):
        exts = (".png", ".jpg", ".jpeg", ".webp")
        paths = []
        if os.path.isdir(root):
            for e in exts:
                paths += glob.glob(os.path.join(root, f"**/*{e}"), recursive=True)
        else:
            if any(root.lower().endswith(x) for x in exts):
                paths = [root]
        return sorted(paths)

    def __len__(self):
        return min(len(self.x_paths), len(self.y_paths))

    def __getitem__(self, idx: int):
        x_path = self.x_paths[idx % len(self.x_paths)]
        y_path = self.y_paths[idx % len(self.y_paths)]
        x = Image.open(x_path).convert('RGB')
        y = Image.open(y_path).convert('RGB')
        return self.tf(x), self.tf(y)

# ------------------------------
# CartoonGAN Generator & Discriminator
# ------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
        )
    def forward(self, x):
        return x + self.block(x)

class CartoonGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=8):
        super().__init__()
        model = [
            nn.Conv2d(in_ch, ngf, 7, 1, 3), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, 2, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, 2, 1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            model += [ResidualBlock(ngf*4)]
        model += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, 2, 1, 1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, 2, 1, 1), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, out_ch, 7, 1, 3), nn.Tanh()
        ]
        self.net = nn.Sequential(*model)
    def forward(self, x):
        return self.net(x)

class CartoonDiscriminator(nn.Module):
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
# Metrics (FID & LPIPS)
# ------------------------------
class InceptionV3Feat(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            net = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            net.fc = nn.Identity()
            self.net = net.eval().to(DEVICE)
            for p in self.net.parameters(): p.requires_grad_(False)
            self.ok = True
        except Exception as e:
            print(f"[WARN] InceptionV3 not available: {e}")
            self.ok = False
    @torch.no_grad()
    def forward(self, imgs: torch.Tensor):
        if not self.ok: return None
        x = denorm(imgs)
        x = F.interpolate(x, size=(299,299), mode="bilinear", align_corners=False)
        return self.net(x)

def _cov_mean(feats: np.ndarray):
    mu = np.mean(feats, axis=0)
    feats_centered = feats - mu
    cov = feats_centered.T @ feats_centered / (feats.shape[0]-1)
    return cov, mu

def _sqrtm_psd(mat: np.ndarray):
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0, None)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T

def fid_score(f1: np.ndarray, f2: np.ndarray):
    c1,m1 = _cov_mean(f1); c2,m2 = _cov_mean(f2)
    diff = m1 - m2

    # covmean 계산 (수치 안정성)
    covmean, _ = linalg.sqrtm(c1 @ c2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(c1 + c2 - 2*covmean)
    return float(max(fid, 0.0))  # 음수 방지

class VGG16LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]])
        for p in self.parameters(): p.requires_grad_(False)
        self.blocks = self.blocks.eval().to(DEVICE)
        self.ok = True
        self.to_imagenet = transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225],
        )
    def _norm(self,x): return self.to_imagenet(denorm(x))
    @torch.no_grad()
    def forward(self,x,y):
        if not self.ok: return None
        x = F.interpolate(x.to(DEVICE), size=(224,224), mode="bilinear", align_corners=False)
        y = F.interpolate(y.to(DEVICE), size=(224,224), mode="bilinear", align_corners=False)
        x,y = self._norm(x), self._norm(y)
        dist=0.0
        for blk in self.blocks:
            x,y = blk(x), blk(y)
            nx,ny = F.normalize(x,p=2,dim=1), F.normalize(y,p=2,dim=1)
            dist += F.mse_loss(nx,ny)
        return dist

@torch.no_grad()
def evaluate_metrics(G, ds: ImageDataset, eval_n: int, incep: InceptionV3Feat, lpips_vgg: VGG16LPIPS):
    G.eval()
    xs, ys = [],[]
    rng = random.Random(1234)
    for _ in range(eval_n):
        x_path = rng.choice(ds.x_paths); y_path = rng.choice(ds.y_paths)
        x = ds.tf(Image.open(x_path).convert('RGB'))
        y = ds.tf(Image.open(y_path).convert('RGB'))
        xs.append(x); ys.append(y)
    xs,ys = torch.stack(xs).to(DEVICE), torch.stack(ys).to(DEVICE)
    ys_fake = G(xs)

    fid_val=float("nan"); lpips_val=float("nan")
    if incep.ok:
        f1=incep(ys_fake).cpu().numpy(); f2=incep(ys).cpu().numpy()
        fid_val=max(fid_score(f1,f2), 0.0)
    if lpips_vgg.ok:
        lp=lpips_vgg(ys_fake,ys)
        lpips_val=lp.item()
    return fid_val, lpips_val

# ------------------------------
# Training (with Resume + Anti-collapse)
# ------------------------------
def train(args):
    seed_all(args.seed)
    ds = ImageDataset(args.data_x, args.data_y, args.size, args.max_x, args.max_y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True)

    G = CartoonGenerator().to(DEVICE)
    D = CartoonDiscriminator().to(DEVICE)

    adv_loss = nn.MSELoss()
    l1 = nn.L1Loss()

    g_opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))

    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        G.load_state_dict(ckpt["G"])
        D.load_state_dict(ckpt["D"])
        g_opt.load_state_dict(ckpt["g_opt"])
        d_opt.load_state_dict(ckpt["d_opt"])
        # ✅ 사용자가 직접 지정한 epoch 우선
        if args.resume_epoch > 0:
            start_epoch = args.resume_epoch
            print(f"[Resume] Loaded checkpoint from {args.resume}, 강제 epoch {start_epoch}부터 시작")
        else:
            start_epoch = ckpt["epoch"] + 1
            print(f"[Resume] Loaded checkpoint from {args.resume}, epoch {ckpt['epoch']} → {start_epoch}부터 시작")

    safe_mkdir(args.out_dir)
    sample_dir = os.path.join(args.out_dir,"samples")
    ckpt_dir = os.path.join(args.out_dir,"checkpoints")
    safe_mkdir(sample_dir); safe_mkdir(ckpt_dir)

    csv_path = os.path.join(args.out_dir,"metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path,"w",newline="") as f:
            w=csv.writer(f); w.writerow(["epoch","g_total","d_loss","content","fid","lpips","lr"])

    incep = InceptionV3Feat(); lpips_vgg=VGG16LPIPS()

    for epoch in range(start_epoch, args.epochs+1):
        G.train(); D.train()
        running={"g":0,"d":0,"c":0,"n":0}

        for i,(x,y) in enumerate(dl,1):
            x,y=x.to(DEVICE),y.to(DEVICE)

            # Input noise
            noise = torch.randn_like(x) * 0.05
            x_noisy = (x + noise).clamp(-1, 1)

            # ---- Train D ----
            d_opt.zero_grad()
            fake_y=G(x_noisy).detach()
            pred_real=D(y); pred_fake=D(fake_y)

            real_label = torch.full_like(pred_real, 0.9) # label smoothing
            fake_label = torch.full_like(pred_fake, 0.0)

            loss_real=adv_loss(pred_real, real_label)
            loss_fake=adv_loss(pred_fake, fake_label)
            d_loss=0.5*(loss_real+loss_fake)
            d_loss.backward(); d_opt.step()

            # ---- Train G ----
            g_opt.zero_grad()
            fake_y=G(x_noisy)
            adv=adv_loss(D(fake_y), torch.ones_like(D(fake_y))*0.9) # label smoothing
            vgg=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23].to(DEVICE).eval()
            for p in vgg.parameters(): p.requires_grad_(False)
            feat_fake=vgg(fake_y); feat_real=vgg(x)
            content=l1(feat_fake,feat_real)
            g_total=adv+args.lambda_content*content
            g_total.backward(); g_opt.step()

            running["g"]+=g_total.item()
            running["d"]+=d_loss.item()
            running["c"]+=content.item()
            running["n"]+=1

        # ✅ 에폭 끝난 뒤 평균 출력
        n = max(1, running["n"])
        g_avg = running["g"]/n
        d_avg = running["d"]/n
        c_avg = running["c"]/n
        print(f"[Epoch {epoch}/{args.epochs}] G:{g_avg:.3f} D:{d_avg:.3f} C:{c_avg:.3f}")

        # Metrics
        fid,lpips=evaluate_metrics(G,ds,args.eval_num,incep,lpips_vgg)
        
        if epoch % args.sample_every == 0:
            with torch.no_grad():
                x, y = next(iter(dl))
                x, y = x.to(DEVICE), y.to(DEVICE)
                fake = G(x)

                grid = make_grid(
                    torch.cat([x, fake], 0).cpu(),
                    nrow=args.batch,
                    normalize=True, value_range=(-1,1)
                )
                save_image(grid, os.path.join(sample_dir, f"epoch_{epoch:03d}.png"))

        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "g_opt": g_opt.state_dict(),
            "d_opt": d_opt.state_dict(),
        }, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt"))

        with open(csv_path,"a",newline="") as f:
            w=csv.writer(f)
            w.writerow([epoch,f"{g_avg:.6f}",f"{d_avg:.6f}",f"{c_avg:.6f}",
                        f"{fid:.6f}",f"{lpips:.6f}",f"{g_opt.param_groups[0]['lr']:.8f}"])

def main():
    class Args:
        mode = "train"
        data_x = "./cutoutface/dog"
        data_y = "./pokemon/Cartoon_Cropped"
        out_dir = "./cartoon_runs/cartoon_dog2"
        size = 128
        epochs = 200
        batch = 4
        lr = 2e-4
        lambda_content = 5.0
        seed = 42
        workers = 2
        max_x = 0
        max_y = 0
        eval_num = 32
        sample_every = 1
        resume = "./cartoon_runs/cartoon_dog2/checkpoints/epoch_058.pt"
        resume_epoch = 59  # ✅ 다음 epoch부터 시작

    args = Args()
    if args.mode == "train":
        train(args)

if __name__=="__main__":
    main()


