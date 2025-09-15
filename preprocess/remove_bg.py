from rembg import remove
from PIL import Image
from pathlib import Path
from tqdm import tqdm

in_root = Path("animal-faces")   # 입력 폴더
out_root = Path("cutout")        # 출력 폴더
out_root.mkdir(parents=True, exist_ok=True)

paths = list(in_root.rglob("*.jpg")) + list(in_root.rglob("*.png"))
for p in tqdm(paths):
    try:
        img = Image.open(p).convert("RGBA")
        cut = remove(img)  # 배경 제거
        out_dir = out_root / p.parent.relative_to(in_root)
        out_dir.mkdir(parents=True, exist_ok=True)
        cut.save(out_dir / (p.stem + ".png"))
    except Exception as e:
        print("skip:", p, e)
