import os
from PIL import Image

def crop_top_center(img, top_ratio=0.1, height_ratio=0.6, size=128):
    """
    포켓몬 얼굴/상체 위주로 상단 중앙 부분 크롭
    - top_ratio: 위쪽 몇 % 잘라낼지 (0.1 = 위쪽 10% 버림)
    - height_ratio: 전체 높이에서 사용할 비율 (0.6 = 위쪽 60% 남김)
    - size: 최종 저장 크기 (정사각형)
    """
    img = img.convert("RGBA")  # 투명 배경 유지
    w, h = img.size
    top = int(h * top_ratio)
    crop_h = int(h * height_ratio)
    left = 0
    right = w
    bottom = top + crop_h
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((size, size), Image.LANCZOS)

def process_folder(in_dir, out_dir, size=128, top_ratio=0.1, height_ratio=0.6):
    os.makedirs(out_dir, exist_ok=True)
    exts = (".png",".jpg",".jpeg")
    for fname in os.listdir(in_dir):
        if fname.lower().endswith(exts):
            path = os.path.join(in_dir, fname)
            img = Image.open(path)
            cropped = crop_top_center(img, top_ratio=top_ratio, height_ratio=height_ratio, size=size)
            cropped.save(os.path.join(out_dir, fname), format="PNG")  # PNG 저장 (투명 유지)
    print(f"✅ Done! Transparent cropped images saved to {out_dir}")

if __name__ == "__main__":
    # 입력 포켓몬 폴더
    in_dir = "./pokemon/Cartoon_Dataset"
    # 크롭된 이미지가 저장될 폴더
    out_dir = "./pokemon/Cartoon_Cropped"
    # 실행
    process_folder(in_dir, out_dir, size=128, top_ratio=0.1, height_ratio=0.6)
