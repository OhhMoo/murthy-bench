"""One-shot center-crop helper for the one-pager.

- Headshots → square (1:1) at 500px
- Solution screenshots → identical 16:9 landscape crop at 1600x900
"""
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
HEADSHOTS_IN = ROOT / "headshots"
HEADSHOTS_OUT = ROOT / "headshots_sq"
PICS_IN = ROOT / "pictures"
PICS_OUT = ROOT / "pictures_169"

HEADSHOTS_OUT.mkdir(exist_ok=True)
PICS_OUT.mkdir(exist_ok=True)


def center_crop(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize so the image covers target box, then center-crop to exact size."""
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale + 0.5), int(src_h * scale + 0.5)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


for f in sorted(HEADSHOTS_IN.iterdir()):
    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
        out = HEADSHOTS_OUT / (f.stem + ".jpg")
        img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        center_crop(img, 500, 500).save(out, "JPEG", quality=88)
        print(f"headshot square: {f.name} → {out.name} ({img.size})")

for name in ("HF.jpg", "convo.jpg"):
    src = PICS_IN / name
    out = PICS_OUT / name
    img = Image.open(src).convert("RGB")
    center_crop(img, 1600, 900).save(out, "JPEG", quality=88)
    print(f"solution 16:9: {name}")
