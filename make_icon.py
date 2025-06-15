# make_icon.py
# يولّد adas_icon.ico بلون أزرق وفيه دائرة بيضا بسيطة
from PIL import Image, ImageDraw

def make_icon(path="adas_icon.ico"):
    size = 256, 256
    img  = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # دائرة سماويّة
    draw.ellipse((32, 32, 224, 224), fill=(0, 153, 255, 255))

    # حرف A بسيط
    draw.text((100, 90), "A", fill="white")

    # احفظ كل المقاسات في ملف .ico
    img.save(path, sizes=[(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)])
    print(f"Icon saved → {path}")

if __name__ == "__main__":
    make_icon()
