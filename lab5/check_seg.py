from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

root = "yolo_seg"
output = "check_seg"
Path(output).mkdir(exist_ok=True)
colors = {"dent": "deepskyblue", "rust": "darkorange", "scratch": "magenta"}
font = ImageFont.load_default(size=14)

with open("polygon_metrics.csv", encoding="utf-8-sig") as file:
    metrics = {(r["split"], r["image"], r["polygon"]): r for r in csv.DictReader(file)}

for split in ["train", "valid", "test"]:
    for path in sorted(Path(f"{root}/{split}/images").glob("*.jpg")):
        with Image.open(path) as source:
            image = source.convert("RGB")
        w, h = image.size
        lines = Path(f"{root}/{split}/labels/{path.stem}.txt").read_text().splitlines()
        result = Image.new("RGB", (w + 280, max(h, 20 + 48 * len(lines))), "white")
        result.paste(image, (0, 0))
        draw = ImageDraw.Draw(result)

        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            parts = line.split()
            points = [(float(x) * w, float(y) * h)
                      for x, y in zip(parts[1::2], parts[2::2])]
            row = metrics[(split, path.name, str(number))]
            color = colors[row["class"]]
            draw.line(points + [points[0]], fill=color, width=2)
            draw.text(points[0], str(number), fill=color, font=font,
                      stroke_width=1, stroke_fill="black")
            text = f"{number}. {row['class']}\nS={row['area_px2']} px2   K={row['fill_ratio']}"
            draw.text((w + 12, 10 + (number - 1) * 48), text, fill=color, font=font)

        result.save(f"{output}/{split}_{path.name.split('.rf.')[0]}.jpg", quality=95)
