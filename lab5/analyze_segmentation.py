from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from PIL import Image

root = "yolo_seg"


def polygon_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    a = np.dot(x, np.roll(y, -1))
    b = np.dot(y, np.roll(x, -1))
    return 0.5 * abs(a - b)


def polygon_to_bbox(pts):
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    return x_min, y_min, x_max - x_min, y_max - y_min


def from_yolo_seg(line, w, h):
    parts = line.split()
    xy = np.array(parts[1:], dtype=float).reshape(-1, 2)
    return int(parts[0]), xy * [w, h]


if __name__ == "__main__":
    names = yaml.safe_load(Path(f"{root}/data.yaml").read_text())["names"]
    rows = []
    for split in ["train", "valid", "test"]:
        for path in sorted(Path(f"{root}/{split}/images").glob("*")):
            if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            with Image.open(path) as image:
                width, height = image.size
            labels = Path(f"{root}/{split}/labels/{path.stem}.txt")
            for number, line in enumerate(labels.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                if len(line.split()) < 7:
                    raise ValueError(f"{labels}:{number}: нужен полигон, а не бокс")
                cls, pts = from_yolo_seg(line, width, height)
                S = polygon_area(pts)
                x, y, w, h = polygon_to_bbox(pts)
                if w <= 0 or h <= 0:
                    raise ValueError(f"{labels}:{number}: вырожденный полигон")
                K = S / (w * h)
                rows.append([split, path.name, number, names[cls], round(S, 2), round(K, 4)])

    table = pd.DataFrame(rows, columns=["split", "image", "polygon", "class", "area_px2", "fill_ratio"])
    table.to_csv("polygon_metrics.csv", index=False, encoding="utf-8-sig")
    print(table.to_string(index=False))
    print(f"\nВсего полигонов: {len(table)}. Результат: polygon_metrics.csv")
    if len(table) < 15:
        print("Для задания требуется минимум 15 объектов.")
