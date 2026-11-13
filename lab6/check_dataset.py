from pathlib import Path
from collections import Counter, defaultdict
import json
import math

import yaml

root = Path("dataset")
output = Path("results")
output.mkdir(exist_ok=True)

cfg = yaml.safe_load((root / "data.yaml").read_text())
names = cfg["names"]

if isinstance(names, list):
    names = dict(enumerate(names))
else:
    names = {int(k): v for k, v in names.items()}

seen_sources = defaultdict(set)
summary = {}
errors = []

for split in ("train", "valid", "test"):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    counts = Counter()

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}

    for stem in sorted(image_stems - label_stems):
        errors.append(f"{split}: отсутствует разметка для {stem}")

    for stem in sorted(label_stems - image_stems):
        errors.append(f"{split}: разметка без изображения: {stem}")

    for image_path in images:
        source = image_path.stem.split(".rf.")[0]
        seen_sources[source].add(split)

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        for number, line in enumerate(label_path.read_text().splitlines(), 1):
            if not line.strip():
                continue

            try:
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError("ожидается рамочная разметка из 5 значений")

                cls = int(fields[0])
                x, y, w, h = map(float, fields[1:])

                if cls not in names:
                    raise ValueError("неизвестный номер класса")

                if not all(math.isfinite(v) for v in (x, y, w, h)):
                    raise ValueError("некорректные координаты")

                if not (
                    0 <= x <= 1 and 0 <= y <= 1
                    and 0 < w <= 1 and 0 < h <= 1
                ):
                    raise ValueError("координаты вне диапазона")

                if (
                    min(x - w / 2, y - h / 2) < -1e-5
                    or max(x + w / 2, y + h / 2) > 1.00001
                ):
                    raise ValueError("рамка выходит за изображение")

                counts[cls] += 1

            except ValueError as error:
                errors.append(f"{label_path}:{number}: {error}")

    summary[split] = {
        "images": len(images),
        "boxes": {name: counts[cls] for cls, name in names.items()},
    }

for source, splits in seen_sources.items():
    if len(splits) > 1:
        errors.append(
            f"Исходный снимок в разных выборках: "
            f"{source}, {sorted(splits)}"
        )

report = {"splits": summary, "errors": errors}

(output / "dataset_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(report, ensure_ascii=False, indent=2))

if errors:
    raise SystemExit("Обнаружены ошибки.")
