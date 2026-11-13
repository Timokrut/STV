from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO


CONF = 0.25


def main():
    root = Path(__file__).resolve().parent
    dataset = root / "dataset" / "valid"
    output = root / "results" / "step8"
    output.mkdir(parents=True, exist_ok=True)

    models = {
        "A": YOLO(str(root / "runs/lr6/v05_a/weights/best.pt")),
        "B": YOLO(str(root / "runs/lr6/v05_b/weights/best.pt")),
    }
    names = models["A"].names

    for path in sorted((dataset / "images").glob("*.jpg")):
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Не удалось прочитать {path}")

        h, w = image.shape[:2]
        annotated = image.copy()

        label_path = dataset / "labels" / f"{path.stem}.txt"
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            cls, x, y, bw, bh = map(float, line.split())
            left = round((x - bw / 2) * w)
            top = round((y - bh / 2) * h)
            right = round((x + bw / 2) * w)
            bottom = round((y + bh / 2) * h)

            cv2.rectangle(
                annotated, (left, top), (right, bottom),
                (0, 255, 0), 2,
            )
            cv2.putText(
                annotated, names[int(cls)],
                (left, max(top - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1,
            )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Разметка")

        for ax, (tag, model) in zip(axes[1:], models.items()):
            prediction = model.predict(
                source=image,
                imgsz=640,
                conf=CONF,
                iou=0.7,
                device=0,
                verbose=False,
            )[0]

            ax.imshow(cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB))
            ax.set_title(f"Опыт {tag}, conf={CONF}")
            print(f"{path.name}: {tag}, рамок: {len(prediction.boxes)}")

        for ax in axes:
            ax.axis("off")

        short_name = path.stem.split(".rf.")[0]
        fig.suptitle(short_name)
        fig.tight_layout()
        fig.savefig(output / f"{short_name}.png", dpi=150)
        plt.close(fig)

    print(f"Сравнения сохранены: {output}")


if __name__ == "__main__":
    main()
