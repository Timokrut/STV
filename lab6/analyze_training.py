from pathlib import Path
import csv

import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent

columns = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))

for tag, label in (("a", "A: hsv_v=0"), ("b", "B: hsv_v=0.3")):
    path = root / "runs" / "lr6" / f"v05_{tag}" / "results.csv"

    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = [
            {key.strip(): float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    epochs = [int(row["epoch"]) for row in rows]

    for ax, column in zip(axes.flat, columns):
        ax.plot(epochs, [row[column] for row in rows], label=label)
        ax.set_title(column)
        ax.set_xlabel("Эпоха")
        ax.grid(alpha=0.3)
        ax.legend()

    maximum = max(row["metrics/mAP50-95(B)"] for row in rows)
    peak_epochs = [
        int(row["epoch"])
        for row in rows
        if row["metrics/mAP50-95(B)"] == maximum
    ]

    print(
        f"{label}: максимум mAP50-95={maximum:.5f}, "
        f"эпохи: {peak_epochs}"
    )

fig.tight_layout()

output = root / "results"
output.mkdir(exist_ok=True)
fig.savefig(output / "step7_comparison.png", dpi=150)
plt.close(fig)

print("Графики сохранены в results/step7_comparison.png")
