from pathlib import Path
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

root = "yolo"
names = ["dent", "rust", "scratch"]
colors = ["deepskyblue", "orange", "magenta"]
rng = random.Random(42)
output = "check"
Path(output).mkdir(exist_ok=True)

for split, count in [("train", 5), ("valid", 3), ("test", 2)]:
    for path in rng.sample(sorted(Path(f"{root}/{split}/images").glob("*.jpg")), count):
        fig, ax = plt.subplots(figsize=(10, 8))
        image = plt.imread(path)
        h, w = image.shape[:2]
        ax.imshow(image)
        labels = Path(f"{root}/{split}/labels/{path.stem}.txt")
        for line in labels.read_text().splitlines():
            cls, x, y, bw, bh = map(float, line.split())
            color = colors[int(cls)]
            left, top = (x - bw / 2) * w, (y - bh / 2) * h
            ax.add_patch(Rectangle((left, top), bw * w, bh * h,
                                   fill=False, edgecolor=color, linewidth=2))
            ax.text(left, top, names[int(cls)], color="black", backgroundcolor=color)
        ax.set_title(f"{split}: {path.name.split('.rf.')[0]}")
        ax.axis("off")
        fig.savefig(f"{output}/{split}_{path.name.split('.rf.')[0]}.png",
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
