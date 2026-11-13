from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from ultralytics import YOLO


VARIANT = 5


def main():
    device = 0 if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)

    print(f"Обучение на: {torch.cuda.get_device_name(device)}")

    root = Path(__file__).resolve().parent
    data = root / "dataset_v2" / "data.yaml"
    project = root / "runs" / "lr6" 
    output = root / "results"

    if not data.is_file():
        raise FileNotFoundError(data)

    # Проверяем оба опыта заранее, чтобы сохранить предыдущие результаты.
    for tag in ("a", "b"):
        name = f"v{VARIANT:02d}_{tag}"

        for directory in (project / name, project / f"{name}_valid"):
            if directory.exists():
                raise FileExistsError(
                    f"Каталог уже существует: {directory}. "
                )

    if output.exists():
        raise FileExistsError(
            f"Каталог результатов уже существует: {output}. "
        )

    output.mkdir(parents=True)

    common = dict(
        data=str(data),
        epochs=40,
        imgsz=640,
        batch=4,
        device=device,
        workers=0,
        seed=42,
        deterministic=True,
        amp=False,
        patience=0,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        freeze=0,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        close_mosaic=0,
        plots=True,
        save=True,
        exist_ok=False,
    )

    rows = []

    for tag, value in (("a", 0.0), ("b", 0.3)):
        name = f"v{VARIANT:02d}_{tag}"

        # Заново задаём начальные установки перед каждым опытом.
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        cfg = common.copy()
        cfg["hsv_v"] = value

        # Каждый опыт начинается с одинаковых предварительных весов.
        source = root / "yolov8n.pt"
        model = YOLO(str(source))

        print(f"\nОпыт {tag.upper()}: hsv_v={value}")

        torch.cuda.synchronize()
        started = time.perf_counter()

        model.train(
            project=str(project),
            name=name,
            **cfg,
        )

        torch.cuda.synchronize()
        train_s = time.perf_counter() - started

        run = Path(model.trainer.save_dir)
        weights = run / "weights" / "best.pt"

        # Отдельно оцениваем сохранённую лучшую модель на valid.
        best = YOLO(str(weights))

        metrics = best.val(
            data=str(data),
            split="val",
            imgsz=cfg["imgsz"],
            batch=1,
            device=device,
            workers=0,
            half=False,
            conf=0.001,
            iou=0.7,
            plots=True,
            project=str(project),
            name=f"{name}_valid",
            exist_ok=False,
        )

        row = dict(
            run=name,
            factor="hsv_v",
            value=value,
            source=str(source),
            imgsz=cfg["imgsz"],
            epochs=cfg["epochs"],
            train_s=round(train_s, 2),
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            weights_mb=round(weights.stat().st_size / 1e6, 3),
            weights=str(weights),
        )
        rows.append(row)

        (output / f"{name}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Обновляем общую таблицу после каждого завершённого опыта.
        with (output / "lr6_compare.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)

        print(json.dumps(row, ensure_ascii=False, indent=2))

        del model, best, metrics
        torch.cuda.empty_cache()

    print(f"\nОба опыта завершены. Таблица: {output / 'lr6_compare.csv'}")


if __name__ == "__main__":
    main()
