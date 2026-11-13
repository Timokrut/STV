from pathlib import Path
import shutil

import yaml

root = Path("dataset").resolve()
path = root / "data.yaml"

cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
names = cfg["names"]

# Приводим перечень классов к словарю {номер: название}.
if isinstance(names, dict):
    names = {int(k): v for k, v in names.items()}

    if sorted(names) != list(range(len(names))):
        raise ValueError("Номера классов должны идти от 0 подряд")
else:
    names = dict(enumerate(names))

if not names or len(set(names.values())) != len(names):
    raise ValueError("Перечень классов пустой или содержит повторения")

# Проверяем наличие каталогов всех выборок.
for split in ("train", "valid", "test"):
    for subdir in ("images", "labels"):
        directory = root / split / subdir

        if not directory.is_dir():
            raise FileNotFoundError(directory)

# Сохраняем исходное описание один раз.
backup = root / "data_original.yaml"

if not backup.exists():
    shutil.copy2(path, backup)

# Записываем корректные пути для Ultralytics.
cfg = {
    "path": str(root),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "names": names,
}

path.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

print(f"Описание набора сохранено: {path}")
print("Классы:")
for number, name in names.items():
    print(f"  {number}: {name}")
