# сохраняем скачаные зависимости в requirements.txt
# python -m pip freeze > requirements.txt

import os
import platform
from pathlib import Path
from importlib.metadata import version

import torch

cpu = next(
    line.split(":", 1)[1].strip()
    for line in Path("/proc/cpuinfo").read_text().splitlines()
    if line.startswith("model name")
)

ram_gb = (
    os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

lines = [
    f"Python: {platform.python_version()}",
    f"System: {platform.platform()}",
    f"CPU: {cpu}",
    f"RAM: {ram_gb:.2f} Gb",
    f"Device: {device}",
    f"CUDAv: {torch.version.cuda}",
]

if torch.cuda.is_available():
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    lines.append(f"VRAM: {vram:.2f} ГиБ")

for package in (
    "ultralytics", "torch", "torchvision",
    "opencv-python", "PyYAML", "numpy"
):
    lines.append(f"{package}: {version(package)}")

report = "\n".join(lines)
Path("environment.txt").write_text(report + "\n", encoding="utf-8")
print(report)

