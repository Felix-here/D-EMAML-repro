from __future__ import annotations

from pathlib import Path

import torch


def snapshot(path: str | Path | None = None) -> dict:
    report: dict[str, object] = {}
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_used_gb"] = round((memory.total - memory.available) / 1024**3, 3)
        report["ram_total_gb"] = round(memory.total / 1024**3, 3)
        report["ram_percent"] = memory.percent
    except Exception as exc:  # pragma: no cover
        report["ram_error"] = str(exc)
    if path is not None:
        usage = __import__("shutil").disk_usage(path)
        report["disk_used_gb"] = round(usage.used / 1024**3, 3)
        report["disk_free_gb"] = round(usage.free / 1024**3, 3)
        report["disk_total_gb"] = round(usage.total / 1024**3, 3)
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["gpu"] = torch.cuda.get_device_name(0)
        report["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 3)
        report["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 3)
        report["vram_max_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )
    return report
