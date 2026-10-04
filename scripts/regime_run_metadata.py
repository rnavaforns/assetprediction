"""Small helpers for linking generated regime outputs to their W&B runs."""

from __future__ import annotations

import json
from pathlib import Path


def write_wandb_run_metadata(model_name: str, output_dir: str | Path = "data") -> Path | None:
    """Save the active W&B run identity and config alongside its model outputs."""
    try:
        import wandb
    except ImportError:
        return None
    run = wandb.run
    if run is None:
        return None

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_name.lower().replace(" ", "_")
    path = directory / f"{safe_model_name}_wandb_run.json"
    payload = {
        "model": model_name,
        "run_id": run.id,
        "run_name": run.name,
        "run_url": run.url,
        "entity": run.entity,
        "project": run.project,
        "group": run.group,
        "tags": list(run.tags or []),
        "config": dict(run.config),
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path
