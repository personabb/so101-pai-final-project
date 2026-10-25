"""Check recording quality decisions before training or transfer."""
from __future__ import annotations

import json
import os
from pathlib import Path
import uuid


def mark_dataset_review_state(dataset_root, *, ready, reason=""):
    path = Path(dataset_root) / "training-selection.json"
    if not path.parent.is_dir():
        raise FileNotFoundError(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps({"schema_version": 1, "ready": bool(ready),
            "reason": str(reason), "scope": "Recording error/deletion selection; not task quality approval"},
            ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def assert_training_eligible(dataset_root):
    root = Path(dataset_root)
    selection = root / "training-selection.json"
    if selection.exists():
        state = json.loads(selection.read_text(encoding="utf-8-sig"))
        if state.get("ready") is not True:
            raise ValueError("Dataset is not eligible for training: " + state.get("reason", "Recording review pending"))
    else:
        report_path = root / "collection-report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8-sig"))
            error = report.get("error", "")
            if error.startswith(("Recording:", "Storage:", "Control:")):
                raise ValueError("Dataset contains unresolved recording errors; remove the failed attempts before training")
    info_path = root / "meta/info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8-sig"))
        if info.get("total_episodes") == 0 or info.get("total_frames") == 0:
            raise ValueError("An empty dataset cannot be used for training")

