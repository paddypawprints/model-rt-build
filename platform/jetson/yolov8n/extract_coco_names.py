#!/usr/bin/env python3
"""
Extract COCO class names from the Ultralytics package (preferred) or a local YAML file
and write them to a standalone config file (JSON or YAML).

Usage examples:
  python extract_coco_names.py --out coco_names.json
  python extract_coco_names.py --out coco_names.yaml

This script writes a simple config that other code (e.g. yolov8n_trt_model.py) can
load without importing Ultralytics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

try:
    import yaml  # PyYAML
except Exception:
    yaml = None

# Optional Ultralytics helpers
try:
    from ultralytics.utils import YAML as U_YAML  # type: ignore
    from ultralytics.utils.checks import check_yaml as U_check_yaml  # type: ignore
except Exception:
    U_YAML = None
    U_check_yaml = None


def _load_names_from_ultralytics(preferred_names=("coco8.yaml", "coco.yaml")) -> Optional[List[str]]:
    """Try to load names using Ultralytics helpers."""
    if U_YAML is None or U_check_yaml is None:
        return None
    for name in preferred_names:
        try:
            path = U_check_yaml(name)
            if path:
                data = U_YAML.load(path)
                if isinstance(data, dict) and "names" in data:
                    return list(data["names"])
                if isinstance(data, list):
                    return list(data)
        except Exception:
            continue
    return None


def _load_names_from_local_yaml(files=("coco8.yaml", "coco.yaml")) -> Optional[List[str]]:
    """Try to read names from local YAML files using PyYAML."""
    if yaml is None:
        return None
    for f in files:
        p = Path(f)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                if isinstance(data, dict) and "names" in data:
                    return list(data["names"])
                if isinstance(data, list):
                    return list(data)
            except Exception:
                continue
    return None


def extract_and_write(output_path: Path) -> List[str]:
    """Extract names and write to the requested output file (json or yaml)."""
    names = _load_names_from_ultralytics()
    if names is None:
        names = _load_names_from_local_yaml()
    if names is None:
        raise RuntimeError(
            "Could not find Ultralytics coco yaml (ultralytics not installed or file not found). "
            "Place coco8.yaml or coco.yaml in the current directory or install ultralytics."
        )

    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(names, fh, indent=2, ensure_ascii=False)
    else:
        if yaml is None:
            # write JSON instead and warn if YAML requested
            with output_path.with_suffix(".json").open("w", encoding="utf-8") as fh:
                json.dump(names, fh, indent=2, ensure_ascii=False)
            raise RuntimeError("PyYAML is not installed; wrote JSON instead. Install PyYAML to write YAML.")
        with output_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump({"names": names}, fh, sort_keys=False, allow_unicode=True)
    return names


def main() -> None:
    p = argparse.ArgumentParser(description="Extract Ultralytics COCO class names to a config file.")
    p.add_argument("--out", "-o", default="coco_names.json", help="Output path (json or yaml). Default coco_names.json")
    args = p.parse_args()

    out_path = Path(args.out)
    try:
        names = extract_and_write(out_path)
        print(f"Wrote {len(names)} class names to {out_path}")
    except Exception as e:
        raise SystemExit(f"Failed to extract coco names: {e}")


if __name__ == "__main__":
    main()