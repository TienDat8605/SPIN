"""
8-cell ablation grid runner for the CB-Spectral head modulation.

Cells:  spectral_mode in {fft, power, block, none} x backbone in {llava-1.5, qwen-vl}

For each cell:
  1. Re-instantiates the model loader (LLaVA or Qwen-VL).
  2. Runs the eval script with the matching --use-cb --spectral-mode flags.
  3. Captures (mode, backbone, lat, chair, pope) into a CSV.

Designed to be invoked manually once a proper GPU env is available.
Run a single cell with --cell "fft,llava-1.5" or the whole grid with --all.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

GRID = [
    ("fft",   "llava-1.5"),
    ("power", "llava-1.5"),
    ("block", "llava-1.5"),
    ("none",  "llava-1.5"),
    ("fft",   "qwen-vl"),
    ("power", "qwen-vl"),
    ("block", "qwen-vl"),
    ("none",  "qwen-vl"),
]

CHAIR_CMD_TEMPLATE = [
    "{python}", "chair_eval.py",
    "--model", "{backbone}",
    "--data-path", "{data_path}",
    "--llava-size", "{llava_size}",
    "--batch-size", "1",
    "--beam", "1",
    "--max-tokens", "512",
    "--start-layer", "0",
    "--end-layer", "32",
    "--use-cb",
    "--spectral-mode", "{mode}",
    "--n-calib-examples", "64",
]

QWEN_CHAIR_CMD_TEMPLATE = [
    "{python}", "chair_eval_qwen.py",
    "--data-path", "{data_path}",
    "--max-tokens", "512",
    "--use-cb",
    "--spectral-mode", "{mode}",
    "--n-calib-examples", "64",
]


def run_cell(mode: str, backbone: str, data_path: str, llava_size: str, results_dir: str) -> dict:
    cell_name = f"{mode}_{backbone}"
    out_json = os.path.join(results_dir, f"chair_{cell_name}.jsonl")
    if os.path.exists(out_json):
        os.remove(out_json)

    if backbone == "llava-1.5":
        cmd = [c.format(python=sys.executable, backbone=backbone,
                        data_path=data_path, llava_size=llava_size, mode=mode)
               for c in CHAIR_CMD_TEMPLATE]
    elif backbone == "qwen-vl":
        cmd = [c.format(python=sys.executable, data_path=data_path, mode=mode)
               for c in QWEN_CHAIR_CMD_TEMPLATE]
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    cmd.extend(["--output-path", out_json])

    print(f"\n=== Running cell: {cell_name} ===")
    print(" ".join(cmd))

    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    rc = subprocess.run(cmd, env=env).returncode
    dt = time.time() - t0

    n_captions = 0
    if os.path.exists(out_json):
        with open(out_json) as f:
            n_captions = sum(1 for _ in f)

    return {
        "cell": cell_name,
        "mode": mode,
        "backbone": backbone,
        "returncode": rc,
        "wall_seconds": round(dt, 1),
        "n_captions": n_captions,
        "out_json": out_json,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/path/to/coco/val2014/")
    parser.add_argument("--llava-size", type=str, default="7b")
    parser.add_argument("--results-dir", type=str, default="runs/ablation")
    parser.add_argument("--cell", type=str, default=None,
                        help="Single cell, e.g. 'fft,llava-1.5'")
    parser.add_argument("--all", action="store_true", help="Run all 8 cells")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    if args.cell:
        mode, backbone = args.cell.split(",")
        cells = [(mode, backbone)]
    elif args.all:
        cells = GRID
    else:
        parser.error("Specify --cell MODE,BACKBONE or --all")

    summary_path = os.path.join(args.results_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["cell", "mode", "backbone", "returncode", "wall_seconds", "n_captions", "out_json"]
        )
        writer.writeheader()
        for mode, backbone in cells:
            row = run_cell(mode, backbone, args.data_path, args.llava_size, args.results_dir)
            writer.writerow(row)
            f.flush()
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
