"""
Rebuild the summary and figures of the FFS baseline from the raw files in this directory.

Inputs:  run_10k/step_metrics.jsonl, run_1k/step_metrics.jsonl, samples_10k_model/sample_*.mp4
Outputs: summary.json, run_*/loss_curve.png, samples_10k_model/frames_*.png

matplotlib is not part of the training environment, so run it in a temporary uv environment:
uv run --no-project --with matplotlib --with numpy --with imageio --with imageio-ffmpeg --with pillow \
    python baseline_results/ffs/scripts/make_figures.py
"""

import os
import glob
import json
import statistics

import numpy as np
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
RAW_LINE = "#b7d3f6"
MAIN_LINE = "#2a78d6"


def load(run):
    with open(os.path.join(BASE, run, "step_metrics.jsonl")) as f:
        return [json.loads(line) for line in f]


def moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def summarize(records, segments):
    losses = [r["loss"] for r in records]
    steps = records[1:]  # step 0 includes DataLoader start-up
    wall = records[-1]["wall_time"] - records[0]["wall_time"]
    step_sum = sum(r["step_time_s"] for r in steps)
    data_sum = sum(r["data_time_s"] for r in steps)
    return {
        "steps": len(records),
        "loss_by_segment": [
            {"steps": f"{lo}-{min(hi, len(losses)) - 1}",
             "mean": round(statistics.mean(losses[lo:hi]), 4),
             "std": round(statistics.pstdev(losses[lo:hi]), 4)}
            for lo, hi in segments
        ],
        "wall_time_h": round(wall / 3600, 3),
        "wall_time_per_step_s": round(wall / len(steps), 4),
        "compute_time_median_s": round(statistics.median(r["compute_time_s"] for r in steps), 4),
        "data_wait_share": round(data_sum / step_sum, 4),
        "max_mem_allocated_mib": round(max(r["max_mem_allocated_mib"] for r in records)),
        "max_mem_reserved_mib": round(max(r["max_mem_reserved_mib"] for r in records)),
    }


def plot_loss(records, window, zoom_from, zoom_ylim, title, path):
    x = np.array([r["step"] for r in records])
    y = np.array([r["loss"] for r in records])
    ma = moving_average(y, window)
    xm = x[window - 1:]
    last = int(x[-1])

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
                         "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY, "text.color": TEXT_PRIMARY})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURFACE, gridspec_kw={"wspace": 0.18})
    panels = [(0, f"Steps 0-{last:,}", (0, 1.1)), (zoom_from, f"Steps {zoom_from:,}-{last:,} (zoomed)", zoom_ylim)]
    for ax, (lo, panel_title, ylim) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.plot(x[x >= lo], y[x >= lo], color=RAW_LINE, lw=0.6 if len(x) > 2000 else 1, label="Per-step loss")
        ax.plot(xm[xm >= lo], ma[xm >= lo], color=MAIN_LINE, lw=2, label=f"{window}-step moving average")
        ax.set_title(panel_title, loc="left", fontsize=11, color=TEXT_PRIMARY)
        ax.set_xlabel("Training step")
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(lo, last + 1)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    axes[0].set_ylabel("Loss (mean over 8 ranks)")
    axes[0].legend(frameon=False, loc="upper right", labelcolor=TEXT_SECONDARY)
    axes[1].annotate(f"{ma[-1]:.3f}", (xm[-1], ma[-1]), xytext=(-4, 12), textcoords="offset points",
                     ha="right", color=TEXT_PRIMARY, fontsize=9)
    fig.suptitle(title, x=0.07, ha="left", fontsize=12, color=TEXT_PRIMARY, y=1.0)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return float(ma[-1])


def frame_grid(video_path, image_path, cols=4):
    frames = imageio.mimread(video_path)
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), SURFACE)
    for i, frame in enumerate(frames):
        sheet.paste(Image.fromarray(frame), ((i % cols) * w, (i // cols) * h))
    sheet.save(image_path)
    return len(frames), (h, w)


def main():
    run_10k = load("run_10k")
    run_1k = load("run_1k")

    summary = {
        "run_10k": summarize(run_10k, [(0, 100), (100, 1000)] + [(k, k + 1000) for k in range(1000, 9000, 1000)]
                             + [(9000, len(run_10k))]),
        "run_1k": summarize(run_1k, [(0, 100), (100, 1000)]),
    }
    same = all(a["loss"] == b["loss"] for a, b in zip(run_10k, run_1k))
    summary["reproducibility"] = {
        "compared_steps": min(len(run_10k), len(run_1k)),
        "per_step_loss_identical": same,
    }
    title = "Latte-XL/2 FFS from scratch - training loss, 8xH200, global batch 40"
    summary["run_10k"]["loss_ma100_last"] = round(plot_loss(
        run_10k, 100, 1000, (0, 0.25), title + " (10k steps)", os.path.join(BASE, "run_10k", "loss_curve.png")), 4)
    summary["run_1k"]["loss_ma50_last"] = round(plot_loss(
        run_1k, 50, 100, (0, 0.35), title + " (1k steps)", os.path.join(BASE, "run_1k", "loss_curve.png")), 4)

    samples = {}
    for video in sorted(glob.glob(os.path.join(BASE, "samples_10k_model", "sample_*.mp4"))):
        name = os.path.splitext(os.path.basename(video))[0]
        image = os.path.join(BASE, "samples_10k_model", name.replace("sample_", "frames_") + ".png")
        count, size = frame_grid(video, image)
        samples[name] = {"frames": count, "height_width": list(size)}
    summary["samples_10k_model"] = samples

    with open(os.path.join(BASE, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
