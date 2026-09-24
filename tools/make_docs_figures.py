#!/usr/bin/env python3
"""Generate the README's static figures so the docs assets stay reproducible.

Pure matplotlib (no project imports), dark theme matched to the BEV demos:

    python tools/make_docs_figures.py --out docs/screenshots
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BG = "#0d1117"      # GitHub-dark canvas
PANEL = "#161b22"
FG = "#e6edf3"
MUT = "#9aa4b2"
ACCENT = {"detect": "#4cc9f0", "fusion": "#f4a261", "track": "#c08cf0", "predict": "#2ec4b6"}


def _card(ax, x, y, w, h, title, lines, edge, *, title_size=12, fill=PANEL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.6",
        linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
            color=FG, fontsize=title_size, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + h - 6.0, "\n".join(lines), ha="center", va="top",
            color=MUT, fontsize=8.4, family="monospace", zorder=3, linespacing=1.5)


def _arrow(ax, x0, y0, x1, y1, color=FG):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16,
        linewidth=2.0, color=color, zorder=1, shrinkA=2, shrinkB=2))


def architecture(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 62)
    ax.axis("off")

    ax.text(1.5, 60, "Auto-Driver — perception → fusion → tracking → prediction",
            ha="left", va="top", color=FG, fontsize=15, fontweight="bold")

    # inputs
    _card(ax, 1.5, 26, 18, 20, "Inputs",
          ["CARLA sim", "KITTI raw", "rosbag2", "camera-only"], MUT, title_size=11)

    # pure-python core container
    ax.add_patch(FancyBboxPatch(
        (23, 20), 84, 30, boxstyle="round,pad=0.6,rounding_size=2.0",
        linewidth=1.6, edgecolor="#30363d", facecolor="#0f141b", zorder=1))
    ax.text(25, 48.4, "perception_core  ·  pure Python (NumPy · SciPy · scikit-learn) — no ROS / CUDA / CARLA",
            ha="left", va="top", color="#7d8792", fontsize=9.5, style="italic")

    stages = [
        (25.5, "detect", "Detection", ["RANSAC ground", "+ DBSCAN + PCA"]),
        (46.0, "fusion", "Fusion", ["project 3D→2D", "IoU label xfer"]),
        (66.5, "track", "Tracking", ["KF / IMM(CV+CT)", "+ Hungarian"]),
        (87.0, "predict", "Prediction", ["CV / CTRV", "multi-modal"]),
    ]
    for x, key, title, lines in stages:
        _card(ax, x, 27, 18, 17, title, lines, ACCENT[key], title_size=12)

    # swappable-detector chips under the Detection card
    ax.text(34.5, 25.0, "swappable Detector backends", ha="center", va="top",
            color="#6e7681", fontsize=7.6, style="italic")
    chips = ["LiDAR cluster", "YOLO fusion", "VGGT cam", "GT replay"]
    cx = 24.4
    for c in chips:
        w = 0.62 * len(c) + 2.2
        ax.add_patch(FancyBboxPatch((cx, 20.4), w, 3.4, boxstyle="round,pad=0.2,rounding_size=1.2",
                                    linewidth=1.0, edgecolor=ACCENT["detect"], facecolor="#12222b", zorder=3))
        ax.text(cx + w / 2, 22.1, c, ha="center", va="center", color="#cbe7f2", fontsize=7.4, zorder=4)
        cx += w + 1.4

    # outputs
    _card(ax, 111, 26, 17.5, 20, "Outputs",
          ["ROS 2 Humble", "rclpy nodes", "RViz2 /", "Foxglove"], "#58a6ff", title_size=11)

    # flow arrows
    _arrow(ax, 19.5, 36, 25.5, 36, MUT)
    _arrow(ax, 43.5, 35.5, 46.0, 35.5, ACCENT["fusion"])
    _arrow(ax, 64.0, 35.5, 66.5, 35.5, ACCENT["track"])
    _arrow(ax, 84.5, 35.5, 87.0, 35.5, ACCENT["predict"])
    _arrow(ax, 107, 36, 111, 36, "#58a6ff")

    ax.text(65, 17.2, "detections lifted to the world frame via ego pose → stable tracks & forecasts under ego motion",
            ha="center", va="top", color="#6e7681", fontsize=8.2, style="italic")

    # evaluation / CI footer band
    ax.add_patch(FancyBboxPatch((1.5, 3.5), 127, 8.5, boxstyle="round,pad=0.4,rounding_size=1.6",
                                linewidth=1.4, edgecolor="#2ea043", facecolor="#0f1b12", zorder=1))
    ax.text(65, 9.6, "Engineering rigor", ha="center", va="center", color="#3fb950",
            fontsize=10.5, fontweight="bold")
    ax.text(65, 6.0, "CLEAR-MOT  ·  ADE / FDE  ·  latency budget gate  ·  66 unit tests  ·  GitHub Actions CI (py3.9–3.11)",
            ha="center", va="center", color=MUT, fontsize=8.8, family="monospace")

    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out_path}")


def fusion_gate(out_path: str) -> None:
    """Before/after of the YOLO camera-LiDAR gate on real KITTI clutter."""
    labels = ["Classical\nLiDAR only", "+ YOLO\ncamera gate"]
    colors = ["#5a6473", "#4cc9f0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=150)
    fig.patch.set_facecolor(BG)
    fig.suptitle("KITTI drive 0014 — detection scored inside the camera frustum",
                 color=FG, fontsize=13, fontweight="bold", y=0.99)

    for ax in (a1, a2):
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors=MUT)
        ax.grid(axis="y", color="#21262d", linewidth=0.8)
        ax.set_axisbelow(True)

    b1 = a1.bar(labels, [0.051, 0.533], color=colors, width=0.62, zorder=3)
    a1.set_title("Precision", color=FG, fontsize=11)
    a1.set_ylim(0, 0.62)
    for r, v in zip(b1, [0.051, 0.533]):
        a1.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                color=FG, fontsize=10, fontweight="bold")
    a1.annotate("×10", xy=(0.5, 0.42), color="#4cc9f0", fontsize=16, fontweight="bold", ha="center")

    b2 = a2.bar(labels, [5198, 177], color=colors, width=0.62, zorder=3)
    a2.set_title("False positives (total)", color=FG, fontsize=11)
    a2.set_ylim(0, 5900)
    for r, v in zip(b2, [5198, 177]):
        a2.text(r.get_x() + r.get_width() / 2, v + 130, f"{v}", ha="center",
                color=FG, fontsize=10, fontweight="bold")
    a2.annotate("−97%", xy=(1, 2600), color="#4cc9f0", fontsize=16, fontweight="bold", ha="center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/screenshots")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    architecture(os.path.join(args.out, "architecture.png"))
    fusion_gate(os.path.join(args.out, "kitti_fusion_gate.png"))


if __name__ == "__main__":
    main()
