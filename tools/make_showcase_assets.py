#!/usr/bin/env python3
"""Generate web previews and benchmark figures from committed replay evidence.

Run from the repository root. Pass --video to transcode the original GIFs with
ffmpeg; their frame timing is retained, independently of pipeline latency.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

BG, PANEL, FG, MUTED = '#0c131b', '#14202b', '#edf4f8', '#a6b8c6'
BLUE, GREEN = '#7096ba', '#90e0b8'


def read_json(path):
    return json.loads(path.read_text())


def configure_axes(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=10)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color('#344552')
    ax.grid(axis='x', color='#263642', linewidth=0.6)
    ax.set_axisbelow(True)


def save_figure(fig, out, name):
    for suffix in ['png', 'svg']:
        fig.savefig(out / f'{name}.{suffix}', dpi=180, facecolor=BG,
                    bbox_inches='tight', metadata={'Date': None} if suffix == 'svg' else None)
        if suffix == 'svg':
            path = out / f'{name}.{suffix}'
            path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
    plt.close(fig)


def latency_figure(root, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout='constrained')
    fig.patch.set_facecolor(BG)
    for ax, scene in zip(axes, ['urban', 'highway']):
        raw = read_json(root / f'{scene}_voxel_0.json')
        voxel = read_json(root / f'{scene}_voxel_0.2.json')
        configure_axes(ax)
        values = [raw['latency_ms']['total_ms']['mean'], voxel['latency_ms']['total_ms']['mean']]
        ax.barh([1, 0], values, height=0.5, color=[BLUE, GREEN])
        ax.set_yticks([1, 0], ['Raw points', '0.2 m voxels'])
        ax.set_xlim(0, 1080)
        ax.set_ylim(-0.65, 1.65)
        for y, value in zip([1, 0], values):
            ax.text(value + 18, y, f'{value:.1f} ms', va='center', color=FG, weight='bold')
        ax.axvline(100, color='#e5bd7e', linestyle='--', linewidth=1)
        ax.set_title(f'{scene.title()}  /  {values[0] / values[1]:.2f}× faster',
                     loc='left', color=FG, fontsize=15, weight='bold', pad=18)
        ax.set_xlabel('Mean pipeline latency (ms) · lower is better', color=MUTED, labelpad=12)
    fig.suptitle('CARLA replay · 400 frames per scene · fusion + IMM', color=FG, fontsize=16)
    fig.supxlabel('Single run per setting. Dashed line: 100 ms sensor interval. '
                  'Excludes I/O, evaluation, rendering and 5 warmup frames.',
                  color=MUTED, fontsize=9)
    save_figure(fig, out, 'latency_comparison')


def forecasting_figure(root, out):
    data = read_json(root / 'forecasting/metrics.json')['results']['test_pooled']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout='constrained')
    fig.patch.set_facecolor(BG)
    names, labels = ['cv', 'ca', 'ridge'], ['CV', 'Constant acceleration', 'Ridge residual']
    for ax, group, title in zip(axes, ['all', 'moving'], ['All test windows', 'Moving actors']):
        configure_axes(ax)
        values = [data[name][group]['ade_m'] for name in names]
        ax.barh([2, 1, 0], values, height=0.5, color=[GREEN, BLUE, '#dbbb86'])
        ax.set_yticks([2, 1, 0], labels)
        ax.set_xlim(0, 0.7)
        ax.set_ylim(-0.6, 2.6)
        for y, value in zip([2, 1, 0], values):
            ax.text(value + 0.018, y, f'{value:.3f}', va='center', color=FG, weight='bold')
        ax.set_title(f'{title} · n={data["cv"][group]["windows"]:,}', loc='left',
                     color=FG, fontsize=13, weight='bold', pad=18)
        ax.set_xlabel('ADE (m) · lower is better', color=MUTED, labelpad=12)
    fig.suptitle('Held-out towns · 2 s observed history → 3 s forecast', color=FG, fontsize=16)
    fig.supxlabel('Ground-truth actor histories; no detection or tracking errors. '
                  'Window-weighted means; overlapping windows are correlated.', color=MUTED, fontsize=9)
    save_figure(fig, out, 'forecasting_comparison')


def font(size):
    return ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans'), size)


def previews(screenshots, out, video):
    for scene in ['urban', 'highway']:
        source = screenshots / f'demo_carla_{scene}.gif'
        with Image.open(source) as gif:
            gif.seek(60)
            poster = gif.convert('RGB')
            poster.save(out / f'{scene}_poster.jpg', quality=90)
        if video:
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                            '-i', str(source), '-an', '-c:v', 'libx264', '-crf', '24',
                            '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                            str(out / f'{scene}.mp4')], check=True)
    banner = Image.new('RGB', (1600, 920), BG)
    draw = ImageDraw.Draw(banner)
    draw.text((62, 40), 'AUTO-DRIVER  /  AUTONOMOUS-DRIVING PERCEPTION', font=font(22), fill=GREEN)
    draw.text((58, 92), 'From raw sensors to tracked motion.', font=font(56), fill=FG)
    draw.text((62, 180), 'Camera + LiDAR  /  World-frame tracking  /  Motion forecasting  /  ROS 2',
              font=font(24), fill=MUTED)
    with Image.open(out / 'urban_poster.jpg') as poster:
        poster = poster.resize((1476, 558), Image.Resampling.LANCZOS)
        banner.paste(poster, (62, 250))
    draw.text((62, 855), 'CARLA 0.9.16  •  Recorded sensor replay  •  Reproducible metrics', font=font(24), fill=FG)
    banner.save(out / 'overview.jpg', quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmarks', type=Path, default=Path('docs/benchmarks'))
    parser.add_argument('--screenshots', type=Path, default=Path('docs/screenshots'))
    parser.add_argument('--out', type=Path, default=Path('docs/assets'))
    parser.add_argument('--video', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams['svg.hashsalt'] = 'auto-driver'
    previews(args.screenshots, args.out, args.video)
    latency_figure(args.benchmarks, args.out)
    forecasting_figure(args.benchmarks, args.out)


if __name__ == '__main__':
    main()
