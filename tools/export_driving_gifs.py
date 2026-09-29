#!/usr/bin/env python3
"""Render annotated GIF excerpts from published driving records (Pillow + ffmpeg).

No simulation is rerun. Synthetic frames are shown at their recorded 5 Hz;
CARLA camera frames and controls are paired by their recorded 10 Hz index.
Run from any directory; --out must be a new directory. See docs/driving_gifs.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageColor, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
BG, PANEL, ROAD = '#0b121c', '#111e2d', '#1c2c3d'
FG, MUTED, GRID = '#edf4fb', '#99aec4', '#25384c'
GREEN, BLUE, AMBER, RED = '#85e4b8', '#68baff', '#ffc778', '#ff8d94'
WIDTH, HEIGHT, SCALE = 1000, 600, 2
STATUS = {
    'cruise': '沿路线巡航', 'avoidance': '绕行障碍', 'following': '保持跟车间距',
    'yielding': '减速让行', 'stop_line': '停车线约束', 'goal_approach': '接近终点',
    'goal_reached': '到达并停车', 'emergency_stop': '紧急制动',
    'stale_input': '感知超时 · 制动',
}
CLIPS = [
    dict(name='carla_lead_braking', source='carla_lead_braking.json',
         title='前车急刹 · 从跟车到停车', subtitle='Lead vehicle braking',
         start=4.0, end=11.0, fps=10, poster=6.8),
    dict(name='cut_in', source='lidar/cut_in.json',
         title='邻道切入 · 减速跟车', subtitle='Cut-in vehicle',
         start=2.0, end=10.0, fps=5, poster=4.0),
    dict(name='obstacle', source='lidar/obstacle.json',
         title='静态绕障 · 规划与实际轨迹', subtitle='Static obstacle avoidance',
         start=2.0, end=9.0, fps=5, poster=5.8),
    dict(name='signal_crossing', source='lidar/signal_crossing.json',
         title='绿灯之后 · 继续礼让行人', subtitle='Green light with crossing pedestrians',
         start=8.0, end=19.0, fps=5, poster=14.0),
    dict(name='dropout', source='lidar/dropout.json',
         title='感知超时 · 制动与恢复', subtitle='Input timeout and recovery',
         start=3.0, end=10.0, fps=5, poster=5.0),
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Canvas:
    def __init__(self, regular, bold):
        self.image = Image.new('RGB', (WIDTH * SCALE, HEIGHT * SCALE), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.font_paths = (regular, bold)
        self.fonts = {}

    def text(self, xy, value, size=16, color=FG, bold=False, anchor=None):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(str(self.font_paths[bold]), size * SCALE)
        self.draw.text(tuple(v * SCALE for v in xy), str(value), fill=color,
                       font=self.fonts[key], anchor=anchor)

    def rect(self, box, fill, radius=0, outline=None, width=1):
        self.draw.rounded_rectangle(tuple(v * SCALE for v in box), radius=radius * SCALE,
                                    fill=fill, outline=outline, width=width * SCALE)

    def line(self, points, fill, width=1):
        if len(points) > 1:
            self.draw.line([(x * SCALE, y * SCALE) for x, y in points],
                           fill=fill, width=width * SCALE, joint='curve')

    def polygon(self, points, fill, outline=None):
        pts = [(x * SCALE, y * SCALE) for x, y in points]
        self.draw.polygon(pts, fill=fill)
        if outline:
            self.draw.line(pts + [pts[0]], fill=outline, width=2 * SCALE, joint='curve')

    def finish(self):
        return self.image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def header(c, clip, frame, carla):
    c.rect((24, 23, 29, 60), GREEN, 2)
    c.text((42, 18), clip['title'], 27, bold=True)
    c.text((43, 58), clip['subtitle'], 15, MUTED)
    source = 'CARLA 0.9.16 · LiDAR' if carla else '合成场景 · LiDAR 检测'
    c.text((976, 22), source, 17, GREEN, anchor='rt')
    c.text((976, 52), f"原始 {clip['fps']} Hz · 1× 仿真时间", 15, MUTED, anchor='rt')
    c.line([(24, 91), (976, 91)], GRID)
    c.text((24, 568), 'AUTO-DRIVER  /  RECORDED REPLAY', 13, MUTED)
    c.text((976, 565), f"节选 {clip['start']:g}–{clip['end']:g} s    当前 {frame['time_s']:05.1f} s",
           16, FG, anchor='rt')
    progress = (frame['time_s'] - clip['start']) / (clip['end'] - clip['start'])
    c.rect((24, 552, 976, 555), GRID)
    c.rect((24, 552, 24 + max(1, min(1, progress) * 952), 555), GREEN)


def chart(c, box, frames, clip, current, series, bounds, label):
    """Plot only available history; axes stay fixed throughout the excerpt."""
    x0, y0, x1, y1 = box
    c.text((x0, y0), label, 15, MUTED)
    left, top, right, bottom = x0 + 31, y0 + 32, x1 - 8, y1 - 24

    def xy(t, value):
        return (left + (t - clip['start']) / (clip['end'] - clip['start']) * (right - left),
                bottom - (value - bounds[0]) / (bounds[1] - bounds[0]) * (bottom - top))

    for value in [bounds[0], (bounds[0] + bounds[1]) / 2, bounds[1]]:
        y = xy(clip['start'], value)[1]
        c.line([(left, y), (right, y)], GRID)
        c.text((left - 7, y), f'{value:g}', 11, MUTED, anchor='rm')
    for t in [clip['start'], (clip['start'] + clip['end']) / 2, clip['end']]:
        x = xy(t, 0)[0]
        c.text((x, bottom + 5), f'{t:g}s', 11, MUTED, anchor='mt')
    for event in clip.get('events', []):
        t = event['time_s']
        if clip['start'] <= t <= current['time_s'] + 1e-5:
            x = xy(t, 0)[0]
            for y in range(int(top), int(bottom), 7):
                c.line([(x, y), (x, min(y + 3, bottom))], AMBER)
    history = [f for f in frames if clip['start'] - 1e-5 <= f['time_s'] <= current['time_s']]
    for getter, color in series:
        values = [getter(f) for f in history]
        if any(v < bounds[0] - 1e-5 or v > bounds[1] + 1e-5 for v in values):
            raise ValueError(f'Chart would clip values: {label}')
        c.line([xy(f['time_s'], value) for f, value in zip(history, values)], color, 2)
    x = xy(current['time_s'], 0)[0]
    c.line([(x, top), (x, bottom)], FG)


def draw_vehicle(c, transform, x, y, length, width, yaw, color, outline_only=False):
    def local(dx, dy):
        return transform(x + dx * math.cos(yaw) - dy * math.sin(yaw),
                         y + dx * math.sin(yaw) + dy * math.cos(yaw))
    pts = [local(dx * length / 2, dy * width / 2)
           for dx, dy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    if outline_only:
        c.line(pts + [pts[0]], color)
        return
    c.polygon(pts, color)
    # A windshield indicates heading; geometry and vehicle center remain unchanged.
    if length > 2:
        c.polygon([local(dx * length, dy * width) for dx, dy in
                   [(0.05, -0.36), (0.25, -0.36), (0.25, 0.36), (0.05, 0.36)]], BG)


def birdseye(c, data, frame):
    # Equal metric scales, with ego fixed in X and world Y fixed. All selected
    # synthetic excerpts have straight reference paths along world +X.
    if any(abs(p[1]) > 1e-6 for p in data['reference_path']):
        raise ValueError('This layout requires a straight world-X reference path')
    x0, y0, x1, y1 = 24, 110, 678, 383
    c.rect((x0, y0, x1, y1), PANEL, 12)
    ego = frame['ego']
    pixels_per_m = 12
    origin_x = ego['x'] - 12
    center_y = 243

    def xy(x, y):
        return x0 + (x - origin_x) * pixels_per_m, center_y - y * pixels_per_m

    # Render on a separate canvas and crop, so objects never overwrite telemetry.
    layer = Canvas(*c.font_paths)
    layer.rect((x0, y0, x1, y1), PANEL)
    for x in range(math.floor(origin_x / 5) * 5, math.ceil((origin_x + 55) / 5) * 5, 5):
        px = xy(x, 0)[0]
        layer.line([(px, y0), (px, y1)], '#192b3e')
    for y in range(-10, 11, 5):
        py = xy(0, y)[1]
        layer.line([(x0, py), (x1, py)], '#192b3e')
    half = data['road_half_width_m']
    top, bottom = xy(0, half)[1], xy(0, -half)[1]
    layer.rect((x0, top, x1, bottom), ROAD)
    for y in [top, bottom]:
        layer.line([(x0, y), (x1, y)], '#7891a9', 2)
    for x in range(math.floor(origin_x / 4) * 4, math.ceil((origin_x + 55) / 4) * 4, 4):
        layer.line([xy(x, 0), xy(x + 1.7, 0)], '#52728c')
    if data.get('stop_s') is not None:
        px = xy(data['stop_s'], 0)[0]
        color = GREEN if frame.get('signal') == 'green' else RED
        layer.line([(px, top), (px, bottom)], color, 3)
        layer.text((px, bottom + 8), '停车线', 13, MUTED, anchor='mt')
    history = [xy(f['ego']['x'], f['ego']['y']) for f in data['frames']
               if f['time_s'] <= frame['time_s']]
    layer.line(history, GREEN, 3)
    layer.line([xy(*p[:2]) for p in frame['trajectory']], BLUE, 3)
    for box in frame['obstacles']:
        draw_vehicle(layer, xy, box[0], box[1], box[3], box[4], box[6], AMBER)
    for box in frame['tracks']:
        draw_vehicle(layer, xy, box[0], box[1], box[3], box[4], box[6], '#be9af5', True)
    vehicle = data['vehicle']
    offset = vehicle['rear_to_center']
    draw_vehicle(layer, xy, ego['x'] + offset * math.cos(ego['yaw']),
                 ego['y'] + offset * math.sin(ego['yaw']), vehicle['length'],
                 vehicle['width'], ego['yaw'], GREEN)
    region = tuple(v * SCALE for v in (x0 + 1, y0 + 1, x1 - 1, y1 - 1))
    c.image.paste(layer.image.crop(region), (region[0], region[1]))
    c.text((40, 120), '等比例俯视 · 自车跟随视角', 14, MUTED)
    c.text((660, 120), '+X →', 14, MUTED, anchor='rt')
    for x, color, label in [(40, GREEN, '自车 / 已行驶'), (203, BLUE, '当前规划'),
                            (334, AMBER, '障碍真值'), (465, '#be9af5', '检测跟踪框')]:
        c.line([(x, 363), (x + 20, 363)], color, 3)
        c.text((x + 27, 353), label, 13, MUTED)
    c.line([(594, 330), (654, 330)], FG, 2)
    c.text((624, 305), '5 m', 12, MUTED, anchor='mt')


def synthetic(c, clip, data, frame):
    birdseye(c, data, frame)
    c.rect((694, 110, 976, 383), PANEL, 12)
    c.text((714, 123), '记录状态', 14, MUTED)
    state_color = RED if frame['command']['emergency'] else GREEN
    c.text((714, 150), STATUS.get(frame['status'], frame['status']), 21, state_color, True)
    c.text((714, 195), f"{frame['ego']['speed']:.2f}", 46, FG, True)
    c.text((937, 227), 'm/s', 16, MUTED, anchor='rt')
    c.text((714, 264), '加速度指令', 14, MUTED)
    c.text((954, 264), f"{frame['command']['acceleration']:+.2f} m/s²", 17, FG, anchor='rt')
    c.text((714, 296), '转向指令', 14, MUTED)
    c.text((954, 296), f"{math.degrees(frame['command']['steering']):+.1f}°", 17, FG, anchor='rt')
    if frame['signal'] != 'none':
        text, color = ('绿灯 · 仍需观察行人', GREEN) if frame['signal'] == 'green' else ('红灯 · 停车线约束', RED)
    elif frame['status'] == 'stale_input':
        text, color = '输入超时 · 紧急制动有效', RED
    else:
        text, color = f"已记录跟踪框：{len(frame['tracks'])}", MUTED
    c.text((714, 339), text, 15, color)
    chart(c, (27, 398, 482, 529), data['frames'], clip, frame,
          [(lambda f: f['ego']['speed'], GREEN)], (0, 8), '自车速度 / m/s')
    chart(c, (513, 398, 973, 529), data['frames'], clip, frame,
          [(lambda f: f['command']['acceleration'], AMBER)], (-7, 3), '加速度指令 / m/s²')
    c.text((24, 530), '合成 LiDAR 未模拟光线遮挡 · 逐帧原值，无运动插值', 12, MUTED)


def carla(c, clip, data, frame, camera):
    c.image.paste(camera.resize((640 * SCALE, 360 * SCALE), Image.Resampling.LANCZOS),
                  (24 * SCALE, 111 * SCALE))
    c.rect((36, 123, 254, 152), BG, 5)
    c.text((46, 127), '自车相机 · autopilot OFF', 14, FG)
    c.rect((681, 111, 976, 471), PANEL, 12)
    c.text((698, 121), f"自车 {frame['ego']['speed']:.2f} m/s", 21, GREEN, True)
    c.text((698, 151), f"参考 {frame['target_speed_mps']:.2f} m/s", 15, BLUE)
    chart(c, (694, 177, 963, 299), data['frames'], clip, frame,
          [(lambda f: f['ego']['speed'], GREEN),
           (lambda f: f['target_speed_mps'], BLUE)], (0, 5), '速度 / m/s    实测 · 参考')
    command = frame['command']
    c.text((698, 302), f"油门 {command['throttle']:.2f}", 17, AMBER)
    c.text((947, 302), f"制动 {command['brake']:.2f}", 17, RED, anchor='rt')
    chart(c, (694, 333, 963, 459), data['frames'], clip, frame,
          [(lambda f: f['command']['throttle'], AMBER),
           (lambda f: f['command']['brake'], RED)], (0, 1), '踏板指令 / 0–1')
    event = '前车开始全制动' if frame['time_s'] >= 6 - 1e-5 else '前车尚未触发全制动'
    c.text((25, 481), f"{event}  ·  {STATUS.get(frame['status'], frame['status'])}", 20, FG, True)
    c.text((25, 518), '6.0 s 触发前车全制动 · 相机与控制同帧 · 曲线保留原始变化', 14, MUTED)
    c.text((976, 521), 'CARLA 仿真记录', 14, MUTED, anchor='rt')


def encode(images, path, fps, gifsicle=None, lossy=0):
    # One palette for the whole excerpt prevents palette flicker. Sampling all
    # frames affects colors only, never geometry, values, or frame timing.
    thumbnails = [im.resize((250, 150)) for im in images]
    strip = Image.new('RGB', (250 * 8, 150 * math.ceil(len(images) / 8)), BG)
    for i, thumb in enumerate(thumbnails):
        strip.paste(thumb, ((i % 8) * 250, (i // 8) * 150))
    # Reserve UI colors so a detailed camera image cannot consume the chart's
    # green/blue/red palette entries and make different series look identical.
    adaptive = strip.quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    colors = adaptive.getpalette()[:192 * 3]
    for color in [BG, PANEL, ROAD, FG, MUTED, GRID, GREEN, BLUE, AMBER, RED,
                  '#be9af5', '#192b3e', '#7891a9', '#52728c', '#ffffff', '#000000']:
        rgb, background = ImageColor.getrgb(color), ImageColor.getrgb(PANEL)
        for fraction in [1, .75, .5, .25]:
            colors.extend(round(a * fraction + b * (1 - fraction))
                          for a, b in zip(rgb, background))
    palette = Image.new('P', (1, 1))
    palette.putpalette(colors)
    frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    durations = [round(1000 / fps)] * (len(frames) - 1) + [1200]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
                   loop=0, optimize=False, disposal=1)
    if gifsicle:
        subprocess.run([str(gifsicle), '-O3', f'--lossy={lossy}', '--batch', str(path)],
                       check=True)
        if lossy:
            # Restrict lossy compression to camera pixels. Restore all plots,
            # labels and numerical readouts from the original palette frames.
            restored = []
            with Image.open(path) as optimized:
                for i, original in enumerate(frames):
                    optimized.seek(i)
                    result = original.convert('RGB')
                    result.paste(optimized.convert('RGB').crop((24, 111, 664, 471)), (24, 111))
                    result.paste(images[i].crop((36, 123, 255, 153)), (36, 123))
                    if i == len(frames) - 1:
                        result.paste(images[i].crop((259, 251, 741, 335)), (259, 251))
                    restored.append(result.quantize(palette=palette, dither=Image.Dither.NONE))
            restored[0].save(path, save_all=True, append_images=restored[1:], duration=durations,
                             loop=0, optimize=False, disposal=1)
            subprocess.run([str(gifsicle), '-O3', '--batch', str(path)], check=True)
    with Image.open(path) as gif:
        actual = []
        for index in range(gif.n_frames):
            gif.seek(index)
            actual.append(gif.info['duration'])
        if actual != durations:
            raise ValueError(f'Encoded timing differs: {path.name}')
    return durations


def export(clip, out, regular, bold, gifsicle=None, camera_lossy=0):
    source = ROOT / 'docs/assets/driving' / clip['source']
    data = json.loads(source.read_text())
    clip = dict(clip, events=data['events'])
    selected = [(i, f) for i, f in enumerate(data['frames'])
                if clip['start'] - 1e-5 <= f['time_s'] <= clip['end'] + 1e-5]
    expected = round((clip['end'] - clip['start']) * clip['fps']) + 1
    if len(selected) != expected:
        raise ValueError(f'Unexpected frame count: {clip["name"]}')
    for (_, a), (_, b) in zip(selected, selected[1:]):
        if abs(b['time_s'] - a['time_s'] - 1 / clip['fps']) > 1e-5:
            raise ValueError('Source timing differs from requested playback rate')
    camera_dir = None
    video = None
    with tempfile.TemporaryDirectory(prefix='driving-gif-') as tmp:
        if 'video' in data:
            video = ROOT / 'docs/assets' / data['video']['file']
            if sha256(video) != data['video']['sha256']:
                raise ValueError('Source video hash mismatch')
            camera_dir = Path(tmp)
            subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vsync', '0',
                            str(camera_dir / '%06d.png')], check=True)
            if len(list(camera_dir.glob('*.png'))) != len(data['frames']):
                raise ValueError('Video and telemetry frame counts differ')
        images = []
        poster = min(selected, key=lambda item: abs(item[1]['time_s'] - clip['poster']))[0]
        for index, frame in selected:
            c = Canvas(regular, bold)
            header(c, clip, frame, camera_dir is not None)
            if camera_dir is None:
                synthetic(c, clip, data, frame)
            else:
                if abs(frame['video_time_s'] - index / clip['fps']) > 1e-5:
                    raise ValueError('Camera and telemetry timestamps differ')
                with Image.open(camera_dir / f'{index + 1:06d}.png') as camera:
                    carla(c, clip, data, frame, camera)
            im = c.finish()
            images.append(im)
            if index == poster:
                im.save(out / f"{clip['name']}.jpg", quality=92)
        # Keep the last data frame visible with an explicit loop message.
        c.rect((260, 252, 740, 334), BG, 12, GREEN)
        c.text((500, 263), '节选结束 · 即将从头回放', 24, FG, True, anchor='mt')
        c.text((500, 303), 'END OF EXCERPT  /  REPLAY RESTARTS', 13, MUTED, anchor='mt')
        images.append(c.finish())
        target = out / f"{clip['name']}.gif"
        durations = encode(images, target, clip['fps'], gifsicle, camera_lossy if video else 0)
        # Contact sheets are review artifacts, separate from published assets.
        sheet = Image.new('RGB', (WIDTH * 2, HEIGHT * 2), BG)
        for i, position in enumerate([0, len(images) // 3, 2 * len(images) // 3, len(images) - 2]):
            sheet.paste(images[position], ((i % 2) * WIDTH, (i // 2) * HEIGHT))
        sheet.save(out / f"{clip['name']}_contact.jpg", quality=90)
    result = {
        'name': clip['name'], 'title': clip['title'], 'file': target.name,
        'poster': f"{clip['name']}.jpg", 'sha256': sha256(target),
        'bytes': target.stat().st_size, 'size_px': [WIDTH, HEIGHT],
        'source': str(source.relative_to(ROOT)), 'source_sha256': sha256(source),
        'source_kind': 'carla' if video else 'synthetic', 'detector': data['detector'],
        'clip_start_s': selected[0][1]['time_s'], 'clip_end_s': selected[-1][1]['time_s'],
        'recorded_fps': clip['fps'], 'playback_rate': 1,
        'interpolation': 'none', 'source_frame_indices': [i for i, _ in selected],
        'gifsicle_lossy': camera_lossy if video and gifsicle else 0,
        'source_times_s': [f['time_s'] for _, f in selected],
        'frame_durations_ms': durations, 'loop_notice_ms': 1200,
        'duration_s': sum(durations) / 1000, 'metrics_scope': 'full_source_run',
        'metrics': data['metrics'],
    }
    if video:
        result['source_video'] = str(video.relative_to(ROOT))
        result['source_video_sha256'] = sha256(video)
    print(f"{target.name}: {result['bytes'] / 1024**2:.2f} MiB, {result['duration_s']:.1f}s", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--font', type=Path,
                        default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'))
    parser.add_argument('--bold-font', type=Path,
                        default=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'))
    parser.add_argument('--gifsicle', type=Path, help='Optional GIF optimizer executable')
    parser.add_argument('--camera-lossy', type=int, default=0,
                        help='gifsicle color compression for CARLA only; does not change timing')
    args = parser.parse_args()
    for font in [args.font, args.bold_font]:
        if not font.is_file():
            parser.error(f'Font not found: {font}; provide --font / --bold-font')
    if args.camera_lossy < 0 or (args.camera_lossy and not args.gifsicle):
        parser.error('--camera-lossy requires --gifsicle and a nonnegative value')
    args.out.mkdir(parents=True, exist_ok=False)
    clips = [export(clip, args.out, args.font, args.bold_font, args.gifsicle, args.camera_lossy)
             for clip in CLIPS]
    manifest = {
        'schema_version': 1, 'generator': 'tools/export_driving_gifs.py',
        'generator_sha256': sha256(Path(__file__)),
        'font_sha256': sha256(args.font), 'bold_font_sha256': sha256(args.bold_font),
        'gifsicle_version': subprocess.check_output([str(args.gifsicle), '--version'], text=True)
        .splitlines()[0] if args.gifsicle else None,
        'pillow_version': pillow_version,
        'ffmpeg_version': subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0],
        'notes': [
            'Selected excerpts, not independent trials or a success-rate estimate.',
            'Recorded sample timing at 1x simulation time; not inference throughput.',
            'No smoothing, interpolation, or edits to controls; history-only plots.',
            'Synthetic BEV shows ground-truth obstacles and detector/tracker boxes separately.',
            'End notice adds 1.2 seconds; the final source sample retains its normal duration.',
        ],
        'clips': clips,
    }
    (args.out / 'index.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
