#!/usr/bin/env python3
"""Transcode existing experiment GIFs for the website, preserving their timing."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'format=duration:stream=width,height,nb_frames', '-of', 'json', str(path)],
                            check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('docs/screenshots'))
    parser.add_argument('--out', type=Path, default=Path('docs/assets'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name in ('kitti', 'vggt'):
        source = args.source / f'demo_{name}.gif'
        output = args.out / f'{name}.mp4'
        info = probe(source)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(source), '-an', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                        '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
                        '-fps_mode', 'vfr', '-movflags', '+faststart', str(output)], check=True)
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                        '-ss', str(float(info['format']['duration']) / 2), '-i', str(output),
                        '-frames:v', '1', '-q:v', '2', str(args.out / f'{name}_poster.jpg')], check=True)
        manifest[name] = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                          'source_metadata': info, 'video_metadata': probe(output),
                          'note': 'Existing experiment visualization; playback timing is not processing latency.'}
    (args.out / 'gallery.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
