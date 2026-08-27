#!/usr/bin/env python3
"""无需第三方 Python 包，直接将 SPH 二进制帧渲染为 GIF。"""

import argparse
import glob
import math
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

MAGIC = 0x53504831
EARTH_RADIUS = 1.0
WATER_MID_RADIUS = 0.5 * (1.012 + 1.075)
MOON_RADIUS = 60.3
MOON_OMEGA = 0.0021480
WIDTH, HEIGHT = 1200, 600


def frame_number(path):
    return int(path.rsplit("_", 1)[1][:-4])


def read_particles(path, stride):
    with open(path, "rb") as stream:
        magic, count, sim_time = struct.unpack("<IIf", stream.read(12))
        if magic != MAGIC:
            raise ValueError(f"{path} 不是有效的 SPH1 文件")
        positions = stream.read(count * 12)
        velocities = stream.read(count * 12)

    particles = []
    for index in range(0, count, stride):
        offset = index * 12
        x, y, z = struct.unpack_from("<fff", positions, offset)
        vx, vy, vz = struct.unpack_from("<fff", velocities, offset)
        particles.append((x, y, z, math.sqrt(vx * vx + vy * vy + vz * vz)))
    return sim_time, particles


def put_pixel(image, x, y, color):
    if 0 <= x < WIDTH and 0 <= y < HEIGHT:
        offset = (y * WIDTH + x) * 3
        image[offset:offset + 3] = bytes(color)


def fill_circle(image, cx, cy, radius, color):
    radius2 = radius * radius
    for y in range(cy - radius, cy + radius + 1):
        span = int(math.sqrt(max(0, radius2 - (y - cy) ** 2)))
        for x in range(cx - span, cx + span + 1):
            put_pixel(image, x, y, color)


def draw_circle(image, cx, cy, radius, color):
    for degree in range(720):
        angle = degree * math.pi / 360.0
        put_pixel(image, int(cx + radius * math.cos(angle)),
                  int(cy + radius * math.sin(angle)), color)


def speed_color(speed):
    # 速度大小：深蓝 -> 浅蓝 -> 白。避免绿/红交替造成方向上的误解。
    value = min(max(speed / 0.20, 0.0), 1.0)
    return (int(20 + 225 * value), int(95 + 150 * value), 255)


def displacement_color(radius):
    # 蓝色表示向内，白色接近中面，红色表示向外。
    value = min(max((radius - WATER_MID_RADIUS) / 0.08, -1.0), 1.0)
    if value < 0.0:
        t = value + 1.0
        return (int(35 + 220 * t), int(95 + 160 * t), 255)
    return (255, int(255 - 180 * value), int(255 - 210 * value))


def render_frame(path, output, stride, radial_scale, moon_speedup, color_mode):
    sim_time, particles = read_particles(path, stride)
    image = bytearray([255]) * (WIDTH * HEIGHT * 3)

    # 左侧为水层近景。缩小实体地球，并放大径向位移以看清薄水层流动。
    earth_cx, earth_cy, scale = 315, 300, 205
    fill_circle(image, earth_cx, earth_cy, int(0.76 * scale), (210, 232, 249))
    yaw, pitch = 0.65, 0.32
    for x, y, z, speed in particles:
        radius = max(math.sqrt(x * x + y * y + z * z), 1.0e-8)
        display_radius = EARTH_RADIUS + radial_scale * (radius - EARTH_RADIUS)
        x, y, z = (x * display_radius / radius, y * display_radius / radius,
                   z * display_radius / radius)
        projected_x = math.cos(yaw) * x - math.sin(yaw) * y
        projected_y = (math.sin(pitch) * (math.sin(yaw) * x + math.cos(yaw) * y)
                       + math.cos(pitch) * z)
        px = int(earth_cx + scale * projected_x)
        py = int(earth_cy - scale * projected_y)
        color = (speed_color(speed) if color_mode == "speed"
                 else displacement_color(radius))
        put_pixel(image, px, py, color)
        put_pixel(image, px + 1, py, color)

    # 右侧为完整地月轨道，背景保持纯白，不绘制坐标轴或网格。
    orbit_cx, orbit_cy, orbit_r = 900, 300, 225
    draw_circle(image, orbit_cx, orbit_cy, orbit_r, (175, 175, 175))
    fill_circle(image, orbit_cx, orbit_cy, 13, (40, 120, 208))
    angle = MOON_OMEGA * sim_time * moon_speedup
    moon_x = int(orbit_cx + orbit_r * math.cos(angle))
    moon_y = int(orbit_cy - orbit_r * math.sin(angle))
    fill_circle(image, moon_x, moon_y, 9, (115, 115, 125))

    with open(output, "wb") as stream:
        stream.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode("ascii"))
        stream.write(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="output/frame_*.bin")
    parser.add_argument("--output", default="earth_moon.gif")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--radial-scale", type=float, default=2.0)
    parser.add_argument("--moon-speedup", type=float, default=500.0)
    parser.add_argument("--color", choices=("displacement", "speed"),
                        default="displacement",
                        help="默认按径向位移着色；speed 表示速度大小")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        raise SystemExit("未找到 ffmpeg，无法编码 GIF")

    files = sorted(glob.glob(args.pattern), key=frame_number)
    if not files:
        raise SystemExit(f"没有找到帧：{args.pattern}")
    if args.max_frames > 0 and len(files) > args.max_frames:
        step = (len(files) - 1) / (args.max_frames - 1)
        files = [files[round(index * step)] for index in range(args.max_frames)]

    output = Path(args.output).resolve()
    with tempfile.TemporaryDirectory(prefix="earth_moon_gif_") as temp:
        for index, path in enumerate(files):
            frame_path = Path(temp) / f"frame_{index:04d}.ppm"
            render_frame(path, frame_path, args.stride, args.radial_scale,
                         args.moon_speedup, args.color)
            print(f"\r渲染帧 {index + 1}/{len(files)}", end="", flush=True)
        print("\n正在编码 GIF...")
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps),
            "-i", f"{temp}/frame_%04d.ppm", "-vf",
            "split[s0][s1];[s0]palettegen=max_colors=128[p];"
            "[s1][p]paletteuse=dither=bayer", str(output)
        ], check=True)
    print(f"已生成 {output}")


if __name__ == "__main__":
    main()
