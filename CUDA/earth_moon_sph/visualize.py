#!/usr/bin/env python3
"""显示 CUDA 后端导出的 SPH 帧，或直接生成轻量 GIF。"""

import argparse
import glob
import struct
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

MAGIC = 0x53504831
MOON_RADIUS = 60.3
MOON_OMEGA = 0.0021480
EARTH_PHYSICAL_RADIUS = 1.0
EARTH_DISPLAY_RADIUS = 0.82


def read_frame(path: str):
    with open(path, "rb") as stream:
        magic, count, sim_time = struct.unpack("<IIf", stream.read(12))
        if magic != MAGIC:
            raise ValueError(f"{path} 不是有效的 SPH1 文件")
        pos = np.fromfile(stream, dtype=np.float32, count=count * 3).reshape(-1, 3)
        vel = np.fromfile(stream, dtype=np.float32, count=count * 3).reshape(-1, 3)
        rho = np.fromfile(stream, dtype=np.float32, count=count)
        press = np.fromfile(stream, dtype=np.float32, count=count)
    return sim_time, pos, vel, rho, press


def exaggerate_radial_displacement(pos, factor):
    """仅放大可视化中的径向形变，不修改仿真数据。"""
    radius = np.linalg.norm(pos, axis=1)
    direction = pos / np.maximum(radius[:, None], 1.0e-8)
    display_radius = EARTH_PHYSICAL_RADIUS + factor * (radius - EARTH_PHYSICAL_RADIUS)
    return direction * display_radius[:, None]


def remove_3d_background(axis):
    axis.set_axis_off()
    axis.grid(False)
    axis.set_facecolor("white")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", nargs="?", default="output/frame_*.bin")
    parser.add_argument("--stride", type=int, default=8, help="每隔多少粒子绘制一个")
    parser.add_argument("--color", choices=("speed", "pressure", "density"),
                        default="speed")
    parser.add_argument("--interval", type=int, default=50, help="帧间隔，毫秒")
    parser.add_argument("--moon-speedup", type=float, default=500.0,
                        help="仅加速月球可视化相位；物理计算不受影响")
    parser.add_argument("--radial-scale", type=float, default=2.0,
                        help="放大水层径向形变，仅影响显示")
    parser.add_argument("--gif", help="输出 GIF 路径；省略时打开交互窗口")
    parser.add_argument("--fps", type=int, default=15, help="GIF 帧率")
    parser.add_argument("--max-frames", type=int, default=160,
                        help="GIF 最多使用多少帧，0 表示全部")
    args = parser.parse_args()
    files = sorted(glob.glob(args.pattern), key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    if not files:
        raise SystemExit(f"没有找到帧：{args.pattern}")
    if args.gif and args.max_frames > 0 and len(files) > args.max_frames:
        selected = np.linspace(0, len(files) - 1, args.max_frames, dtype=int)
        files = [files[index] for index in selected]

    fig = plt.figure(figsize=(13, 6), facecolor="white")
    ax = fig.add_subplot(121, projection="3d")
    system_ax = fig.add_subplot(122, projection="3d")
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(EARTH_DISPLAY_RADIUS * np.cos(u) * np.sin(v),
                    EARTH_DISPLAY_RADIUS * np.sin(u) * np.sin(v),
                    EARTH_DISPLAY_RADIUS * np.cos(v),
                    color="#2878d0", alpha=0.16, linewidth=0, shade=False)
    scatter = ax.scatter([], [], [], s=3, cmap="turbo", alpha=0.9,
                         depthshade=False)
    moon_direction = ax.scatter([], [], [], s=70, color="gray", label="月球方向")
    direction_orbit = np.linspace(0.0, 2.0 * np.pi, 240)
    direction_radius = 1.48
    ax.plot(direction_radius * np.cos(direction_orbit),
            direction_radius * np.sin(direction_orbit),
            np.zeros_like(direction_orbit), "--", color="gray", alpha=0.5)
    ax.set(xlim=(-1.58, 1.58), ylim=(-1.58, 1.58), zlim=(-1.58, 1.58))
    ax.set_box_aspect((1, 1, 1))
    ax.set_title("地球水层近景")
    ax.legend(loc="upper left")
    remove_3d_background(ax)

    orbit = np.linspace(0.0, 2.0 * np.pi, 360)
    system_ax.plot(MOON_RADIUS * np.cos(orbit), MOON_RADIUS * np.sin(orbit),
                   np.zeros_like(orbit), color="silver", label="月球轨道")
    system_ax.scatter([0.0], [0.0], [0.0], s=120, color="#2878d0", label="地球")
    moon = system_ax.scatter([], [], [], s=70, color="gray", label="月球")
    limit = MOON_RADIUS * 1.12
    system_ax.set(xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit))
    system_ax.set_box_aspect((1, 1, 1))
    system_ax.legend(loc="upper left")
    remove_3d_background(system_ax)

    def update(frame_index):
        sim_time, pos, vel, rho, press = read_frame(files[frame_index])
        pos = pos[::args.stride]
        pos = exaggerate_radial_displacement(pos, args.radial_scale)
        if args.color == "speed":
            color = np.linalg.norm(vel[::args.stride], axis=1)
        elif args.color == "pressure":
            color = press[::args.stride]
        else:
            color = rho[::args.stride]
        scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        scatter.set_array(color)
        # 真实月球周期远长于这段 SPH 仿真。只加速显示相位，让轨道运动可见；
        # CUDA 后端仍使用未加速的物理时间和角速度。
        visual_angle = MOON_OMEGA * sim_time * args.moon_speedup
        moon_direction._offsets3d = (
            [direction_radius * np.cos(visual_angle)],
            [direction_radius * np.sin(visual_angle)], [0.0])
        moon._offsets3d = ([MOON_RADIUS * np.cos(visual_angle)],
                           [MOON_RADIUS * np.sin(visual_angle)], [0.0])
        ax.set_title(f"地球水层近景  t={sim_time:.4f}  frame={frame_index}")
        system_ax.set_title(f"地月系统轨道视图（月球显示相位 x{args.moon_speedup:g}）")
        return scatter, moon_direction, moon

    animation = FuncAnimation(fig, update, frames=len(files),
                              interval=args.interval, repeat=not args.gif,
                              blit=False)
    if args.gif:
        print(f"正在生成 {args.gif}：{len(files)} 帧，每帧约绘制 "
              f"{read_frame(files[0])[1].shape[0] // args.stride} 个粒子")
        animation.save(args.gif, writer=PillowWriter(fps=args.fps), dpi=100)
        print(f"已生成 {args.gif}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
