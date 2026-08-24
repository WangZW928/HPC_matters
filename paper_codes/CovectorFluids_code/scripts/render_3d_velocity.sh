#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dependency_root="${repo_root}/.deps/ubuntu24.04-x86_64/usr"
dependency_lib="${dependency_root}/lib/x86_64-linux-gnu"
input_dir="${1:-${repo_root}/Out_3D/TrefoilKnot/SF}"
output_dir="${2:-${repo_root}/visualizations/TrefoilKnot/SF_vorticity}"
first_frame="${3:-0}"
last_frame="${4:-269}"
ppm_dir=$(mktemp -d /tmp/covector-vorticity.XXXXXX)
trap 'rm -rf "${ppm_dir}"' EXIT

mkdir -p "${repo_root}/build/tools" "${output_dir}/frames"

g++ -std=c++17 -O3 \
    -I"${dependency_root}/include" \
    "${repo_root}/tools/render_vdb_velocity.cpp" \
    -L"${dependency_lib}" -Wl,-rpath,"${dependency_lib}" \
    -lopenvdb -lImath -ltbb \
    -o "${repo_root}/build/tools/render_vdb_velocity"

LD_LIBRARY_PATH="${dependency_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${repo_root}/build/tools/render_vdb_velocity" \
    "${input_dir}" "${ppm_dir}" "${first_frame}" "${last_frame}" \
    > "${output_dir}/max_vorticity.csv"

python3 - "${ppm_dir}" "${output_dir}" "${first_frame}" "${last_frame}" <<'PY'
from pathlib import Path
import sys
from PIL import Image, ImageDraw, features

ppm_dir = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
first = int(sys.argv[3])
last = int(sys.argv[4])
frames_dir = output_dir / "frames"
images = []

for frame in range(first, last + 1):
    with Image.open(ppm_dir / f"vorticity_{frame:04d}.ppm") as source:
        image = source.copy()
    image.save(frames_dir / f"vorticity_{frame:04d}.png", optimize=True)
    images.append(image)

if features.check("webp"):
    images[0].save(
        output_dir / "vorticity_animation.webp",
        save_all=True,
        append_images=images[1:],
        duration=42,
        loop=0,
        quality=88,
        method=6,
    )

selected = [first, (first + last) // 2, last]
for frame in selected:
    images[frame - first].save(output_dir / f"preview_{frame:04d}.png", optimize=True)

sheet = Image.new("RGB", (images[0].width * 3, images[0].height + 24), "black")
draw = ImageDraw.Draw(sheet)
for column, frame in enumerate(selected):
    sheet.paste(images[frame - first], (column * images[0].width, 24))
    draw.text((column * images[0].width + 6, 5), f"frame {frame}", fill="white")
sheet.save(output_dir / "contact_sheet.png", optimize=True)
PY

{
    echo "# Trefoil Knot vorticity visualization"
    echo
    echo "- Source: three staggered velocity grids per frame from ${input_dir}."
    echo "- Frames: ${first_frame} through ${last_frame}."
    echo "- Top half: maximum vorticity magnitude projected along z (top view)."
    echo "- Bottom half: maximum vorticity magnitude projected along y (side view)."
    echo "- Color scale: fixed to the maximum vorticity magnitude in frame ${first_frame}; square-root transfer function."
    echo "- frames/: lossless PNG sequence."
    echo "- vorticity_animation.webp: 24 fps animated preview when Pillow WebP support is available."
    echo "- max_vorticity.csv: maximum magnitude for every frame."
} > "${output_dir}/README.md"

echo "Visualization written to ${output_dir}"
