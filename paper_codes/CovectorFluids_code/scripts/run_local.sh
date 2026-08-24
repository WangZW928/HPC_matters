#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dependency_lib="${repo_root}/.deps/ubuntu24.04-x86_64/usr/lib/x86_64-linux-gnu"

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <2d|3d> <method> <experiment>" >&2
    exit 1
fi

case "$1" in
    2d) executable="Covector2D" ;;
    3d) executable="Covector3D" ;;
    *)
        echo "The first argument must be 2d or 3d." >&2
        exit 1
        ;;
esac

if [[ ! -x "${repo_root}/build/${executable}" ]]; then
    echo "${executable} is not built; run scripts/build_local.sh first." >&2
    exit 1
fi

export LD_LIBRARY_PATH="${dependency_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${repo_root}/build"
exec "./${executable}" "$2" "$3"

