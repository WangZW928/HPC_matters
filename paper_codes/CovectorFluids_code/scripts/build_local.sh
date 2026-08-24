#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
dependency_root="${repo_root}/.deps/ubuntu24.04-x86_64/usr"
dependency_lib="${dependency_root}/lib/x86_64-linux-gnu"

if [[ ! -f "${dependency_lib}/libtbb.so" || ! -f "${dependency_lib}/libopenvdb.so" ]]; then
    echo "Local dependencies are missing; run scripts/bootstrap_local_deps.sh first." >&2
    exit 1
fi

cmake -S "${repo_root}" -B "${repo_root}/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DTBB_ROOT_DIR="${dependency_root}" \
    -DOPENVDB_INCLUDE_DIRS="${dependency_root}/include" \
    -DOPENVDB_LIBRARIES="${dependency_lib}/libopenvdb.so" \
    -DBOOST_ROOT="${dependency_root}" \
    -DBoost_NO_SYSTEM_PATHS=ON \
    -DCMAKE_PREFIX_PATH="${dependency_root}"

cmake --build "${repo_root}/build" -j "${COVECTOR_BUILD_JOBS:-4}"

