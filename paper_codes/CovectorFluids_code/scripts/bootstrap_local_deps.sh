#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
install_root="${repo_root}/.deps/ubuntu24.04-x86_64"

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    echo "This bootstrap script supports Ubuntu 24.04 on x86_64 only." >&2
    exit 1
fi

if ! command -v apt-get >/dev/null || ! command -v dpkg-deb >/dev/null; then
    echo "apt-get and dpkg-deb are required." >&2
    exit 1
fi

packages=(
    libsnappy1v5
    libblosc1
    libblosc-dev
    libboost1.83-dev
    libboost-atomic1.83.0
    libboost-atomic1.83-dev
    libboost-filesystem1.83.0
    libboost-system1.83.0
    libboost-system1.83-dev
    libboost-filesystem1.83-dev
    libboost-filesystem-dev
    libboost-iostreams1.83.0
    libboost-system-dev
    libimath-3-1-29t64
    libimath-dev
    liblog4cplus-2.0.5t64
    libtbbmalloc2
    libtbbbind-2-5
    libtbb12
    libopenvdb10.0t64
    libtbb-dev
    libopenvdb-dev
)

package_cache=$(mktemp -d /tmp/covector-deps.XXXXXX)
trap 'rm -rf "${package_cache}"' EXIT

mkdir -p "${install_root}"
(
    cd "${package_cache}"
    apt-get download "${packages[@]}"
)

for package_file in "${package_cache}"/*.deb; do
    dpkg-deb -x "${package_file}" "${install_root}"
done

echo "Dependencies installed in ${install_root}"

