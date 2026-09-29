#!/usr/bin/env bash
# Start the demonstration setup: the container with GPU and webcam
# (`docker compose up`, DECISIONS.md §9). Stop with Ctrl+C.
#
#   ./start-container.sh
#
# Checks first what would otherwise fail late or silently in front of an
# audience: a missing .env, a missing image (Compose would quietly build it,
# downloading about 6 GB), a missing camera. Other variants: README.md.
set -euo pipefail

cd "$(dirname "$(readlink -f "$0")")"

fail() {
    echo "[error] $*" >&2
    exit 1
}

command -v docker >/dev/null || fail "docker not found; host setup: README.md, section Container"
docker info >/dev/null 2>&1 || fail "cannot reach the Docker daemon (running? user in group docker?)"

[ -f .env ] || fail "no .env; create it from the template and set CAMERA_DEVICE and VIDEO_GID:
    cp .env.example .env"

# The tag comes from compose.yaml, so a version bump needs no change here.
image="$(docker compose config --images)"
docker image inspect "$image" >/dev/null 2>&1 || fail "image $image not found; load the archive or build it:
    zstd -dc ${image/:/-}.tar.zst | docker load
    docker compose build"

# Shell variables win over .env, as they do for Compose.
camera="${CAMERA_DEVICE:-$(sed -n 's/^CAMERA_DEVICE=//p' .env)}"
[ -e "$camera" ] || fail "camera $camera not found (CAMERA_DEVICE in .env; list: ls /dev/v4l/by-id/)
    Without a camera: docker compose -f compose.yaml -f compose.sample.yaml up"

port="${HOST_PORT:-$(sed -n 's/^HOST_PORT=//p' .env)}"
echo "[info] $image, camera $camera"
echo "[info] stream: http://localhost:${port:-8080}/ (Ctrl+C stops)"
exec docker compose up
