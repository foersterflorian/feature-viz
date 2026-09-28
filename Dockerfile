# feature-viz container image. Rationale: DECISIONS.md §9.
#
#   docker compose build
#   docker compose up                                        # GPU + webcam
#   docker compose -f compose.yaml -f compose.sample.yaml up # GPU, sample clip
#   docker compose -f compose.yaml -f compose.cpu.yaml up    # CPU, sample clip
#
# The image runs exactly the versions pdm.lock pins - the ones the test suite
# checked - on a digest-pinned Python base. torch's PyPI wheels bring their own
# CUDA runtime, so no CUDA base image is needed; the host provides only the
# NVIDIA driver (>= 580 for CUDA 13.0) and the CDI device.

# Same digest in both stages: the venv is built against this exact Python.
ARG PYTHON_IMAGE=python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f

# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE} AS build

# pdm only turns the lockfile into a hashed requirements list; it does not
# reach the runtime image. Pinned so that the export itself is reproducible.
RUN pip install --no-cache-dir pdm==2.29.2

WORKDIR /build
COPY pyproject.toml pdm.lock ./
RUN pdm export --prod -o requirements.txt \
 && python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.txt

# The weights are part of the image, at a fixed path (§11): downloaded into
# the working directory at runtime, they would land in an overlay layer that
# `--rm` discards, and the demonstrator could not start without network.
# The checksum pins the exact file the test suite ran against.
# --chmod: a URL source lands as 0600 root, unreadable for the runtime user.
ADD --checksum=sha256:9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef --chmod=644 \
    https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt \
    /opt/feature-viz/weights/yolo26n.pt

# --------------------------------------------------------------------------
FROM ${PYTHON_IMAGE}

LABEL org.opencontainers.image.title="feature-viz" \
      org.opencontainers.image.source="https://github.com/foersterflorian/feature-viz" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later"

# Runtime libraries the opencv-python wheel links against (it is the GUI
# build, not -headless, because DISPLAY_MODE=window exists outside the
# container).
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0t64 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/feature-viz/weights /opt/feature-viz/weights
COPY src /opt/feature-viz/src
COPY assets/sample.mp4 /opt/feature-viz/assets/sample.mp4
COPY LICENSE /opt/feature-viz/LICENSE

# Unprivileged. Camera access comes from the host's video group, added by
# compose (group_add), not from running as root.
# ultralytics needs YOLO_CONFIG_DIR to exist and creates Ultralytics/ inside.
RUN useradd --system --uid 10001 --no-create-home feature-viz \
 && install -d -o feature-viz /tmp/ultralytics
USER feature-viz
WORKDIR /opt/feature-viz

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/opt/feature-viz/src \
    PYTHONUNBUFFERED=1 \
    WEIGHTS=/opt/feature-viz/weights/yolo26n.pt \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLO_OFFLINE=1

# YOLO_CONFIG_DIR: the user has no home, and ultralytics would warn on every
# start that it cannot write settings.json. YOLO_OFFLINE: no DNS probes and no
# usage events leave the container; a demonstrator must not depend on, or
# report to, the network at a talk.

EXPOSE 8080

# /healthz is 503 until the first frame and whenever frames stop (§7). The
# start period covers model loading on the CPU profile.
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request, sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=2).status != 200)"]

ENTRYPOINT ["python", "-m", "feature_viz.demonstrator"]
