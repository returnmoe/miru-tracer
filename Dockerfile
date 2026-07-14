# syntax=docker/dockerfile:1.7

# CUDA 12.6 is the compatibility-first default. Release/CI builds override
# these three arguments together for the CUDA 13.0 variant.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04@sha256:8aef630a54bc5c5146ae5ce68e6af5caa3df0fb690bb91544175c91f307e4356
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG EXPECTED_CUDA=12.6

FROM ${CUDA_IMAGE} AS builder
ARG TORCH_INDEX
ARG EXPECTED_CUDA
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates python3.12 python3.12-venv && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src/ src/
RUN python3.12 -m venv /opt/miru && \
    /opt/miru/bin/pip install --upgrade pip && \
    /opt/miru/bin/pip install torch==2.12.1 --index-url "${TORCH_INDEX}" && \
    /opt/miru/bin/pip install '.[gpu]' -c constraints.txt && \
    /opt/miru/bin/pip check && \
    /opt/miru/bin/python -c \
      "import torch; assert torch.version.cuda == '${EXPECTED_CUDA}', torch.version.cuda"

FROM ${CUDA_IMAGE} AS runtime
ARG EXPECTED_CUDA
ARG DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/miru/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MIRU_SERVER_NAME=0.0.0.0 \
    MIRU_SERVER_PORT=7860 \
    MIRU_AUTO_START_UI=1 \
    MIRU_SSH_ENABLE=auto \
    HOME=/home/miru

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl openssh-server python3.12 tini util-linux && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin miru && \
    passwd -l root && \
    mkdir -p /run/miru /run/sshd /root/.ssh /home/miru/.cache/miru-tracer && \
    chmod 0700 /root/.ssh && \
    chown -R miru:miru /home/miru

COPY --from=builder /opt/miru /opt/miru
COPY docker/sshd-miru.conf /etc/ssh/sshd_config.d/90-miru-hardening.conf
COPY docker/healthcheck.sh /usr/local/bin/miru-healthcheck
COPY entrypoint.sh /usr/local/bin/miru-entrypoint
RUN chmod 0755 /usr/local/bin/miru-entrypoint /usr/local/bin/miru-healthcheck && \
    ln -sf /opt/miru/bin/miru-tracer /usr/local/bin/miru-tracer && \
    ln -sf /opt/miru/bin/miru-tracer-fit-lens /usr/local/bin/miru-tracer-fit-lens && \
    ln -sf /opt/miru/bin/miru-tracer-convert-lens /usr/local/bin/miru-tracer-convert-lens && \
    /opt/miru/bin/pip check && \
    /opt/miru/bin/python -c \
      "import torch; assert torch.version.cuda == '${EXPECTED_CUDA}', torch.version.cuda" && \
    ssh-keygen -A && \
    /usr/sbin/sshd -t && \
    rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub

LABEL org.opencontainers.image.title="Miru Tracer" \
      io.returnmoe.miru-tracer.cuda="${EXPECTED_CUDA}"

EXPOSE 22 7860
VOLUME ["/home/miru/.cache/miru-tracer"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/miru-healthcheck"]
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/miru-entrypoint"]
CMD ["miru-auto"]
