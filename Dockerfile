# Multi-stage Dockerfile for Miru Tracer
#
# Build args:
#   TORCH_INDEX - PyTorch wheel index. Default is CUDA 13.0; pass
#                 "https://download.pytorch.org/whl/cpu" for a CPU-only image.

# Stage 1: Builder
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS builder

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip

WORKDIR /build

# Install torch first (pinned wheel index), then the package itself
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install torch --index-url "$TORCH_INDEX" --break-system-packages && \
    pip install . --break-system-packages

# ---

# Stage 2: Runtime
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MIRU_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH (optional; enabled at runtime with MIRU_SSH_ENABLE=1)
RUN mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh

# Copy installed packages (miru_tracer is installed as a package)
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages
COPY --from=builder /usr/local/bin/miru-tracer /usr/local/bin/miru-tracer

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# SSH (optional) and Gradio
EXPOSE 22
EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]
