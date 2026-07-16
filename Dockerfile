# syntax=docker/dockerfile:1.7

# CUDA 13.0 is PyTorch's stable default and supports current RunPod GPUs,
# including Blackwell. Release/CI builds override these arguments together for
# the legacy CUDA 12.6 variant used by older drivers and supported pre-Turing
# GPUs.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG EXPECTED_CUDA=13.0
ARG CUDA_TOOLKIT_VERSION=13.0.2
ARG CUDA_BINDINGS_SPEC=cuda-bindings==13.0.3
ARG CUDA_DNN_SPEC=nvidia-cudnn-cu13==9.20.0.48
ARG CUDA_SPARSELT_SPEC=nvidia-cusparselt-cu13==0.8.1
ARG CUDA_NCCL_SPEC=nvidia-nccl-cu13==2.29.7
ARG CUDA_NVSHMEM_SPEC=nvidia-nvshmem-cu13==3.4.5

# Keep the CUDA/PyTorch environment in one stage. Copying /opt/miru out of a
# builder duplicates several gigabytes in BuildKit and exhausts hosted runners.
FROM ${CUDA_IMAGE} AS runtime
ARG TORCH_INDEX
ARG EXPECTED_CUDA
ARG CUDA_TOOLKIT_VERSION
ARG CUDA_BINDINGS_SPEC
ARG CUDA_DNN_SPEC
ARG CUDA_SPARSELT_SPEC
ARG CUDA_NCCL_SPEC
ARG CUDA_NVSHMEM_SPEC
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl openssh-server python3.12 python3.12-venv \
        tini util-linux && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin miru && \
    passwd -l root && \
    mkdir -p /run/miru /run/sshd /root/.ssh /home/miru/.cache/miru-tracer && \
    chmod 0700 /root/.ssh && \
    chown -R miru:miru /home/miru

COPY pyproject.toml README.md LICENSE constraints.txt ./
RUN python3.12 -m venv /opt/miru && \
    /opt/miru/bin/pip install --upgrade pip

# PyTorch's CUDA wheels provide the user-space libraries. Install their large
# requirements in bounded layers so registries and cloud runtimes never need
# to transfer the entire GPU environment as one multi-gigabyte blob.
RUN /opt/miru/bin/pip install \
      "cuda-toolkit[cublas,cudart,cupti,nvtx]==${CUDA_TOOLKIT_VERSION}" \
      --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install \
      "cuda-toolkit[cufft,cufile,curand]==${CUDA_TOOLKIT_VERSION}" \
      --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install \
      "cuda-toolkit[cusolver,cusparse,nvjitlink,nvrtc]==${CUDA_TOOLKIT_VERSION}" \
      --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install "${CUDA_DNN_SPEC}" --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install \
      "${CUDA_BINDINGS_SPEC}" \
      "${CUDA_SPARSELT_SPEC}" \
      "${CUDA_NCCL_SPEC}" \
      "${CUDA_NVSHMEM_SPEC}" \
      --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install triton==3.7.1 \
      --index-url "${TORCH_INDEX}"
RUN /opt/miru/bin/pip install torch==2.12.1 --no-deps \
      --index-url "${TORCH_INDEX}"

COPY src/ src/
RUN /opt/miru/bin/pip install '.[gpu]' -c constraints.txt && \
    /opt/miru/bin/pip check && \
    /opt/miru/bin/python -c \
      "import torch; assert torch.version.cuda == '${EXPECTED_CUDA}', torch.version.cuda"

ENV PATH=/opt/miru/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MIRU_SERVER_PORT=7860 \
    MIRU_AUTO_START_UI=1 \
    MIRU_SSH_ENABLE=auto \
    HOME=/home/miru

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
ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/usr/local/bin/miru-entrypoint"]
CMD ["miru-auto"]
