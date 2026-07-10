# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:8a7e7cc04fd3e2bd787f7f24e22d5d119aa590d429b50c95dfe12b3abe52f48b

FROM ${PYTHON_IMAGE} AS builder
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

COPY pyproject.toml README.md LICENSE constraints.txt ./
COPY src/ src/
RUN python -m venv /opt/miru && \
    /opt/miru/bin/pip install --upgrade pip && \
    /opt/miru/bin/pip install torch==2.12.1 --index-url "${TORCH_INDEX}" && \
    /opt/miru/bin/pip install '.[gpu]' -c constraints.txt

FROM ${PYTHON_IMAGE} AS runtime
ENV PATH=/opt/miru/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MIRU_SERVER_NAME=0.0.0.0 \
    MIRU_SERVER_PORT=7860 \
    HOME=/home/miru

RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server tini util-linux curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 --shell /usr/sbin/nologin miru && \
    mkdir -p /var/run/sshd /root/.ssh /home/miru/.cache/miru-tracer && \
    chmod 700 /root/.ssh && \
    chown -R miru:miru /home/miru

COPY --from=builder /opt/miru /opt/miru
COPY entrypoint.sh /usr/local/bin/miru-entrypoint
RUN chmod 0755 /usr/local/bin/miru-entrypoint

EXPOSE 22 7860
VOLUME ["/home/miru/.cache/miru-tracer"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail --silent http://127.0.0.1:7860/ >/dev/null || exit 1
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/miru-entrypoint"]
CMD ["python", "-m", "miru_tracer"]
