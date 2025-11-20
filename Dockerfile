# Multi-stage Dockerfile for Miru Tracer
# Optimized for production deployment with GPU support

# Stage 1: Builder
FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04 AS builder

# Install Python and build dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .

# Install torch separately with specific CUDA version
RUN pip install "torch~=2.9.1" --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# Filter out torch and audioop-lts from requirements.txt and install the rest
RUN grep -vE "^(torch|audioop-lts)" requirements.txt > requirements.filtered.txt && \
    pip install -r requirements.filtered.txt --break-system-packages

# ---

# Stage 2: Runtime
FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MIRU_SERVER_NAME="0.0.0.0" \
    MIRU_SERVER_PORT=7860

# Install Python runtime
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd && \
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
ssh-keygen -A

# Create SSH directory for root
RUN mkdir -p /root/.ssh && \
chmod 700 /root/.ssh

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/dist-packages /usr/local/lib/python3.12/dist-packages

# Set working directory
WORKDIR /app

# Copy application code
COPY src/ ./

# Expose Gradio port
EXPOSE 7860

# Expose SSH port
EXPOSE 22

# Run the application
CMD ["python3", "app.py"]
