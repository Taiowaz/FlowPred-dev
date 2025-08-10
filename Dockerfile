FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Shanghai
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# Copy sources.list and pip.conf
COPY sources.list /etc/apt/sources.list
COPY pip.conf /etc/pip.conf

# Install system dependencies and Python 3.9
RUN apt update && \
    apt install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y \
    python3.9 \
    python3.9-distutils \
    python3-pip && \
    # Set Python 3.9 as default
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 && \
    # Cleanup
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Ensure pip uses the correct Python version
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA 11.8
RUN pip3 install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install other Python dependencies using Chinese mirrors
RUN pip3 install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --ignore-installed blinker

# Copy the rest of the application
COPY . .

# Make the startup script executable
RUN chmod +x start.sh

# Expose the port the app runs on
EXPOSE 5002

# Verify Python version
RUN python3 --version  # 应该显示 Python 3.9.21

# Command to run the application
CMD ["./start.sh"]