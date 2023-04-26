# Base image -> Ubuntu 20.04 with cuda 12.1.0
FROM nvcr.io/nvidia/cuda:12.1.0-base-ubuntu20.04

# To get video driver libraries at runtime (libnvidia-encode.so/libnvcuvid.so)
ENV NVIDIA_DRIVER_CAPABILITIES $NVIDIA_DRIVER_CAPABILITIES,video,compute,graphics,utility

# Select fastest Ubuntu mirror for downloading packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl && \
    curl -s http://mirrors.ubuntu.com/mirrors.txt | xargs -n1 -I {} sh -c 'echo `curl -r 0-102400 -s -w %{speed_download} -o /dev/null {}/ls-lR.gz` {}' |sort -g -r |head -1| awk '{ print $2  }'

# Install required packages.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    wget python3 python3-pip python3.8-venv python-is-python3 python3-dev git cmake ninja-build build-essential && \
    python3 -m pip install --upgrade build pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get autoremove -y

# Install miniconda and set libmamba as the solver
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    conda install -n base conda-libmamba-solver && \
    conda config --set solver libmamba

# Copy Conda environment file
COPY whiteboxcartoon.yml /tmp/

# Create Conda environment
RUN conda env create -f /tmp/whiteboxcartoon.yml

# Activate whiteboxcartoon environment
SHELL ["conda", "run", "-n", "whiteboxcartoon", "/bin/bash", "-c"]

