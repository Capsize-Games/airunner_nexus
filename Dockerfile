FROM ubuntu:latest as base_image
RUN apt-get update && apt-get install -y curl git libxml2

FROM base_image as python_install
RUN apt-get install -y python3 python3-pip

FROM python_install as cuda_download
RUN curl -O https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run

FROM cuda_download as cuda_install
RUN sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.7 \
    && rm -f cuda_11.7.0_515.43.04_linux.run

FROM cuda_install as pip_install
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt \
    && pip3 install torch torchvision torchaudio
