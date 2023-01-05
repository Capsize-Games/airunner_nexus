FROM ubuntu:latest as base_image
RUN apt-get update && apt-get install -y curl git libxml2

FROM base_image as python_install
RUN apt-get install -y python3 python3-pip

FROM python_install as cuda_download
RUN curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
RUN mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN curl -O https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2204-11-7-local_11.7.0-515.43.04-1_amd64.deb
RUN cp /var/cuda-repo-ubuntu2204-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/
RUN apt-get update
RUN apt-get -y install cuda

FROM cuda_download as cuda_install
RUN chmod +x cuda_11.7.0_515.43.04_linux.run
RUN sh cuda_11.7.0_515.43.04_linux.run

FROM python_install as cuda_install
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install torch torchvision torchaudio
