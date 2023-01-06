FROM ubuntu:latest as base_image
ENV TZ=America/Denver
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update \
    && apt-get upgrade -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && dpkg --add-architecture i386 \
    && apt-get update \
    && apt install libtinfo6 -y \
    && apt-get install -y git wget curl vim software-properties-common gcc-9 g++-9 \
    && apt-get install -y bash \
    && apt-get install -y build-essential \
    && apt-get install -y libssl-dev \
    && apt-get install -y python3 \
    && apt-get install -y python3-pip \
    && apt-get install -y libffi-dev \
    && apt-get install -y ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/ \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
ENV DISPLAY :0

FROM base_image as cuda_download
RUN curl -O https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run

FROM cuda_download as cuda_install
RUN sh cuda_11.7.0_515.43.04_linux.run --silent --toolkit --toolkitpath=/usr/local/cuda-11.7 \
    && rm -f cuda_11.7.0_515.43.04_linux.run

FROM cuda_install as pip_install
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install torch torchvision torchaudio \
    && pip3 install sqlalchemy \
    && pip3 install -r requirements.txt

FROM pip_install as opencv
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv-python
