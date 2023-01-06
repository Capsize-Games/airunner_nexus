FROM nvidia/cuda:12.0.0-base-ubuntu22.04 as base_image
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
    && apt-get install -y libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/ \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

FROM base_image as pip_install
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install torch torchvision torchaudio \
    && pip3 install -r requirements.txt \
    && pip3 install opencv-python
