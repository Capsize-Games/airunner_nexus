FROM ubuntu:latest as base_image
RUN apt-get update && apt-get install -y curl git libxml2

FROM base_image as python_install
RUN apt-get install -y python3 python3-pip

FROM python_install as cuda_download
RUN curl -O https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run

FROM cuda_download as cuda_install
RUN chmod +x cuda_11.7.0_515.43.04_linux.run
RUN sh cuda_11.7.0_515.43.04_linux.run

FROM cuda_install as install_requirements
WORKDIR /app
RUN pip3 install -r requirements.txt
WORKDIR /app/lib
# install each folder in lib
RUN for d in */ ; do cd $d && python3 setup.py install && cd .. ; done

FROM install_requirements as run
WORKDIR /app
CMD ["python", "server.py"]
