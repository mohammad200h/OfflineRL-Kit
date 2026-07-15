# FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt install -y mesa-utils
#### Instaling python 3.10
RUN apt-get update && apt-get install -y python3.10
RUN apt-get update && apt-get install -y python3.10-dev
RUN apt-get update && apt-get install -y python3.10-venv
RUN apt-get update && apt-get install -y python3.10-distutils
RUN apt-get update && apt-get install -y python3-pip

# EGL / OpenGL libs for headless MuJoCo rendering
RUN apt-get update && apt-get install -y \
    libegl1-mesa libgl1-mesa-glx libglib2.0-0 libfuse2 \
    wget unzip patchelf \
    libosmesa6-dev libglfw3 libglew-dev gcc g++

ENV MUJOCO_GL=egl

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install "gymnasium[mujoco]>=1.0.0" "gymnasium-robotics>=1.3.0" "mujoco>=3.0.0" h5py \
    && python3 -c "import mujoco; import gymnasium; print('mujoco', mujoco.__version__, 'gymnasium', gymnasium.__version__)"

###### Data Collection #######
RUN python3 -m pip install pynput pygame

###### WorkDir #######
WORKDIR /workspace
# COPY . /tmp/
# RUN cd /tmp && python3 -m pip install -e .
###### Shell #######
CMD /bin/bash
