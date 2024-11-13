FROM nvidia/cuda:12.1.0-base-ubuntu22.04

LABEL maintainer="Jason Hughes"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc cmake sudo
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git python3-dev libglib2.0.0 wget python3-pybind11 vim tmux
RUN apt-get install -y --no-install-recommends libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python3-pip

# Add user
ARG GID=1000
ARG UID=1000
ENV USER jason
RUN addgroup --gid $GID $USER 
RUN useradd --system --create-home --shell /bin/bash --groups sudo -p "$(openssl passwd -1 ${USER})" --uid $UID --gid $GID $USER
WORKDIR /home/$USER

RUN chown -R $USER: /home/$USER
USER $USER

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install opencv-python \
 transformers \
 numpy \
 matplotlib \
 jupyterlab \
 notebook 

RUN pip3 install pytorch-lightning
RUN pip3 install h5py

RUN echo 'export PS1="\[$(tput setaf 2; tput bold)\]\u\[$(tput setaf 7)\]@\[$(tput setaf 3)\]\h\[$(tput setaf 7)\]:\[$(tput setaf 4)\]\W\[$(tput setaf 7)\]$ \[$(tput sgr0)\]"' >> /home/$USER/.bashrc
RUN echo 'export PATH=$PATH:/home/jasonah/.local/bin' >> /home/$USER/.bashrc

WORKDIR /home/$USER
CMD ["bash"]

