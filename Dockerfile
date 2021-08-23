FROM nvidia/cudagl:11.4.1-devel-ubuntu20.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
lsb-release \
net-tools \
iputils-ping \
apt-utils \
build-essential \
psmisc \
vim-gtk \
mongodb \
scons \
bison \
wget \
flex \
git \
sudo python3-pip \
keyboard-configuration \
 && rm -rf /var/lib/apt/lists/*

ENV USERNAME bot
RUN adduser --ingroup sudo --disabled-password --gecos "" --shell /bin/bash --home /home/$USERNAME $USERNAME
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN bash -c 'echo $USERNAME:bot | chpasswd'
ENV HOME /home/$USERNAME
USER $USERNAME

COPY . /app/defect_detector/

WORKDIR /app/defect_detector/install/

RUN ls

RUN ./install_dependencies.sh

