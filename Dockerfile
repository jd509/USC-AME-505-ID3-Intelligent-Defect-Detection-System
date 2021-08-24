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

RUN sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN sudo wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.0-455.23.05-1_amd64.deb
RUN sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
RUN sudo apt-get update
RUN sudo apt-get -y install cuda


#Setting up the locales
RUN sudo apt-get update && sudo apt-get install locales
RUN sudo locale-gen en_US en_US.UTF-8
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
RUN export LANG=en_US.UTF-8

COPY . /app/defect_detector/

WORKDIR /app/defect_detector/install/

RUN sudo pip3 install -r requirements.txt

RUN sudo apt-get install --reinstall libxcb-xinerama0

RUN echo "All dependencies have been installed for ID3!"

WORKDIR /app/defect_detector/scripts/

CMD [ "python3 /app/defect_detector/scripts/user_interface.py" ]

