FROM continuumio/miniconda3:4.9.2
ENV PYTHONIOENCODING "utf-8"
ARG MINI_CONDA_SH="Miniconda3-latest-Linux-x86_64.sh"

ARG USER="aisg"
ARG WORK_DIR="/home/${USER}"

WORKDIR $WORK_DIR

RUN apt-get --allow-releaseinfo-change update && \
    apt-get -y install curl sudo && \
    apt-get -y install libgl1-mesa-dev && \
    apt-get -y install libsm6 libxext6 libxrender-dev libglib2.0-0 && \
    apt-get clean

# Create a new user for the miniconda environment
RUN groupadd --gid 2222 $USER && \
    adduser --disabled-password --uid 2222 --gid 2222 $USER && \
    chown -R 2222:2222 $WORK_DIR

# Enable this user to use sudo without password
RUN echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER $USER

ARG CONDA_PATH="${WORK_DIR}/miniconda3/bin"
RUN curl -O https://repo.anaconda.com/miniconda/$MINI_CONDA_SH && \
    chmod +x $MINI_CONDA_SH && \
    ./Miniconda3-latest-Linux-x86_64.sh -b && \
    rm $MINI_CONDA_SH

ENV PATH $CONDA_PATH:$PATH

RUN conda install python=3.6.8
COPY requirements_train.txt requirements_train.txt
RUN pip3 install -r requirements_train.txt --timeout 100 --no-cache-dir

COPY src src
# copy any other files here

ENV PYTHONPATH "${PYTHONPATH}:${CONDA_PATH}:${WORK_DIR}/src/life3_biotech/modeling/EfficientDet"
CMD [ "python3", "-m" , "src.train_model"]
