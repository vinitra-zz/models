# Dockerfile to run ONNXRuntime with TensorRT integration

FROM nvidia/cuda:9.0-cudnn7-devel

# Ubuntu-core
RUN apt-get update && \
  apt-get install -y --no-install-recommends sudo \
    libopenblas-base \
    python3 \
    python3-pip \
    wget \
    build-essential curl libcurl4-openssl-dev libssl-dev python3-dev git

RUN pip3 install numpy

RUN ln -s $(which python3) /usr/bin/python && \
    ln -s $(which pip3) /usr/bin/pip

RUN echo "Install TensorRT." && \
  wget -qO tensorrt.deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64/nvinfer-runtime-trt-repo-ubuntu1604-4.0.1-ga-cuda9.0_1-1_amd64.deb && \
  dpkg -i tensorrt.deb && \
  apt-get update && \
  apt-get install -y --allow-downgrades libnvinfer-dev && \
  rm tensorrt.deb

# Build the latest cmake
WORKDIR /code
RUN wget https://cmake.org/files/v3.12/cmake-3.12.3.tar.gz;
RUN tar zxf cmake-3.12.3.tar.gz

WORKDIR /code/cmake-3.12.3
RUN ./configure --system-curl
RUN make
RUN sudo make install

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

WORKDIR /code

# Prepare onnxruntime Repo
RUN git clone --recursive https://github.com/Microsoft/onnxruntime
WORKDIR /code/onnxruntime

EXPOSE 8888

# Build onnxruntime with TensorRT
RUN git fetch
RUN git checkout trt_execution_provider
RUN /bin/sh build.sh --use_cuda --cuda_home /usr/local/cuda \
--cudnn_home /usr/share/doc/libcudnn7 --use_tensorrt --tensorrt_home /usr/src/tensorrt \
--config Release --enable_pybind
