FROM nvcr.io/nvidia/pytorch:24.12-py3-igpu
RUN pip uninstall -y opencv
RUN pip install transformers flask flask-cors
RUN apt update && apt install -y htop git

### COMPILE OPENCV
RUN cd ~ && \
    git clone --depth=1 https://github.com/opencv/opencv.git && \
    git clone --depth=1 https://github.com/opencv/opencv_contrib.git 


RUN apt install -y cmake libjpeg-dev libjpeg8-dev libjpeg-turbo8-dev libpng-dev libtiff-dev libglew-dev libavcodec-dev libavformat-dev libswscale-dev libgtk2.0-dev libgtk-3-dev libcanberra-gtk* python3-pip libxvidcore-dev libx264-dev libtbb-dev libdc1394-dev libxine2-dev libv4l-dev v4l-utils qv4l2 libtesseract-dev libpostproc-dev libswresample-dev libvorbis-dev libfaac-dev libmp3lame-dev libtheora-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenblas-dev libblas-dev liblapack-dev liblapacke-dev libeigen3-dev gfortran libgoogle-glog-dev libgflags-dev 

RUN cd ~/opencv && \ 
mkdir build && \
cd build && \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=OFF \
-D CUDA_ARCH_BIN=8.7 \
-D CUDA_ARCH_PTX="sm_87" \
-D WITH_CUDA=ON \
-D WITH_CUDNN=ON \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D WITH_QT=OFF \
-D WITH_OPENMP=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D WITH_PROTOBUF=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D PYTHON3_PACKAGES_PATH=/usr/lib/python3/dist-packages \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D BUILD_EXAMPLES=OFF ..

RUN cd ~/opencv/build && \
    make -j4 && \
    make install && \
    make clean
    
RUN apt update
# RUN apt install -y software-properties-common
# RUN add-apt-repository -y ppa:deadsnakes/ppa
# RUN apt install -y python3.11 python3.11-venv

WORKDIR /workspace
COPY backend /workspace
COPY frontend/dist /workspace/dist

# RUN python3 -m venv env
# RUN env/bin/pip install -r requirements.txt

EXPOSE 5000
#CMD ["sleep","infinity"]
CMD [ "python","main.py" ]



