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

RUN echo "deb https://repo.download.nvidia.com/jetson/common r36.3 main" >  /etc/apt/sources.list.d/nvidia.list \
       && echo "deb https://repo.download.nvidia.com/jetson/t234 r36.3 main" >>  /etc/apt/sources.list.d/nvidia.list \
       && apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
       && mkdir -p /opt/nvidia/l4t-packages/ && touch /opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall

RUN \
    apt-get update && apt-get install -y wget tar lbzip2 binutils xz-utils zstd qemu-user-static cpio git && cd /tmp/ && wget https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v3.0/release/jetson_linux_r36.3.0_aarch64.tbz2 && \
    tar xf jetson_linux_r36.3.0_aarch64.tbz2 && \
    cd Linux_for_Tegra && \
    mkdir -p /tmp/Linux_for_Tegra/rootfs/boot/ && \
    mkdir -p /tmp/Linux_for_Tegra/rootfs/usr/bin && \
    mkdir -p /tmp/Linux_for_Tegra/rootfs/etc && touch /tmp/Linux_for_Tegra/rootfs/etc/resolv.conf && \
    sed -i 's/config.tbz2\"/config.tbz2\" --exclude=etc\/hosts --exclude=etc\/hostname/g' apply_binaries.sh && \
    sed -i 's/CheckPackage qemu-user-static/#CheckPackage qemu-user-static/g' tools/l4t_update_initrd.sh && \
    sed -i 's/trap CleanupVirEnv/#trap CleanupVirEnv/g' tools/l4t_update_initrd.sh&& \
    sed -i 's|cp /usr/bin/qemu-aarch64-static|#cp /usr/bin/qemu-aarch64-static|g' tools/l4t_update_initrd.sh && \
    sed -i 's|^UpdateInitrd|#UpdateInitrd|g' tools/l4t_update_initrd.sh && \
    sed -i 's|^UpdateBackToBaseInitrd|#UpdateBackToBaseInitrd|g' tools/l4t_update_initrd.sh && \
    sed -i 's|cp /etc/resolv.conf|#cp /etc/resolv.conf|g' tools/l4t_update_initrd.sh && \
    sed -i 's|mv "${LDK_ROOTFS_DIR}/etc/resolv.conf"|cp "${LDK_ROOTFS_DIR}/etc/resolv.conf"|g' tools/l4t_update_initrd.sh && \
    sed -i 's|  PrepareVirEnv|#PrepareVirEnv|g' tools/l4t_update_initrd.sh && \
    sed -i 's/install --owner=root --group=root \"${QEMU_BIN}\" \"${L4T_ROOTFS_DIR}\/usr\/bin\/\"/#install --owner=root --group=root \"${QEMU_BIN}\" \"${L4T_ROOTFS_DIR}\/usr\/bin\/\"/g' nv_tegra/nv-apply-debs.sh && \
    sed -i 's/chroot . \//  /g' nv_tegra/nv-apply-debs.sh && \
    cd /tmp/Linux_for_Tegra/ && ./apply_binaries.sh -r / --target-overlay && cd .. && \
    rm -rf Linux_for_Tegra && \
    echo "/usr/lib/aarch64-linux-gnu/tegra" > /etc/ld.so.conf.d/nvidia-tegra.conf && ldconfig

ENV LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra
ENV UDEV=1

RUN apt update && apt-get install -y -o Dpkg::Options::="--force-confdef" nvidia-l4t-cuda nvidia-cuda build-essential cuda-nvcc-12-2
RUN wget https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-dla-compiler/nvidia-l4t-dla-compiler_36.3.1-20240516220919_arm64.deb && dpkg -i nvidia-l4t-dla-compiler_36.3.1-20240516220919_arm64.deb

WORKDIR /workspace
COPY backend /workspace
COPY frontend/dist /workspace/dist

# RUN python3 -m venv env
# RUN env/bin/pip install -r requirements.txt

EXPOSE 5000
CMD [ "python","main.py" ]



