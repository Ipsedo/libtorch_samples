#!/bin/bash
BUILD_DIR=build_cu90
if [ -d "./${BUILD_DIR}" ]; then
	rm -rf ./${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 -DCUDNN_INCLUDE_DIR=/usr/local/cuda-9.0/include -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 -DCUDNN_LIBRARY=/usr/local/cuda-9.0/lib64/libcudnn.so --std=c++11 ..
make -j 8
cd ..
