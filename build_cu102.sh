#!/usr/bin/env bash
#!/bin/bash
BUILD_DIR=build_cu102
if [ -d "./${BUILD_DIR}" ]; then
	rm -rf ./${BUILD_DIR}
fi
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}
cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-8 -DCUDNN_INCLUDE_DIR=/usr/local/cuda-10.2/include -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DCUDNN_LIBRARY=/usr/local/cuda-10.2/lib64/libcudnn.so --std=c++14 ..
make -j 8
cd ..
