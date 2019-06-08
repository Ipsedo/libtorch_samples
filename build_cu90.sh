#!/bin/bash
if [ -d "./build" ]; then
	rm -rf ./build
fi
mkdir build
cd build
cmake -DCUDA_HOST_COMPILER=/usr/bin/gcc-5 -DCUDNN_INCLUDE_DIR=/usr/local/cuda-9.0/include -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-9.0 -DCUDNN_LIBRARY=/usr/local/cuda-9.0/lib64/libcudnn.so ..
make
cd ..