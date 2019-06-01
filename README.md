# LibTorch Samples

*Samuel Berrien*

## Installation

First step you need to download and install [CUDA 9.2](https://developer.nvidia.com/cuda-92-download-archive)

Next download `libtorch` on [official website](https://pytorch.org/) with the correct version depending your CUDA version :
```
$ # For CUDA 9.0 (works also fine with CUDA 9.2 !)
$ wget https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip
```
Then place the extracted zip in the folder of your choice (`/opt/libtorch` for example).
Finally clone this repo and edit its CMake file by replacing the line :
```cmake
set(CMAKE_PREFIX_PATH "/home/samuel/Programs/libtorch")
```
with the correct libtorch root dir (`/opt/libtorch` for example) :
```cmake
set(CMAKE_PREFIX_PATH "/opt/libtorch")
```

Build this repository with following commands :
```
$ cd /path/to/libtorch_samples
$ sh build.sh
$ ./build/libtorch_samples
```

## Note
If your CUDA installation version is different from 9.2 or is not located in `/usr/local`, you need to check if it is a compatible version for libtorch 
and you must edit `libtorch_samples/CMakeLists.txt` at the line :
```cmake
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.2")
```
and set the correct CUDA root dir.