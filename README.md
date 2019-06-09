# LibTorch Samples

*Samuel Berrien*

## Installation

First you need to download and install [CUDA 9.0](https://developer.nvidia.com/cuda-90-download-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) (I use version 7.4.1).

Next download `libtorch` on [official website](https://pytorch.org/) with the correct version depending your CUDA version :
```bash
$ # For CUDA 9.0
$ wget https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip
```
Then place the extracted zip in the folder of your choice (`/opt/libtorch` for example).

Clone this repository with the following command :
```bash
$ git clone https://github.com/Ipsedo/libtorch_samples.git
```

## Building
Build this repository with following commands :
```bash
$ cd /path/to/libtorch_samples
$ # CUDA 9.0
$ bash build_cu90.sh
$ # CUDA 9.2
$ bash build_cu92.sh
```

## Run
You must run the executable in `libtorch_samples` root dir
```bash
$ cd /path/to/libtorch_samples
$ # CUDA 9.0
$ ./build_cu90/libtorch_samples
$ # CUDA 9.2
$ ./build_cu92/libtorch_samples
```

## Note
If you want to copy libtorch in another directory you must edit `libtorch_samples/CMakeLists.txt` at the line :
```cmake
set(CMAKE_PREFIX_PATH "/opt/libtorch")
```
and set the correct libtorch root directory.