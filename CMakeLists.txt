cmake_minimum_required(VERSION 3.0)
project(libtorch_samples)

set(CC gcc-7)
set(CXX g++-7)

set(CMAKE_PREFIX_PATH "/home/samuel/Programs/libtorch")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-9.2")

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

file(GLOB_RECURSE SRC "src/*.cpp" "src/*.c")

add_executable(libtorch_samples ${SRC})
target_link_libraries(libtorch_samples "${TORCH_LIBRARIES}")
