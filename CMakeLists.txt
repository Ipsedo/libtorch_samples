cmake_minimum_required(VERSION 3.0)
project(libtorch_samples)

set(CMAKE_PREFIX_PATH "/opt/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

file(GLOB_RECURSE SRC "src/*.cpp" "src/*.c")

include_directories("libs/tqdm/include/")

add_executable(libtorch_samples ${SRC})
target_link_libraries(libtorch_samples "${TORCH_LIBRARIES}")
