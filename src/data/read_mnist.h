//
// Created by samuel on 01/06/19.
//

#ifndef LIBTORCH_SAMPLES_READ_MNIST_H
#define LIBTORCH_SAMPLES_READ_MNIST_H

#include "repr.h"

#define ERROR_LOADING_MNIST 1000

data_set read_mnist(std::string image_file_name, std::string label_file_name);

#endif //LIBTORCH_SAMPLES_READ_MNIST_H
