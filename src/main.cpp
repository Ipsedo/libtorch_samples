#include <torch/torch.h>
#include <iostream>

#include "data/read_mnist.h"

void test_tensor() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    torch::Tensor cuda_tensor = tensor.cuda();
    std::cout << cuda_tensor.sum(1) << std::endl;

    auto a = torch::rand({200,500}).cuda();
    auto b = torch::rand({500, 300}).cuda();
    auto c = torch::matmul(a, b);
    std::cout << c << std::endl;
}

void test_load_mnist() {
    data_set mnist = read_mnist("./datasets/downloaded/mnist/train-images-idx3-ubyte",
            "./datasets/downloaded/mnist/train-labels-idx1-ubyte");

    torch::Tensor x = get_data(mnist);
    torch::Tensor y = get_labels(mnist);

    std::cout << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << std::endl;
    std::cout << y.size(0) << std::endl;

    int idx_data = 60000;

    std::cout << "Digit == " << y[idx_data].item().toDouble() << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (x[idx_data][i][j].item().toDouble() > 0.5 ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    test_load_mnist();
}