#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    torch::Tensor cuda_tensor = tensor.cuda();
    std::cout << cuda_tensor.sum(1) << std::endl;
}