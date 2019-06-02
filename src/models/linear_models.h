//
// Created by samuel on 02/06/19.
//

#ifndef LIBTORCH_SAMPLES_LINEAR_MODELS_H
#define LIBTORCH_SAMPLES_LINEAR_MODELS_H

#include <torch/torch.h>

struct MNIST_LinearNet : torch::nn::Module {
    MNIST_LinearNet() {
        l1 = register_module("l1", torch::nn::Linear(784, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = x.view({x.size(0), 28 * 28});
        x = torch::softmax(l1->forward(x), 1);
        return x;
    }

    torch::nn::Linear l1{nullptr};
};

#endif //LIBTORCH_SAMPLES_LINEAR_MODELS_H
