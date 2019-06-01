//
// Created by samuel on 01/06/19.
//

#ifndef LIBTORCH_SAMPLES_CONV_MODELS_H
#define LIBTORCH_SAMPLES_CONV_MODELS_H

#include <torch/torch.h>

struct MNIST_ConvNet : torch::nn::Module {
    MNIST_ConvNet() {
        c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, {3, 3})));
        c2 = register_module("c2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, {3, 3})));

        l1 = register_module("l1", torch::nn::Linear(64 * 5 * 5, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(c1->forward(x), {2, 2}, {2, 2}));
        x = torch::relu(torch::max_pool2d(c2->forward(x), {2, 2}, {2, 2}));
        x = x.view({-1, 64 * 5 * 5});
        x = torch::softmax(l1->forward(x), 1);
        return x;
    }

    torch::nn::Conv2d c1{nullptr}, c2{ nullptr};
    torch::nn::Linear l1{nullptr};
};

#endif //LIBTORCH_SAMPLES_CONV_MODELS_H
