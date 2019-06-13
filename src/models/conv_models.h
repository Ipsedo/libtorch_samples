//
// Created by samuel on 01/06/19.
//

#ifndef LIBTORCH_SAMPLES_CONV_MODELS_H
#define LIBTORCH_SAMPLES_CONV_MODELS_H

#include <torch/torch.h>

struct MNIST_ConvNet : torch::nn::Module {
    MNIST_ConvNet() {
        c1 = register_module("c1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 8, {3, 3})));
        c2 = register_module("c2", torch::nn::Conv2d(torch::nn::Conv2dOptions(8, 16, {3, 3})));

        l1 = register_module("l1", torch::nn::Linear(16 * 5 * 5, 10));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(torch::max_pool2d(c1->forward(x), {2, 2}, {2, 2}));
        x = torch::relu(torch::max_pool2d(c2->forward(x), {2, 2}, {2, 2}));
        x = x.view({x.size(0), 16 * 5 * 5});
        x = torch::softmax(l1->forward(x), 1);
        return x;
    }

    torch::nn::Conv2d c1{nullptr}, c2{ nullptr};
    torch::nn::Linear l1{nullptr};
};

struct CIFAR_ConvNet : torch::nn::Module {
    CIFAR_ConvNet() {
        int c1_ft_m = 9;
        int c2_ft_m = 32;

        linear_dim = int(
                pow(int(int(((32.0 - (conv_kernel_sizes[0] - conv_kernel_sizes[0] % 2))
                / float(maxpool_size[0])) - (conv_kernel_sizes[1] - conv_kernel_sizes[1] % 2))
                / float(maxpool_size[1])), 2.0)
                * c2_ft_m);

        c1 = register_module("c1",
                torch::nn::Conv2d(torch::nn::Conv2dOptions(3, c1_ft_m,
                        {conv_kernel_sizes[0], conv_kernel_sizes[0]})));
        c2 = register_module("c2",
                torch::nn::Conv2d(torch::nn::Conv2dOptions(c1_ft_m, c2_ft_m,
                        {conv_kernel_sizes[1], conv_kernel_sizes[1]})));

        l1 = register_module("l1", torch::nn::Linear(linear_dim, linear_dim * 2));
        l2 = register_module("l2", torch::nn::Linear(linear_dim * 2, nb_class));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::elu(torch::max_pool2d(c1->forward(x),
                {maxpool_size[0], maxpool_size[0]},
                {maxpool_size[0], maxpool_size[0]}));

        x = torch::elu(torch::max_pool2d(c2->forward(x),
                {maxpool_size[1], maxpool_size[1]},
                {maxpool_size[1], maxpool_size[1]}));

        x = x.flatten(1, -1);

        x = l1->forward(x);
        x = torch::elu(x);
        x = l2->forward(x);

        x = torch::softmax(x, 1);

        return x;
    }

    torch::nn::Conv2d c1{nullptr}, c2{nullptr};
    torch::nn::Linear l1{nullptr}, l2{nullptr};

    int linear_dim, nb_class = 10;
    int conv_kernel_sizes[2]{3, 3};
    int maxpool_size[2]{2, 2};
};

#endif //LIBTORCH_SAMPLES_CONV_MODELS_H
