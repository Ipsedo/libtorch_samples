#include <torch/torch.h>
#include <iostream>

#include "data/read_mnist.h"
#include "models/conv_models.h"
#include "models/linear_models.h"

void test_tensor() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;
    torch::Tensor cuda_tensor = tensor.cuda();
    std::cout << cuda_tensor.sum(1) << std::endl;

    auto a = torch::rand({200,500}).cuda();
    auto b = torch::rand({500, 300}).cuda();
    auto c = torch::matmul(a, b);
    std::cout << c << std::endl;
    std::cout << c.slice(0, 50, 75) << std::endl;
}

void test_load_mnist() {
    data_set mnist = read_mnist("./datasets/downloaded/mnist/train-images-idx3-ubyte",
            "./datasets/downloaded/mnist/train-labels-idx1-ubyte");

    torch::Tensor x = get_data(mnist);
    torch::Tensor y = get_labels(mnist);

    std::cout << x.size(0) << ", " << x.size(1) << ", " << x.size(2) << std::endl;
    std::cout << y.size(0) << std::endl;

    auto net = MNIST_ConvNet();
    //auto net = MNIST_LinearNet();

    auto x_train = x.slice(0, 0, 50000);
    auto y_train = y.slice(0, 0, 50000);
    auto x_dev = x.slice(0, 50001, 60001);
    auto y_dev = y.slice(0, 50001, 60001);

    net.to(torch::Device(torch::kCUDA));

    auto optim = torch::optim::SGD(net.parameters(), 1e-3);

    int batch_size = 4;
    int nb_batch  = (int) ceil(double(x_train.size(0)) / double(batch_size));

    int nb_epoch = 30;

    for (int e = 0; e < nb_epoch; e++) {

        float sum_loss = 0.f;

        net.train();

        for (int b_idx = 0; b_idx < nb_batch; b_idx++) {
            int i_min = batch_size * b_idx;
            int i_max = batch_size * (b_idx + 1);
            i_max = (int) (i_max > x_train.size(0) ? x_train.size(0) : i_max);

            auto x_batch = x_train.slice(0, i_min, i_max).to(torch::Device(torch::kCUDA));
            auto y_batch = y_train.slice(0, i_min, i_max).to(torch::Device(torch::kCUDA));

            optim.zero_grad();

            auto pred = net.forward(x_batch);

            auto loss = torch::nll_loss(pred, y_batch);

            loss.backward();
            optim.step();

            sum_loss += loss.item().toDouble();
        }
        std::cout << "Epoch " << e << ", mean loss = " << (sum_loss / nb_batch) << std::endl;

        net.eval();
        std::cout << (net.forward(x_dev.cuda()).argmax(1) == y_dev.cuda()).sum().item().toDouble()
            << " / " << x_dev.size(0) << std::endl;
    }
}

int main() {
    test_load_mnist();
    //test_tensor();
}