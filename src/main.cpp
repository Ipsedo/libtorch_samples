#include <torch/torch.h>
#include <iostream>

#include "data/read_mnist.h"
#include "models/conv_models.h"

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

    int idx_data = 60000;

    std::cout << "Digit == " << y[idx_data].item().toDouble() << std::endl;
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            std::cout << (x[idx_data][i][j].item().toDouble() > 0.5 ? "#" : ".") << " ";
        }
        std::cout << std::endl;
    }

    auto net = std::make_shared<MNIST_ConvNet>();

    x = x.unsqueeze(1);
    x = x.to(at::kFloat);
    y = y.to(at::kLong).squeeze(1);

    net->to(torch::Device(torch::kCUDA));

    auto optim = torch::optim::SGD(net->parameters(), 1e-3);

    int batch_size = 16;
    int nb_batch  = (int) ceil(double(x.size(0)) / double(batch_size));

    int nb_epoch = 30;

    for (int e = 0; e < nb_epoch; e++) {

        float sum_loss = 0.f;

        for (int b_idx = 0; b_idx < nb_batch; b_idx++) {
            int i_min = batch_size * b_idx;
            int i_max = batch_size * (b_idx + 1);
            i_max = (int) (i_max > x.size(0) ? x.size(0) : i_max);

            auto x_batch = x.slice(0, i_min, i_max).to(torch::Device(torch::kCUDA));
            auto y_batch = y.slice(0, i_min, i_max).to(torch::Device(torch::kCUDA));

            optim.zero_grad();

            auto pred = net->forward(x_batch);

            auto loss = torch::nll_loss(pred, y_batch);
            loss.backward();

            optim.step();

            sum_loss += loss.item().toDouble();
        }
        std::cout << "Epoch " << e << ", mean loss = " << (sum_loss / nb_batch) << std::endl;

        std::cout << (net->forward(x.cuda()).argmax(1) == y.cuda()).sum().item().toDouble()
            << " / " << x.size(0) << std::endl;
    }
}

int main() {
    test_load_mnist();
    //test_tensor();
}