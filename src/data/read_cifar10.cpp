//
// Created by samuel on 02/06/19.
//

#include "read_cifar10.h"

data_set read_cifar(std::string file_name) {
    std::ifstream in_data(file_name.c_str());

    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> labels;

    while (!in_data.eof()) {
        char red[1024];
        char green[1024];
        char blue[1024];
        char label;
        in_data.read(&label, 1);
        in_data.read(red, 1024);
        in_data.read(green, 1024);
        in_data.read(blue, 1024);

        float r[1024];
        float g[1024];
        float b[1024];

        for (int i = 0; i < 1024; i++) {
            r[i] = (unsigned char) red[i] / 255.f;
            g[i] = (unsigned char) green[i] / 255.f;
            b[i] = (unsigned char) blue[i] / 255.f;
        }

        auto img = torch::stack(torch::TensorList({torch::tensor(at::ArrayRef<float>(r)).view({32, 32}),
                                                   torch::tensor(at::ArrayRef<float>(g)).view({32, 32}),
                                                   torch::tensor(at::ArrayRef<float>(b)).view({32, 32})
        }), 0);

        images.push_back(img);
        labels.push_back(torch::tensor((int) label));
    }

    torch::Tensor x = torch::stack(torch::TensorList(images), 0);
    torch::Tensor y = torch::stack(torch::TensorList(labels), 0).to(at::kLong).squeeze(1);

    return data_set(x, y);
}
