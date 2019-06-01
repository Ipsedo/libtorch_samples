//
// Created by samuel on 01/06/19.
//

#include "read_mnist.h"

#include <iostream>

data_set read_mnist(std::string image_file_name, std::string label_file_name) {

    std::ifstream in_data(image_file_name.c_str());
    std::ifstream in_label(label_file_name.c_str());

    if (!in_data || !in_label) {
        std::cout << "Error during opening models file" << std::endl;
        exit(ERROR_LOADING_MNIST);
    }

    char unsued[4 * 4];
    in_label.read(unsued, 2 * 4);
    in_data.read(unsued, 4 * 4);

    char curr_label = -1;
    char curr_image[28 * 28];

    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> labels;

    while (!in_data.eof()) {
        in_label.read(&curr_label, 1);
        int y = (unsigned int) curr_label;
        labels.emplace_back(torch::tensor(at::ArrayRef<int>{y}));
        in_data.read(curr_image, 28 * 28);
        double tmp[28 * 28];
        for (int i = 0; i < 28 * 28; i++)
            tmp[i] = (unsigned int) curr_image[i] / 255.;
        images.emplace_back(torch::tensor(at::ArrayRef<double>(tmp)).view({28, 28}));
    }

    torch::Tensor x = torch::stack(torch::TensorList(images), 0);
    torch::Tensor y = torch::stack(torch::TensorList(labels), 0);

    return data_set(x, y);
}
