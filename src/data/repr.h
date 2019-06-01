//
// Created by samuel on 01/06/19.
//

#ifndef LIBTORCH_SAMPLES_REPR_H
#define LIBTORCH_SAMPLES_REPR_H

#include <tuple>
#include <torch/torch.h>

typedef std::tuple<torch::Tensor, torch::Tensor> data_set;

torch::Tensor get_data(const data_set set);
torch::Tensor get_labels(const data_set set);

#endif //LIBTORCH_SAMPLES_REPR_H
