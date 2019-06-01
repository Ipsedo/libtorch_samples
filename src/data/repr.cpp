//
// Created by samuel on 01/06/19.
//

#include "repr.h"

torch::Tensor get_data(const data_set set) {
    return std::get<0>(set);
}

torch::Tensor get_labels(const data_set set) {
    return std::get<1>(set);
}
