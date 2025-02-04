#pragma once

#include "Usings.h"
#include "Layer.h"
#include <vector>

namespace NNS {

class Optimizer {
    double learning_rate_;
    double beta_1_;
    double beta_2_;
    double eps_;

    MatrixSet momentums_w_;
    MatrixSet velocities_w_;
    VectorSet momentums_b_;
    VectorSet velocities_b_;

public:
    Optimizer(const std::vector<Layer>& layers, double learning_rate = 1, double beta_1 = 0.9,
              double beta_2 = 0.99, double eps = 1e-8);

    void Step(Layer& layer, Matrix weights_grad, Vector bias_grad, size_t index);
    void SetZeros();
};
}  // namespace NNS
