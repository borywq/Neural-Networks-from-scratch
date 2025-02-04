#include "Optimizer.h"
#include <iostream>

namespace NNS {
Optimizer::Optimizer(const std::vector<Layer>& layers, double learning_rate, double beta_1,
                     double beta_2, double eps)
    : learning_rate_(learning_rate), beta_1_(beta_1), beta_2_(beta_2), eps_(eps) {
    momentums_w_.reserve(layers.size());
    momentums_b_.reserve(layers.size());
    velocities_w_.reserve(layers.size());
    velocities_b_.reserve(layers.size());

    for (const auto& layer : layers) {
        momentums_w_.emplace_back(Matrix::Zero(layer.GetWeightsRows(), layer.GetWeightsCols()));
        momentums_b_.emplace_back(Vector::Zero(layer.GetWeightsRows()));
        velocities_w_.emplace_back(Matrix::Zero(layer.GetWeightsRows(), layer.GetWeightsCols()));
        velocities_b_.emplace_back(Vector::Zero(layer.GetWeightsRows()));
    }
}

void Optimizer::Step(Layer& layer, Matrix weights_grad, Vector bias_grad, size_t index) {
    momentums_w_[index] = beta_1_ * momentums_w_[index] + (1 - beta_1_) * weights_grad;
    momentums_b_[index] = beta_1_ * momentums_b_[index] + (1 - beta_1_) * bias_grad;
    velocities_w_[index] =
        beta_2_ * velocities_w_[index] + (1 - beta_2_) * (weights_grad.cwiseProduct(weights_grad));
    velocities_b_[index] =
        beta_2_ * velocities_b_[index] + (1 - beta_2_) * (bias_grad.cwiseProduct(bias_grad));
    Matrix divisor_w = learning_rate_ + (velocities_w_[index].array() + eps_).sqrt();
    Matrix change_w = learning_rate_ * momentums_w_[index].array() / divisor_w.array();
    Vector divisor_b = learning_rate_ + (velocities_b_[index].array() + eps_).sqrt();
    Vector change_b = learning_rate_ * momentums_b_[index].array() / divisor_b.array();
    layer.UpdateWeights(change_w, change_b);
}

void Optimizer::SetZeros() {
    for (int i = 0; i < momentums_w_.size(); ++i) {
        momentums_w_[i].setZero();
        momentums_b_[i].setZero();
        velocities_w_[i].setZero();
        velocities_b_[i].setZero();
    }
}
}  // namespace NNS
