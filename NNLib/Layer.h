#pragma once

#include <functional>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "ActivationFunction.h"

namespace NNS {

class Layer {
     Matrix weights_;
     Vector biases_;
     ActivationFunction activation_;

public:
    Layer();
    Layer(int input_size, int output_size, const std::string &activation_function);

    Matrix Linear(const Matrix &input) const;
    Matrix Activate(const Matrix &input) const;
    Matrix BackPropagate(const Matrix &input, const Matrix &gradient) const;
    void CalculateGradient(const Matrix &input, const Matrix &gradient, Matrix &weight_grad,
                           Vector &biases_grad) const;

    void UpdateWeights(const Matrix &delta_w, const Vector &delta_b);
    Index GetWeightsCols() const;
    Index GetWeightsRows() const;
};

}  // namespace NNS
