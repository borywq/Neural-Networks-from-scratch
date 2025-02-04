#include <stdexcept>
#include "Layer.h"

namespace NNS {

Layer::Layer() : activation_(ActivationFunction("sigmoid")) {
}

Layer::Layer(int input_size, int output_size, const std::string &activation_function)
    : weights_(Matrix::Random(output_size, input_size)),
      biases_(Vector::Random(output_size)),
      activation_(ActivationFunction(activation_function)) {
}

Matrix Layer::Linear(const Matrix &input) const {
    assert(input.rows() == weights_.cols() && "Input batch and weights sizes mismatch.");
    return (weights_ * input).colwise() + biases_;
}

Matrix Layer::Activate(const Matrix &input) const {
    assert(input.rows() == weights_.cols() && "Input batch and weights sizes mismatch.");
    return activation_.Calculate(Linear(input));
}

Matrix Layer::BackPropagate(const Matrix &input, const Matrix &gradient) const {
    assert(input.cols() == gradient.rows() && "Input batch and gradient sizes mismatch.");
    assert(weights_.rows() == gradient.cols() && "Weights and gradient sizes mismatch.");
    assert(weights_.cols() == input.rows() && "Weights and input batch sizes mismatch.");

    Matrix temp(input.cols(), weights_.rows());
    Matrix linear = Linear(input);
    for (int i = 0; i < temp.rows(); ++i) {
        temp.row(i) = gradient.row(i) * activation_.GetGrad(linear.col(i));
    }
    return temp * weights_;
}

void Layer::CalculateGradient(const Matrix &input, const Matrix &gradient, Matrix &weight_grad,
                              Vector &biases_grad) const {
    assert(input.rows() == weights_.cols() && "Input and weights sizes mismatch.");

    Matrix temp(weights_.rows(), input.cols());
    Matrix linear = Linear(input);
    for (int i = 0; i < input.cols(); ++i) {
        temp.col(i) = (gradient.row(i) * activation_.GetGrad(linear.col(i))).transpose();
    }
    weight_grad = temp * input.transpose() / input.cols();
    biases_grad = temp.rowwise().mean();
}

void Layer::UpdateWeights(const Matrix &delta_w, const Vector &delta_b) {
    assert(weights_.size() == delta_w.size() && "Weights and delta_w sizes mismatch.");
    assert(biases_.size() == delta_b.size() && "Biases and delta_b sizes mismatch.");
    weights_ -= delta_w;
    biases_ -= delta_b;
}

Index Layer::GetWeightsCols() const {
    return weights_.cols();
}

Index Layer::GetWeightsRows() const {
    return weights_.rows();
}

}  // namespace NNS
