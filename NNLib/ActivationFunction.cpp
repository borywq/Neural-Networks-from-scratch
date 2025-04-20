#include "ActivationFunction.h"
#include <iostream>
#include <algorithm>

namespace NNS {

namespace functions {

static Vector Sigmoid(const Vector& x) {
    return x.unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
}

static Matrix SigmoidDer(const Vector& x) {
    Vector res = Sigmoid(x);
    Vector diff = res.unaryExpr([](double x) { return (1.0 - x) * x; });
    return diff.asDiagonal();
}

static Vector ReLU(const Vector& x) {
    return x.unaryExpr([](double x) { return std::max(0.0, x); });
}
static Matrix ReLUDer(const Vector& x) {
    Vector diff = x.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; });
    return diff.asDiagonal();
}

static Vector Softmax(const Vector& x) {
    Vector temp = exp(x.array());
    return temp / temp.sum();
}
static Matrix SoftmaxDer(const Vector& x) {
    Vector res = Softmax(x);
    Matrix diff = res.asDiagonal();
    return diff - res * res.transpose();
}

}  // namespace functions

ActivationFunction::ActivationFunction(const std::string& name) {
    if (name == "relu") {
        function_ = functions::ReLU;
        derivative_ = functions::ReLUDer;
    } else if (name == "sigmoid") {
        function_ = functions::Sigmoid;
        derivative_ = functions::SigmoidDer;
    } else if (name == "softmax") {
        function_ = functions::Softmax;
        derivative_ = functions::SoftmaxDer;
    } else {
        throw std::runtime_error("Unknown activation function_: " + name);
    }
}

Matrix ActivationFunction::Calculate(const Matrix& input) const {
    assert(function_ && "Activation function is empty.");
    Matrix output = Matrix(input.rows(), input.cols());
    for (int i = 0; i < input.cols(); ++i) {
        Vector col = input.col(i);
        output.col(i) = function_(col);
    }
    return output;
}

Matrix ActivationFunction::GetGrad(const Vector& input) const {
    assert(derivative_ && "Activation function derivative is empty.");
    auto grad = derivative_(input);
    return grad;
}

}  // namespace NNS
