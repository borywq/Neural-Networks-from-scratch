#include "LossFunction.h"
#include <string>
#include <iostream>

namespace NNS {

namespace functions {

double MSE(const Vector &output, const Vector &target) {
    return (output - target).squaredNorm();
}

Vector MSEDer(const Vector &output, const Vector &target) {
    return 2.0 * (output - target);
};

double CrossEntropy(const Vector &output, const Vector &target) {
    return -(target.array() * (output.array() + 1e-12).log()).sum();
};

Vector CrossEntropyDer(const Vector &output, const Vector &target) {
    return -(target.array() / (output.array() + 1e-12));
};

}  // namespace functions

LossFunction::LossFunction(const std::string &name) {
    if (name == "MSE") {
        loss_function_ = functions::MSE;
        derivative_ = functions::MSEDer;
    } else if (name == "Cross-entropy") {
        loss_function_ = functions::CrossEntropy;
        derivative_ = functions::CrossEntropyDer;
    } else {
        assert(false && "No such activation function");
    }
}

double LossFunction::GetError(const Vector &output, const Vector &target) {
    assert(output.size() == target.size() &&
           "Loss function cannot take two vectors of different sizes.");
    return loss_function_(output, target);
}

Matrix LossFunction::GetMatrixGrad(const Matrix &output, const Matrix &target) {
    assert(output.cols() == target.cols() &&
           "Loss function cannot take two matrices of different shapes.");
    Matrix grad(output.cols(), output.rows());
    for (int i = 0; i < grad.rows(); ++i) {
        grad.row(i) = derivative_(output.col(i), target.col(i)).transpose();
    }
    return grad;
}

}  // namespace NNS
