#pragma once

#include <functional>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Usings.h"

namespace NNS {

class LossFunction {
    std::function<double(const Vector &, const Vector &)> loss_function_;
    std::function<Vector(const Vector &, const Vector &)> derivative_;

public:
    LossFunction() = default;
    explicit LossFunction(const std::string &);
    double GetError(const Vector &, const Vector &);
    Matrix GetMatrixGrad(const Matrix &, const Matrix &);
};

}  // namespace NNS
