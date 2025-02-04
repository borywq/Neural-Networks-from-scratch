#pragma once

#include "Usings.h"

namespace NNS {

class ActivationFunction {
    std::function<Vector(const Vector &)> function_;
    std::function<Matrix(const Vector &)> derivative_;

public:
    ActivationFunction() = default;
    explicit ActivationFunction(const std::string &);
    Matrix Calculate(const Matrix &) const;
    Matrix GetGrad(const Vector &) const;
};

}  // namespace NNS
