#pragma once

#include <Eigen/Dense>
#include <vector>

namespace NNS {

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

using VectorSet = std::vector<Eigen::VectorXd>;
using MatrixSet = std::vector<Eigen::MatrixXd>;

using Index = Eigen::Index;

}  // namespace NNS
