#pragma once

#include <vector>
#include <Eigen/Dense>
#include "LossFunction.h"
#include "Layer.h"
#include "DataLoader.h"
#include "Optimizer.h"

namespace NNS {

class Network {
    std::vector<Layer> layers_;

public:
    Network(const std::vector<int> &sizes, const std::vector<std::string> &activation_function);

    MatrixSet FeedForward(const Matrix &);

    Vector Predict(const Matrix &);

    void FeedBackward(const MatrixSet &mid_results, const Matrix &target,
                      LossFunction &loss_function, Optimizer &sch);

    void Train(Data &data, int epochs, int mini_batch, DataLoader &dl, LossFunction &loss_function,
               double lr, double beta_1, double beta_2, double eps);

    void Estimate(const Data &data);
};

}  // namespace NNS
