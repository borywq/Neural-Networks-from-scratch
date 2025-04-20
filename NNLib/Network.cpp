#include "Network.h"
#include <iostream>

namespace NNS {

Network::Network(const std::vector<int> &sizes, const std::vector<std::string> &act_funcs)
    : layers_(std::vector<Layer>(sizes.size() - 1)) {
    assert(sizes.size() == act_funcs.size() + 1 && "Parameters for layers do not correspond.");
    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i] = Layer(sizes[i], sizes[i + 1], act_funcs[i]);
    }
}

MatrixSet Network::FeedForward(const Matrix &input) {
    assert(layers_[0].GetWeightsCols() == input.rows() && "Input size mismatch.");
    MatrixSet mid_results(layers_.size() + 1);
    mid_results[0] = input;
    for (size_t i = 0; i < layers_.size(); ++i) {
        mid_results[i + 1] = layers_[i].Activate(mid_results[i]);
    }
    return mid_results;
}

void Network::FeedBackward(const MatrixSet &mid_results, const Matrix &target,
                           LossFunction &loss_function, Optimizer &sch) {
    Matrix gradient = loss_function.GetMatrixGrad(mid_results.back(), target);
    for (int i = layers_.size() - 1; i >= 0; --i) {
        Matrix weights_grad = Matrix(layers_[i].GetWeightsRows(), layers_[i].GetWeightsCols());
        Vector biases_grad = Vector(layers_[i].GetWeightsRows());
        layers_[i].CalculateGradient(mid_results[i], gradient, weights_grad, biases_grad);
        gradient = layers_[i].BackPropagate(mid_results[i], gradient);
        sch.Step(layers_[i], weights_grad, biases_grad, i);
    }
}

void Network::Train(Data &data, int epochs, int mini_batch, DataLoader &dl, LossFunction &loss_function,
                    double lr, double beta_1, double beta_2, double eps) {
    Optimizer sch = Optimizer(layers_, lr, beta_1, beta_2, eps);
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        sch.SetZeros();
        std::cout << "\nEpoch: " << epoch << '\n';
        dl.ShuffleData(data);
        std::vector<Data> batches(std::move(dl.GetBatches(data, mini_batch)));
        for (const Data &batch : batches) {
            Matrix input = batch.input_vectors;
            Matrix target = batch.output_vectors;
            MatrixSet mid_results = FeedForward(input);
            FeedBackward(mid_results, target, loss_function, sch);
        }
        Estimate(data);
    }
}

Vector Network::Predict(const Matrix &input) {
    Vector res = Vector(input.cols());
    for (int i = 0; i < input.cols(); ++i) {
        Vector cur_inp = input.col(i);
        for (const Layer &layer : layers_) {
            cur_inp = layer.Activate(cur_inp);
        }
    }
    return res;
}

void Network::Estimate(const Data &data) {
    Matrix input = data.input_vectors;
    Matrix output = FeedForward(input).back();
    int cnt = 0;
    for (int i = 0; i < output.cols(); ++i) {
        int predicted = 0;
        output.col(i).maxCoeff(&predicted);

        int actual = 0;
        data.output_vectors.col(i).maxCoeff(&actual);

        if (predicted != actual) {
            ++cnt;
        }
    }
    std::cout << "Total: " << output.cols() << " Wrong: " << cnt
              << " Accuracy: " << (1.0 - static_cast<double>(cnt) / output.cols()) << std::endl;
}

}  // namespace NNS
