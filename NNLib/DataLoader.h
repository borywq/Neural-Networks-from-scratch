#pragma once

#include <random>
#include "Usings.h"

namespace NNS {

struct Data {
    Matrix input_vectors;
    Matrix output_vectors;

    Data(const Matrix& in, const Matrix& out) {
        input_vectors = in;
        output_vectors = out;
    }
};

class DataLoader {
    std::mt19937 gen_;

public:
    DataLoader(int seed);
    void ShuffleData(Data& data);
    std::vector<Data> GetBatches(const Data& data, int batch_size);
};

}  // namespace NNS
