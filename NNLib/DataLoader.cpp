#include "DataLoader.h"
#include "Usings.h"

#include <iostream>

namespace NNS {

DataLoader::DataLoader(int seed) : gen_(seed) {
}

void DataLoader::ShuffleData(Data& data) {
    Index cols = data.input_vectors.cols();

    std::vector<Index> indices(cols);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), gen_);

    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, Index> P(cols);
    for (Index i = 0; i < cols; ++i) {
        P.indices()(i) = indices[i];
    }

    data.input_vectors = data.input_vectors * P;
    data.output_vectors = data.output_vectors * P;
}

std::vector<Data> DataLoader::GetBatches(const Data& data, int batch_size) {
    std::vector<Data> batches;
    Index total_cols = data.input_vectors.cols();
    assert(batch_size > 0 && "Batch size must be positive");

    for (Index start = 0; start < total_cols; start += batch_size) {
        Index end = std::min(start + batch_size, total_cols);
        Index actual_batch_size = end - start;

        Data batch = {data.input_vectors.middleCols(start, actual_batch_size),
                      data.output_vectors.middleCols(start, actual_batch_size)};
        batches.push_back(std::move(batch));
    }

    return batches;
}

}  // namespace NNS
