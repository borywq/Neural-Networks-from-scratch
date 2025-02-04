#include <iostream>
#include <Eigen/Dense>
#include <fstream>

#include "NNLIB/Network.h"

NNS::Matrix ReadMNISTImages(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);
    assert(magic_number == 2051 && "Invalid MNIST image file magic number.");

    uint32_t num_images, num_rows, num_cols;
    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));

    num_images = __builtin_bswap32(num_images);
    num_rows = __builtin_bswap32(num_rows);
    num_cols = __builtin_bswap32(num_cols);

    assert(num_rows == 28 && num_cols == 28 && "MNIST images must be 28x28 pixels.");

    std::vector<uint8_t> image_data(num_images * num_rows * num_cols);
    file.read(reinterpret_cast<char*>(image_data.data()), image_data.size());

    NNS::Matrix images(num_rows * num_cols, num_images);
    for (size_t i = 0; i < num_images; ++i) {
        for (size_t row = 0; row < num_rows; ++row) {
            for (size_t col = 0; col < num_cols; ++col) {
                size_t pixel_index = (i * num_rows * num_cols) + (row * num_cols) + col;
                size_t matrix_index = (row * num_cols + col);
                images(matrix_index, i) =
                    static_cast<double>(image_data[pixel_index]) / 255.0;  // Normalize to [0, 1]
            }
        }
    }

    return images;
}

Eigen::VectorXd ReadMNISTLabels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл: " + filename);
    }

    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);  // Big-endian → Little-endian
    assert(magic_number == 2049 && "Неверный magic number для файла меток MNIST.");

    uint32_t num_labels;
    file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels);

    std::vector<uint8_t> label_data(num_labels);
    file.read(reinterpret_cast<char*>(label_data.data()), num_labels);

    Eigen::VectorXd labels(num_labels);
    for (size_t i = 0; i < num_labels; ++i) {
        labels(i) = static_cast<int>(label_data[i]);
    }

    return labels;
}

NNS::Matrix OneHotEncodeLabels(const Eigen::VectorXd& labels, int num_classes = 10) {
    int num_samples = labels.size();
    NNS::Matrix one_hot = NNS::Matrix::Zero(num_classes, num_samples);
    for (int i = 0; i < num_samples; ++i) {
        int label = static_cast<int>(labels(i));
        one_hot(label, i) = 1.0;
    }
    return one_hot;
}

int main() {
    try {
        NNS::Matrix images = ReadMNISTImages("../Data/train-images-idx3-ubyte");
        NNS::Vector labels = ReadMNISTLabels("../Data/train-labels-idx1-ubyte");
        NNS::Matrix lbls = OneHotEncodeLabels(labels, 10);
        NNS::Matrix small_images = images.leftCols(60000);
        NNS::Matrix small_lbls = lbls.leftCols(60000);
        NNS::Data train_data = NNS::Data(small_images, small_lbls);

        int choice;
        std::cout << "Select layers sizes:\n\t1. 784 10, 10, 10 \n\t2. 784, 30, 20 ,10 \n\t3. 784, 128, 64, 10\n";
        std::cin >> choice;
        std::vector<int> sizes;

        switch (choice) {
            case 1:
                sizes = {784, 10, 10, 10};
                break;
            case 2:
                sizes = {784, 30, 20, 10};
                break;
            case 3:
                sizes = {784, 128, 64, 10};
                break;
            default:
                sizes = {784, 30, 20, 10};
                break;
        }

        std::cout << "Select activation functions:\n\t"
                     "1. sigmoid, sigmoid, sigmoid \n\t"
                     "2. sigmoid, sigmoid, softmax \n\t"
                     "3. relu, relu, sigmoid\n";
        std::cin >> choice;
        std::vector<std::string> act_funcs;

        switch (choice) {
            case 1:
                act_funcs = {"sigmoid", "sigmoid", "sigmoid"};
                break;
            case 2:
                act_funcs = {"sigmoid", "sigmoid", "softmax"};
                break;
            case 3:
                act_funcs = {"relu", "relu", "sigmoid"};
                break;
            default:
                act_funcs = {"sigmoid", "sigmoid", "softmax"};
                break;
        }

        NNS::Network network = NNS::Network(sizes, act_funcs);

        std::cout << "Select loss functions:\n\t"
                     "1. MSE\n\t"
                     "2. Cross-entropy \n\t";
        std::cin >> choice;
        NNS::LossFunction lf;
        switch (choice) {
            case 1:
                lf = NNS::LossFunction("MSE");
                break;
            case 2:
                lf = NNS::LossFunction("Cross-entropy");
                break;
            default:
                lf = NNS::LossFunction("Cross-entropy");
                break;
        }

        NNS::DataLoader dl = NNS::DataLoader(1337);

        double lr;
        double beta_1;
        double beta_2;
        double eps;
        std::cout << "Enter learning rate:\t";
        std::cin >> lr;
        std::cout << "Enter beta_1:\t";
        std::cin >> beta_1;
        std::cout << "Enter beta_2:\t";
        std::cin >> beta_2;
        std::cout << "Enter eps:\t";
        std::cin >> eps;

        network.Train(train_data, 5, 100, dl, lf, lr, beta_1, beta_2, eps);

        NNS::Matrix test_images = ReadMNISTImages("../Data/t10k-images.idx3-ubyte");
        NNS::Vector test_labels = ReadMNISTLabels("../Data/t10k-labels.idx1-ubyte");
        NNS::Matrix test_lbls = OneHotEncodeLabels(test_labels, 10);
        NNS::Data test_data = NNS::Data(test_images, test_lbls);
        std::cout << "\nTest data:\n";
        network.Estimate(test_data);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Exception : " << e.what() << '\n';
    }
}
