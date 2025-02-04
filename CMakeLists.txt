cmake_minimum_required(VERSION 3.27)
project(NNS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Eigen3 REQUIRED)

add_executable(NNS main.cpp
        NNLib/Usings.h
        NNLib/Layer.cpp
        NNLib/Layer.h
        NNLib/LossFunction.cpp
        NNLib/LossFunction.h
        NNLib/Network.cpp
        NNLib/Network.h
        NNLib/ActivationFunction.cpp
        NNLib/ActivationFunction.h
        NNLib/DataLoader.cpp
        NNLib/DataLoader.h
        NNLib/Optimizer.cpp
        NNLib/Optimizer.h)


target_link_libraries(NNS PRIVATE Eigen3::Eigen)
