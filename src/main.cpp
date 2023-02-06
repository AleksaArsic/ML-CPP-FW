#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/NNFramework"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    Model::Model model;

    Model::ModelConfiguration::ModelConfiguration modelConfig { Loss::LossType<Loss::BinaryCrossEntropy>(), 
                                                                Metrics::MetricsType<Metrics::MeanSquaredError>(), 
                                                                Optimizers::OptimizersType<Optimizers::StohasticGradientDescent>() };

    model.addLayer(Layers::Dense(3)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layers::Dense(4, Activations::ActivationType<Activations::Sigmoid>()));
    model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::Sigmoid>()));

    model.compileModel(modelConfig);
    model.modelSummary();

    // read input data and labels from input file
    std::tuple loadedData = loadData("./data/input_data.txt", 200, 1, 1);
    Eigen::MatrixXd inData = std::get<0>(loadedData);
    Eigen::MatrixXd labelsData = std::get<1>(loadedData);

    Eigen::MatrixXd dummyInData(6, 3);
    Eigen::MatrixXd dummyOutData(6, 3);

    dummyInData << 0.1, 0.2, 0.3, 
                   0.4, 0.5, 0.6,
                   0.7, 0.8, 0.9,
                   0.10, 0.11, 0.12,
                   0.13, 0.14, 0.15,
                   0.16, 0.17, 0.18;
    dummyOutData << 0.2, 0.4, 0.6,
                    0.8, 0.10, 0.12,
                    0.14, 0.16, 0.18,
                    0.20, 0.22, 0.24,
                    0.26, 0.28, 0.30,
                    0.32, 0.34, 0.36;
    model.modelFit(dummyInData, dummyOutData, 10);

    std::cout << "Predict: " << std::endl << model.modelPredict(dummyInData) << std::endl << std::endl;

    Activations::LeakyRelu sig;
    Eigen::VectorXd vec(3);
    Eigen::VectorXd vec2(4);
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(3, 4);
    vec << 1, 2, 3;
    vec2 << 2, 4, 6, 8;

#if 0
    std::cout << "vec: " << vec << std::endl;
    std::cout << "vec2: " << vec2.transpose() << std::endl;
    mat.rowwise() += vec2.transpose();
    std::cout << "mat: " << std::endl << mat.array().colwise() * vec.array() << std::endl;
#endif

    //std::cout << vec * vec2 << std::endl;

    //std::cout << "Activations:" << std::endl;
    //std::cout << sig(vec) << std::endl << std::endl << sig(vec, true) << std::endl;

    //std::cout << model.modelPredict(dummyInData) << std::endl;
    // sort data based on xi values for the purposes of graph plotting 
    //std::tuple sortedData = sortData(inData, labelsData);

    // can't export graphs???
    //plotData(sortedData);
    //plotData();

    // unfinished
    //model.saveModel();

    return 0;
}