#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/NNFramework"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    Model model;

    model.addLayer(Layers::Dense(3)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layers::Dense(4, Activations::ActivationType<Activations::Relu>()));
    model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::Sigmoid>()));

    model.compileModel(Loss::LossType<Loss::BinaryCrossEntropy>(), Metrics::MetricsType<Metrics::MeanSquaredError>());
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
    model.modelFit(dummyInData, dummyOutData, 1);

    Activations::Sigmoid sig;
    std::cout << sig(1.0) << " " << sig(1.0, true) << std::endl;
    std::cout << sig(-1.0) << " " << sig(-1.0, true) << std::endl;

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