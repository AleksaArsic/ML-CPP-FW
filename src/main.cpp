#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/inc/Eigen/Dense"
#include "NNFramework/Activations"
#include "NNFramework/Loss"
#include "NNFramework/Layers"
#include "NNFramework/Model"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    Model model;

    model.addLayer(Layers::Dense(3)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layers::Dense(4, Activations::ActivationType<Activations::Relu>()));
    model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::LeakyRelu>()));

    model.compileModel(Loss::LossType<Loss::MeanAbsoluteError>());
    model.modelSummary();

    // read input data and labels from input file
    std::tuple loadedData = loadData("./data/input_data.txt", 200, 1, 1);
    Eigen::MatrixXd inData = std::get<0>(loadedData);
    Eigen::MatrixXd labelsData = std::get<1>(loadedData);

    Eigen::MatrixXd dummyInData(1, 3);
    dummyInData << 1, 2, 3;
    model.modelFit(dummyInData, 1);

    // sort data based on xi values for the purposes of graph plotting 
    //std::tuple sortedData = sortData(inData, labelsData);

    // can't export graphs???
    //plotData(sortedData);
    //plotData();

    // unfinished
    //model.saveModel();

    return 0;
}