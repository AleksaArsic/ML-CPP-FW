#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/inc/Eigen/Dense"
#include "NNFramework/Activations"
#include "NNFramework/Loss"
#include "NNFramework/Model"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    Model model;

    model.addLayer(Layer(3)); // or -> model.addLayer(Layer(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layer(4, Activations::ActivationType<Activations::Relu>()));
    model.addLayer(Layer(3, Activations::ActivationType<Activations::LeakyRelu>()));

    model.compileModel(Loss::LossType<Loss::MeanAbsoluteError>());
    model.modelSummary();

    Eigen::VectorXd predicted = Eigen::VectorXd(4);
    Eigen::VectorXd expected = Eigen::VectorXd(4);

    predicted << 0.8,
                0.7,
                0.3,
                0.9;

    expected << 1,
                0,
                1,
                1;

    Loss::BinaryCrossEntropy bce;
    std::cout << bce(expected, predicted) << std::endl;
    // read input data and labels from input file
    std::tuple loadedData = loadData("./data/input_data.txt");
    Eigen::VectorXd inData = std::get<0>(loadedData);
    Eigen::VectorXd labelsData = std::get<1>(loadedData);

    // sort data based on xi values for the purposes of graph plotting 
    std::tuple sortedData = sortData(inData, labelsData);

    plotData(sortedData);

    // unfinished
    //model.saveModel();

    return 0;
}