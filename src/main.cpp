#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/NNFramework"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    Model::Model model;

    Model::ModelConfiguration::ModelConfiguration modelConfig { Loss::LossType<Loss::MeanSquaredError>(), 
                                                                Metrics::MetricsType<Metrics::MeanSquaredError>(), 
                                                                Optimizers::OptimizersType<Optimizers::GradientDescent>() };

    modelConfig.mOptimizerPtr->learningRate = 0.1;

    model.addLayer(Layers::Dense(1)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layers::Dense(20, Activations::ActivationType<Activations::LeakyRelu>()));
    model.addLayer(Layers::Dense(1, Activations::ActivationType<Activations::Sigmoid>()));

    model.compileModel(modelConfig);
    model.modelSummary();

    // read input data and labels from input file
    std::tuple loadedData = loadData("./data/input_data.txt", 200, 1, 1);
    Eigen::MatrixXd inData = std::get<0>(loadedData);
    Eigen::MatrixXd labelsData = std::get<1>(loadedData);

    Eigen::MatrixXd inDataNormalized = normalizeData(inData); 
    Eigen::MatrixXd outDataNormalized = normalizeData(labelsData);

    model.modelFit(inDataNormalized, outDataNormalized, 50);

    Eigen::MatrixXd predictedData = model.modelPredict(inDataNormalized);
    predictedData = denormalizeData(predictedData);

    //sort data based on xi values for the purposes of graph plotting 
    std::tuple sortedPredData = sortData(inData, predictedData);
    std::tuple sortedExpData = sortData(inData, labelsData);

    std::tuple plotTuple = std::tuple_cat(sortedPredData, std::make_tuple(std::get<1>(sortedExpData)));

    // can't export graphs???
    plotData(plotTuple);
    //plotData();

    return 0;
}