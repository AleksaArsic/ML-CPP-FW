#include <iostream>
#include <vector>
#include <cmath>
#include "NNFramework/NNFramework"
#include "Utilities.hpp"

using namespace NNFramework;

int main()
{
    // Define Model object
    Model::Model model;

    // Define Model configuration
    Model::ModelConfiguration::ModelConfiguration modelConfig { Loss::LossType<Loss::MeanSquaredError>(), 
                                                                Metrics::MetricsType<Metrics::MeanSquaredError>(), 
                                                                Optimizers::OptimizersType<Optimizers::GradientDescent>(),
                                                                Model::ModelConfiguration::ShuffleData { true, 5 } };

    // change/set parameters of model configuration
    modelConfig.mOptimizerPtr->learningRate = 0.1;
    modelConfig.mShuffleData->mShuffleStep = 10;

    // Add layers to NN model
    model.addLayer(Layers::Dense(1)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layers::Dense(20, Activations::ActivationType<Activations::LeakyRelu>()));
    model.addLayer(Layers::Dense(1, Activations::ActivationType<Activations::Sigmoid>()));

    // compile model and bind modelConfig to the model configuration
    model.compileModel(modelConfig);

    // Log model summary
    model.modelSummary();

    // read input data and labels from input file
    std::tuple loadedData = loadData("./data/input_data.txt", 200, 1, 1);
    Eigen::MatrixXd inData = std::get<0>(loadedData);
    Eigen::MatrixXd labelsData = std::get<1>(loadedData);

    // retireve instance of DataHandler class
    std::unique_ptr<NNFramework::DataHandler::DataHandler>& dHandleRef = NNFramework::DataHandler::DataHandler::getInstance();

    // Normalize input and expected data
    Eigen::MatrixXd inDataNormalized = dHandleRef->normalizeData(inData); 
    Eigen::MatrixXd outDataNormalized = dHandleRef->normalizeData(labelsData);

    // train model on normalized data for 50 epochs
    model.modelFit(inDataNormalized, outDataNormalized, 50);

    // retrieve Model.fit() history
    auto modelHistory = model.get_mModelHistory();
    //std::cout << "Model history: " << std::endl << "Loss: " << modelHistory.hLoss << std::endl << "Accuracy: " << modelHistory.hAccuracy << std::endl;

    // Predict on trained model
    Eigen::MatrixXd predictedData = model.modelPredict(inDataNormalized);

    // denormalize predicted data
    predictedData = dHandleRef->denormalizeData(predictedData, labelsData.minCoeff(), labelsData.maxCoeff());

    //sort data based on xi values for the purposes of graph plotting 
    std::tuple sortedPredData = sortData(inData, predictedData);
    std::tuple sortedExpData = sortData(inData, labelsData);

    std::tuple historyData = std::make_tuple(modelHistory.hLoss, modelHistory.hAccuracy);

    // prepare data for plotting 
    std::tuple plotTuple = std::tuple_cat(sortedPredData, std::make_tuple(std::get<1>(sortedExpData)));

    // plot results
    plotData(plotTuple);
    plotModelHistory(50, historyData);

    return 0;
}