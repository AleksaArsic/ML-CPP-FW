#include "Model.hpp"
#include <fstream>
#include <iostream>
#include <string>

// Add new layer to the NN Model
bool Model::addLayer(Layer layer)
{
    try
    {
        layer.set_mLayerId(this->mLayersNo++);

        this->mLayers.push_back(std::move(std::make_unique<Layer>(layer)));

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Model.addLayer() operation failed: ";
        std::cerr << e.what() << std::endl;
        return false;
    }
    
}

// Compile model with added layers, optimizer, loss function and metrics 
bool Model::compileModel()
{
    // iterate trough layers
    for(auto it = mLayers.begin(); it != mLayers.end(); it++)
    {
        uint8_t layerId = (*it)->get_mLayerId();
        uint8_t perceptronNo = (*it)->get_mPerceptronNo();
        uint8_t prevPercNo = (0 == layerId ? perceptronNo : mLayers[layerId - 1]->get_mPerceptronNo());

        // calculate learnable coefficients
        // learnableCoeffs = noOfPerceptrons * (noOfWeights + noOfInputs) + 1 (bias)
        uint32_t noOfCoeffs = (perceptronNo * (2 * prevPercNo)) + 1;

        (*it)->set_mLearnableCoeffs(noOfCoeffs);
        mLearnableCoeffs += noOfCoeffs;

        // initialize layer bias
        std::shared_ptr<Eigen::MatrixXd> layerWeights = (*it)->get_mLayerWeights();
        std::shared_ptr<Eigen::MatrixXd> layerX = (*it)->get_mLayerX();
        std::shared_ptr<Eigen::MatrixXd> layerBias = (*it)->get_mLayerBias();

        *layerWeights = Eigen::MatrixXd::Random(perceptronNo, prevPercNo);
        *layerX = Eigen::MatrixXd::Random(prevPercNo, 1);

        if(layerId == (this->mLayersNo - 1))
        {
            *layerBias = Eigen::MatrixXd::Zero(perceptronNo, 1);
        }
        else
        {
            *layerBias = Eigen::MatrixXd::Ones(perceptronNo, 1);
        }

#if 0        
        std::cout << "perceptronNo: " << static_cast<int>(perceptronNo) << ", " << "prevPercNo: " << static_cast<int>(prevPercNo) << std::endl;
        std::cout << *layerWeights << std::endl;
#endif
    }

    this->mIsCompiled = true;

    return this->mIsCompiled;
}

// Save model weights to desired location
bool Model::saveModel(std::string modelPath) const
{
    try
    {
        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Model.saveModel() failed: ";
        std::cerr << e.what() << '\n';
        return false;
    }

}

// Load model weights from desired location
bool Model::loadModel()
{
    return false;
}

// Train desired model
void Model::modelFit()
{

}

// Trained model predict on provided input data
void Model::modelPredict() const
{

}

// Show model summary by printing it on std::cout
void Model::modelSummary() const
{
    if(this->mIsCompiled)
    {
        std::cout << "**************************************" << std::endl;
        std::cout << "Model summary: " << std::endl;
        std::cout << "**************************************" << std::endl;

        for (auto it = mLayers.begin(); it != mLayers.end(); it++)
        {
            std::cout << "Layer: " << static_cast<uint32_t>((*it)->get_mLayerId()) << std::endl;
            std::cout << "\t Perceptrons = " << static_cast<uint32_t>((*it)->get_mPerceptronNo()) << std::endl;
            std::cout << "\t Coeffs = " << (*it)->get_mLearnableCoeffs() << std::endl;
            std::cout << "**************************************" << std::endl;
        }
        std::cout << "Total learnable coefficients = " << this->mLearnableCoeffs << std::endl;
        std::cout << "**************************************" << std::endl;
    }
    else
    {
        std::cout << "Model is not compiled!" << std::endl;
    }
}