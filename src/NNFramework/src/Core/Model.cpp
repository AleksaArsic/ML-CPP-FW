#include "Core/Model.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace NNFramework
{
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

    // Initialize all layers coefficients
    void Model::initializeLayers()
    {
        // iterate trough layers
        for(auto it = mLayers.begin(); it != mLayers.end(); it++)
        {
            uint8_t layerId = (*it)->get_mLayerId();
            uint8_t perceptronNo = (*it)->get_mPerceptronNo();
            uint8_t prevPercNo = ((0L) == layerId ? perceptronNo : mLayers[layerId - 1]->get_mPerceptronNo());
    
            // initialize layer coefficients
            std::shared_ptr<Eigen::MatrixXd> layerWeights = (*it)->get_mLayerWeights();
            std::shared_ptr<Eigen::VectorXd> layerZ = (*it)->get_mLayerZ();
            std::shared_ptr<Eigen::VectorXd> layerBias = (*it)->get_mLayerBias();

            *layerZ = Eigen::VectorXd::Random(perceptronNo);

            if((0L) == layerId)
            {
                // Input layer does not contain Weights, Biases nor Activation
                *layerWeights = Eigen::MatrixXd::Zero(perceptronNo, prevPercNo);
                *layerBias = Eigen::VectorXd::Zero(perceptronNo);

                (*it)->set_mLearnableCoeffs(0);
            }
            else
            {
                *layerWeights = Eigen::MatrixXd::Random(perceptronNo, prevPercNo);
                *layerBias = Eigen::VectorXd::Ones(perceptronNo);

                // calculate learnable coefficients
                // learnableCoeffs = noOfPerceptrons * (noOfWeights + noOfInputs) + 1 (bias)
                uint32_t noOfCoeffs = (perceptronNo * (2 * prevPercNo)) + 1;

                (*it)->set_mLearnableCoeffs(noOfCoeffs);
                mLearnableCoeffs += noOfCoeffs;
            }

    #if 0       
            std::cout << (*it)->mActivationPtr->name() << std::endl;
            std::cout << (*(*it)->mActivationPtr)(3.23) << std::endl;

            std::cout << *layerZ << std::endl;
            std::cout << std::endl;
            std::cout << *layerBias << std::endl;
            std::cout << std::endl;

            std::cout << "perceptronNo: " << static_cast<int>(perceptronNo) << ", " << "prevPercNo: " << static_cast<int>(prevPercNo) << std::endl;
            std::cout << *layerWeights << std::endl;
    #endif
        }
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

    // Train compiled model
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
                if((0L) != (*it)->get_mLayerId())
                {
                    std::cout << "\t Activation = " << (*it)->mActivationPtr->name() << std::endl;
                }
                std::cout << "**************************************" << std::endl;
            }
            std::cout << "Total learnable coefficients = " << this->mLearnableCoeffs << std::endl;
            std::cout << "Loss function: " << this->mLossPtr->name() << std::endl;
            std::cout << "**************************************" << std::endl;
        }
        else
        {
            std::cout << "Model is not compiled!" << std::endl;
        }
    }
}