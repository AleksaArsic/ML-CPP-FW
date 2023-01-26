#include "Core/Model.hpp"
#include <fstream>
#include <iostream>
#include <string>

namespace NNFramework
{
    // Add new layer to the NN Model
    bool Model::addLayer(Layers::Layer layer)
    {
        try
        {
            layer.set_mLayerId(this->mLayersNo++);

            this->mLayers.push_back(std::move(std::make_unique<Layers::Layer>(std::move(layer))));

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
    void Model::__initializeLayers()
    {
        // iterate trough layers
        for(auto it = mLayers.begin(); it != mLayers.end(); ++it)
        {
            uint8_t layerId = (*it)->get_mLayerId();
            uint8_t perceptronNo = (*it)->get_mPerceptronNo();
            uint8_t prevPercNo = ((0L) == layerId ? perceptronNo : mLayers[layerId - 1]->get_mPerceptronNo());
    
            // initialize layer coefficients
            std::shared_ptr<Eigen::MatrixXd> layerWeights = (*it)->get_mLayerWeights();
            std::shared_ptr<Eigen::MatrixXd> layerZ = (*it)->get_mLayerZ();
            std::shared_ptr<Eigen::MatrixXd> layerBias = (*it)->get_mLayerBias();

            *layerZ = Eigen::MatrixXd::Zero(perceptronNo, 1);

            if((0L) == layerId)
            {
                // Input layer does not contain Weights, Biases nor Activation
                *layerWeights = Eigen::MatrixXd::Zero(perceptronNo, prevPercNo);
                *layerBias = Eigen::MatrixXd::Zero(perceptronNo, 1);

                (*it)->set_mLearnableCoeffs(0);
            }
            else
            {
                *layerWeights = Eigen::MatrixXd::Random(perceptronNo, prevPercNo);
                *layerBias = Eigen::MatrixXd::Ones(perceptronNo, 1);

                // calculate learnable coefficients
                // learnableCoeffs = noOfPerceptrons * (noOfWeights + noOfInputs) + 1 (bias)
                uint32_t noOfCoeffs = (perceptronNo * (2 * prevPercNo)) + 1;

                (*it)->set_mLearnableCoeffs(noOfCoeffs);
                mLearnableCoeffs += noOfCoeffs;
            }
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

    // Forward pass
    void Model::__forwardPass(const Eigen::MatrixXd inputData, const uint32_t rowId)
    {
        // set input layer data
        std::shared_ptr<Eigen::MatrixXd> inputLayerZ = mLayers[0]->get_mLayerZ();

        *inputLayerZ = inputData.row(rowId);
        (*inputLayerZ).transposeInPlace();
        
        // iterate trough layers 
        // skip first layer, as first (input) layer does not have weights nor activations
        for (uint32_t i = 1; i < mLayersNo; ++i)
        {
            // get previous layer data
            std::shared_ptr<Eigen::MatrixXd> prevLayerZ = mLayers[i - 1]->get_mLayerZ();

            // get current layer data
            std::shared_ptr<Eigen::MatrixXd> layerWeights = mLayers[i]->get_mLayerWeights();
            std::shared_ptr<Eigen::MatrixXd> layerZ = mLayers[i]->get_mLayerZ();
            std::shared_ptr<Eigen::MatrixXd> layerBias = mLayers[i]->get_mLayerBias();

            // z = Wx + b
            (*layerZ) = ((*layerWeights) * (*prevLayerZ)) + (*layerBias);

            // apply activation functor to the layer Z values
            std::reference_wrapper activationFunRef = *(mLayers[i]->mActivationPtr);
            (*layerZ) = (*layerZ).unaryExpr(activationFunRef);
        }
    }


    // Train compiled model
    void Model::modelFit(Eigen::MatrixXd& inputData, Eigen::MatrixXd& expectedData, const uint32_t epochs)
    {
        // check if model is compiled
        if(mIsCompiled == false)
        {
            std::cout << "Model.modelFit(): Model is not compiled!" << std::endl;
            return;
        }

        // check if input data and expected data are empty
        // check if input data and expected data have same number of rows

        // For number of provided epochs train the model
        for (uint32_t i = 0; i < epochs; i++)
        {
            // for each data row in inputData
            for (uint32_t rowIdx = 0; rowIdx < 1/*inputData.rows()*/; rowIdx++)
            {
                // forward pass trough NNetwork
                __forwardPass(inputData, rowIdx);
                
                // backpropagation trough NNetwork

                // update layer coefficients


            }

            // calculate losses
            Eigen::MatrixXd outputLayerZ = *(mLayers[mLayersNo - 1]->get_mLayerZ());
            Eigen::VectorXd modelOutput(Eigen::Map<Eigen::VectorXd>(outputLayerZ.data(), outputLayerZ.cols() * outputLayerZ.rows()));
            
            Eigen::VectorXd expectedOutput(Eigen::Map<Eigen::VectorXd>(expectedData.data(), expectedData.cols() * expectedData.rows()));
           
            double loss = (*mLossPtr)(modelOutput, expectedOutput);
            
            std::cout << modelOutput << std::endl;
            std::cout << "Loss: " << loss << std::endl;

            // calculate metrics

            // if the result of current epoch is better than overall best training result
            // save relevant model coefficients
        }

    }

    // Trained model predict on provided input data
    Eigen::MatrixXd Model::modelPredict(Eigen::MatrixXd& inputData)
    {
        // check if model is compiled
        if(mIsCompiled == false)
        {
            std::cout << "Model.modelPredict(): Model is not compiled!" << std::endl;
            return Eigen::MatrixXd();
        }      

        // check if input data is empty

        // start predicting
        uint32_t outputLayerRows = mLayers[mLayersNo - 1]->get_mLayerZ()->rows();
        Eigen::MatrixXd predictedData(inputData.rows(), outputLayerRows);
        
        // for each data row in inputData
        for (uint32_t rowIdx = 0; rowIdx < inputData.rows(); rowIdx++)
        {
            // forward pass trough NNetwork
            __forwardPass(inputData, rowIdx);

            // save outputs
            std::shared_ptr<Eigen::MatrixXd> outputLayerZ = mLayers[mLayersNo - 1]->get_mLayerZ();
            Eigen::VectorXd predictions = (*outputLayerZ);
            predictedData.row(rowIdx) = predictions;
        }

        // return output of the Neural Network
        return predictedData;
    }

    // Show model summary by printing it on std::cout
    void Model::modelSummary() const
    {
        if(this->mIsCompiled)
        {
            std::cout << "**************************************" << std::endl;
            std::cout << "Model summary: " << std::endl;
            std::cout << "**************************************" << std::endl;

            for (auto it = mLayers.begin(); it != mLayers.end(); ++it)
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
            std::cout << "Model.modelSummary(): Model is not compiled!" << std::endl;
        }
    }
}