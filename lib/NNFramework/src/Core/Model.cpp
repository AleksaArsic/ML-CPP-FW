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
            layer.set_mLayerId(mLayersNo++);

            mLayers.push_back(std::move(std::make_unique<Layers::Layer>(std::move(layer))));

            return true;
        }
        catch(const std::exception& e)
        {
            std::cerr << __FUNCTION__ << ": ";
            std::cerr << e.what() << std::endl;
            return false;
        }   
    }

    // Save model weights to desired location
    bool Model::saveModel(std::string modelPath) const
    {
        try
        {
            /* Not supported in this version of NNFramework */
            return false;
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
        /* Not supported in this version of NNFramework */
        return false;
    }

    // Train compiled model
    void Model::modelFit(Eigen::MatrixXd& inputData, Eigen::MatrixXd& expectedData, const uint32_t epochs)
    {
        // check if model is compiled
        __checkIsModelCompiled(__FUNCTION__);

        // check if input data and expected data are empty
        __isDataEmpty(__FUNCTION__, inputData);
        __isDataEmpty(__FUNCTION__, expectedData);

        // check if input data and expected data have same number of rows
        __checkInExpRowDim(__FUNCTION__, inputData, expectedData);

        // check if input data has the same number of columns as number of rows in input layer of NN
        // check if expectedData has the same number of columns as number of rows in output layer of NN
        __checkRowColDim(__FUNCTION__, inputData, *(mLayers[INPUT_LAYER_IDX]->get_mLayerZActivated()));
        __checkRowColDim(__FUNCTION__, expectedData, *(mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated()));

        // Split data to training data - validation data

        // For provided number of epochs train the model
        for (uint32_t ep = 0; ep < epochs; ++ep)
        {
            // loss and metrics
            Eigen::VectorXd loss(expectedData.cols());
            double metrics;

            // for each data row in inputData
            for (uint32_t rowIdx = 0; rowIdx < 1 /*inputData.rows()*/; rowIdx++)
            {
                // forward pass trough NNetwork
                __forwardPass(inputData, rowIdx);
                
                // calculate losses and metrics
                auto [l, m] = __calculateLossAndMetrics(expectedData, rowIdx);
                loss += l;
                metrics = m;
                //std::cout << "Epoch: " << ep << " -> Loss: " << loss << " Accuracy: " << metrics << "\r";
                //std::cout.flush();  


                // backpropagation trough the NNetwork

                // update layer coefficients

               
            }
            std::cout << loss << std::endl << std::endl;
            std::cout << "Loss: " << loss.sum() / inputData.rows() <<  std::endl;

            // save loss and metrics of each epoh
            //mHistory.hLoss.resize(ep + 1);
            //mHistory.hLoss[ep] = loss;

            mHistory.hAccuracy.resize(ep + 1);
            mHistory.hAccuracy[ep] = metrics;

            std::cout << mHistory.hLoss << std::endl;
            std::cout << mHistory.hAccuracy << std::endl;

            // if the result of current epoch is better than overall best training result
            // save relevant model coefficients
        }

    }

    // Trained model predict on provided input data
    Eigen::MatrixXd Model::modelPredict(Eigen::MatrixXd& inputData)
    {
        // check if model is compiled
        __checkIsModelCompiled(__FUNCTION__);

        // check if input data is empty
        __isDataEmpty(__FUNCTION__, inputData);

        // check if input data has the same number of columns as number of rows in input layer of NN
        __checkRowColDim(__FUNCTION__, inputData, *(mLayers[INPUT_LAYER_IDX]->get_mLayerZActivated()));

        // start predicting
        uint32_t outputLayerRows = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated()->rows();
        Eigen::MatrixXd predictedData(inputData.rows(), outputLayerRows);
         
        // for each data row in inputData
        for (uint32_t rowIdx = 0; rowIdx < inputData.rows(); rowIdx++)
        {
            // forward pass trough NNetwork
            __forwardPass(inputData, rowIdx);

            // save outputs
            std::shared_ptr<Eigen::MatrixXd> outputLayerZActivated = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated();
            Eigen::VectorXd predictions = (*outputLayerZActivated);
            predictedData.row(rowIdx) = predictions;
        }

        // return output of the Neural Network
        return predictedData;
    }

    // Show model summary by printing it on std::cout
    void Model::modelSummary() const
    {
        // check if model is compiled
        __checkIsModelCompiled(__FUNCTION__);

        std::cout << "**************************************" << std::endl;
        std::cout << "Model summary: " << std::endl;
        std::cout << "**************************************" << std::endl;

        for (auto it = mLayers.begin(); it != mLayers.end(); ++it)
        {
            std::cout << "Layer: " << static_cast<uint32_t>((*it)->get_mLayerId()) << std::endl;
            std::cout << "\t Perceptrons = " << static_cast<uint32_t>((*it)->get_mPerceptronNo()) << std::endl;
            std::cout << "\t Coeffs = " << (*it)->get_mLearnableCoeffs() << std::endl;
            if(INPUT_LAYER_IDX != (*it)->get_mLayerId())
            {
                std::cout << "\t Activation = " << (*it)->mActivationPtr->name() << std::endl;
            }
            std::cout << "**************************************" << std::endl;
        }
        std::cout << "Total learnable coefficients = " << mLearnableCoeffs << std::endl;
        std::cout << "Loss function: " << mLossPtr->name() << std::endl;
        std::cout << "**************************************" << std::endl;

    }

    // Check if model is compiled
    void Model::__checkIsModelCompiled(std::string fName) const
    {
        if(false == mIsCompiled)
        {
            std::cout << fName << ": ";
            throw std::runtime_error("Model is not compiled!");
        }   
    }

    // Check if data matrix (Eigen::MatrixXd) is empty
    // throws an exception if data matrix is empty
    void Model::__isDataEmpty(std::string fName, const Eigen::MatrixXd& data) const
    {
        if(NNFRAMEWORK_ZERO == data.size())
        {
            std::cout << fName << ": ";
            throw std::runtime_error("Matrix is empty!");            
        }
    }

    // Check if input data and expected data have the same amount of rows
    // Check if there is a pair for each input data tensor in expected data and vice versa
    void Model::__checkInExpRowDim(std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& expData) const
    {
        if(inData.rows() != expData.rows())
        {
            std::cout << fName << ": ";
            throw std::runtime_error("Input data and expected data don't have the same amount of rows! \
                                        Cannot create pairs (xi, yi) for each data entry.");            
        }
    }

    // Check if the input Matrix has the same amount of columns as the number of rows in layer data
    void Model::__checkRowColDim(std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& layerData) const
    {
        if(inData.cols() != layerData.rows())
        {
            std::cout << fName << ": ";
            throw std::runtime_error("Input data Matrix does not have the same amount of rows as the number of columns in layer data!");
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
            uint8_t prevPercNo = (INPUT_LAYER_IDX == layerId ? perceptronNo : mLayers[PREVIOUS_LAYER_IDX(layerId)]->get_mPerceptronNo());
    
            // initialize layer coefficients
            std::shared_ptr<Eigen::MatrixXd> layerWeights = (*it)->get_mLayerWeights();
            std::shared_ptr<Eigen::MatrixXd> layerZ = (*it)->get_mLayerZ();
            std::shared_ptr<Eigen::MatrixXd> layerBias = (*it)->get_mLayerBias();
            std::shared_ptr<Eigen::MatrixXd> layerZActivated = (*it)->get_mLayerZActivated();

            *layerZ = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);
            *layerZActivated = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);

            if(INPUT_LAYER_IDX == layerId)
            {
                // Input layer does not contain Weights, Biases nor Activation
                *layerWeights = Eigen::MatrixXd::Zero(perceptronNo, prevPercNo);
                *layerBias = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);

                (*it)->set_mLearnableCoeffs(NNFRAMEWORK_ZERO);
            }
            else
            {
                *layerWeights = Eigen::MatrixXd::Random(perceptronNo, prevPercNo);
                *layerBias = Eigen::MatrixXd::Ones(perceptronNo, MATRIX_COL_INIT_VAL);

                // calculate learnable coefficients
                // learnableCoeffs = noOfPerceptrons * (noOfWeights + noOfInputs) + 1 (bias)
                uint32_t noOfCoeffs = (perceptronNo * (2 * prevPercNo)) + 1;

                (*it)->set_mLearnableCoeffs(noOfCoeffs);
                mLearnableCoeffs += noOfCoeffs;
            }
        }
    }

    // Forward pass
    void Model::__forwardPass(const Eigen::MatrixXd& inputData, const uint32_t rowIdx)
    {
        // set input layer data
        std::shared_ptr<Eigen::MatrixXd> inputLayerZ = mLayers[INPUT_LAYER_IDX]->get_mLayerZ();
        std::shared_ptr<Eigen::MatrixXd> inputLayerZActivated = mLayers[INPUT_LAYER_IDX]->get_mLayerZActivated();

        *inputLayerZ = inputData.row(rowIdx);
        (*inputLayerZ).transposeInPlace();
        
        // passtrough input values as activated
        // x = f(x)
        (*inputLayerZActivated) = (*inputLayerZ);     

        // iterate trough layers 
        // skip first layer, as first (input) layer does not have weights nor activations
        for (uint32_t i = 1; i < mLayersNo; ++i)
        {
            // get previous layer data
            std::shared_ptr<Eigen::MatrixXd> prevLayerZActivated = mLayers[PREVIOUS_LAYER_IDX(i)]->get_mLayerZActivated();

            // get current layer data
            std::shared_ptr<Eigen::MatrixXd> layerWeights = mLayers[i]->get_mLayerWeights();
            std::shared_ptr<Eigen::MatrixXd> layerZ = mLayers[i]->get_mLayerZ();
            std::shared_ptr<Eigen::MatrixXd> layerBias = mLayers[i]->get_mLayerBias();
            std::shared_ptr<Eigen::MatrixXd> layerZActivated = mLayers[i]->get_mLayerZActivated();

            // z = Wx + b
            (*layerZ) = ((*layerWeights) * (*prevLayerZActivated)) + (*layerBias);

            // apply activation functor to the layer Z activated values
            std::reference_wrapper activationFunRef = *(mLayers[i]->mActivationPtr);
            (*layerZActivated) = (*layerZ).unaryExpr(activationFunRef);
        } 
    }

    // Back propagation
    void __backPropagation(const uint32_t rowIdx)
    {
        // calculate gradients of the output layer

        // calculate gradients of the rest of the layers

    }

    // Return values: tuple[0] = loss, tuple[1] = metrics
    std::tuple<Eigen::VectorXd, double> Model::__calculateLossAndMetrics(const Eigen::MatrixXd& expectedData, const uint32_t rowIdx)
    {
        Eigen::MatrixXd outputLayerZActivated = *(mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated());
        Eigen::VectorXd modelOutput(Eigen::Map<Eigen::VectorXd>(outputLayerZActivated.data(), outputLayerZActivated.cols() * outputLayerZActivated.rows()));
        
        Eigen::VectorXd expectedOutput = expectedData.row(rowIdx);
        Eigen::VectorXd loss = (*mLossPtr)(expectedOutput, modelOutput);
        double metrics = (*mMetricsPtr)(modelOutput, expectedOutput);       

        return std::make_tuple(loss, metrics);    
    }
}