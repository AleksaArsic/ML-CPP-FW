#include "Core/Model.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <limits>

namespace NNFramework
{
    namespace Model
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

        // Compile model with added layers, optimizer, loss function and metrics 
        bool Model::compileModel(ModelConfiguration::ModelConfiguration& modelConfig)
        {
            // bind model configuration to the neural network model
            mModelConfigPtr = std::make_unique<ModelConfiguration::ModelConfiguration>(std::move(modelConfig));
            
            // create mWeightInitializerPtr object
            mWeightInitializerPtr = std::make_unique<WeightInitializer::WeightInitializer>();

            // initialize all layers coefficients
            initializeLayers();

            // set model compiled 
            mIsCompiled = true;

            return this->mIsCompiled;
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
                std::cerr << __FUNCTION__ << ": ";
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
        void Model::modelFit(const Eigen::MatrixXd& inData, const Eigen::MatrixXd& expData, const uint16_t epochs)
        {
            // check if model is compiled
            checkIsModelCompiled(__FUNCTION__);

            // check if input data and expected data are empty
            isDataEmpty(__FUNCTION__, inData);
            isDataEmpty(__FUNCTION__, expData);

            // check if input data and expected data have same number of rows
            checkInExpRowDim(__FUNCTION__, inData, expData);

            // check if input data has the same number of columns as number of rows in input layer of NN
            // check if expectedData has the same number of columns as number of rows in output layer of NN
            checkRowColDim(__FUNCTION__, inData, *(mLayers[INPUT_LAYER_IDX]->get_mLayerZActivated()));
            checkRowColDim(__FUNCTION__, expData, *(mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated()));

            // Configure the rest of the model in "train-time"

            // Data handler reference used for shuffling the data
            std::unique_ptr<DataHandler::DataHandler>& mDataHandlerRef = DataHandler::DataHandler::getInstance(); 

            // construct new matrices for data shuffle between epoch
            // more memory consumption, less error prone (moral dilema?)
            Eigen::MatrixXd inputData = inData;
            Eigen::MatrixXd expectedData = expData;

            // For provided number of epochs train the model
            for (uint32_t ep = 0; ep < epochs; ++ep)
            {
                // shuffle training data for better problem generalization
                if (true == (mModelConfigPtr->mShuffleData->mShuffleOnFit))
                {
                    if (NNFRAMEWORK_ZERO == (ep % mModelConfigPtr->mShuffleData->mShuffleStep))
                    {
                        mDataHandlerRef->shuffleData(inputData, expectedData);
                    }
                }

                // loss and metrics
                Eigen::VectorXd loss = Eigen::VectorXd::Zero(expectedData.cols());
                double metrics;
                
                // for each data row in inputData
                for (uint32_t rowIdx = 0; rowIdx < inputData.rows(); ++rowIdx)
                {
                    // forward pass trough NNetwork
                    forwardPass(inputData, rowIdx);
                    
                    // calculate losses and metrics
                    auto [l, m] = calculateLossAndMetrics(expectedData, rowIdx);
                    loss += l;
                    metrics += m;

                    // Log epoch status
                    std::cout << "Epoch: " << (ep + 1) << " -> Loss: " << (loss.sum() / inputData.rows()) << " Accuracy: " << (metrics / inputData.rows()) << "\r";
                    std::cout.flush();  

                    // backpropagation trough the NNetwork
                    backPropagation(expectedData.row(rowIdx));

                    // update layer coefficients based on backpropagation gradient calculation
                    ((*mModelConfigPtr->mOptimizerPtr))(mLayers);
                }
                std::cout << std::endl;

                // save loss and metrics of each epoh
                mHistory.hLoss.resize(ep + 1);
                mHistory.hLoss[ep] = loss.sum() / inputData.rows();

                mHistory.hAccuracy.resize(ep + 1);
                mHistory.hAccuracy[ep] = metrics;
            }
        }

        // Trained model predict on provided input data
        Eigen::MatrixXd Model::modelPredict(Eigen::MatrixXd& inputData)
        {
            // check if model is compiled
            checkIsModelCompiled(__FUNCTION__);

            // check if input data is empty
            isDataEmpty(__FUNCTION__, inputData);

            // check if input data has the same number of columns as number of rows in input layer of NN
            checkRowColDim(__FUNCTION__, inputData, *(mLayers[INPUT_LAYER_IDX]->get_mLayerZActivated()));

            // start predicting
            uint32_t outputLayerRows = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated()->rows();
            Eigen::MatrixXd predictedData(inputData.rows(), outputLayerRows);
            
            // for each data row in inputData
            for (uint32_t rowIdx = 0; rowIdx < inputData.rows(); ++rowIdx)
            {
                // forward pass trough NNetwork
                forwardPass(inputData, rowIdx);

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
            checkIsModelCompiled(__FUNCTION__);

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
            std::cout << "Loss function: " << mModelConfigPtr->mLossPtr->name() << std::endl;
            std::cout << "Metrics: " << mModelConfigPtr->mMetricsPtr->name() << std::endl;
            std::cout << "Optimizer: " << mModelConfigPtr->mOptimizerPtr->name() << std::endl;
            std::cout << "**************************************" << std::endl;

        }

        // Check if model is compiled
        void Model::checkIsModelCompiled(std::string fName) const
        {
            if(false == mIsCompiled)
            {
                std::cout << fName << ": ";
                throw std::runtime_error("Model is not compiled!");
            }   
        }

        // Check if data matrix (Eigen::MatrixXd) is empty
        // throws an exception if data matrix is empty
        void Model::isDataEmpty(std::string fName, const Eigen::MatrixXd& data) const
        {
            if(NNFRAMEWORK_ZERO == data.size())
            {
                std::cout << fName << ": ";
                throw std::runtime_error("Matrix is empty!");            
            }
        }

        // Check if input data and expected data have the same amount of rows
        // Check if there is a pair for each input data tensor in expected data and vice versa
        void Model::checkInExpRowDim(std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& expData) const
        {
            if(inData.rows() != expData.rows())
            {
                std::cout << fName << ": ";
                throw std::runtime_error("Input data and expected data don't have the same amount of rows! \
                                            Cannot create pairs (xi, yi) for each data entry.");            
            }
        }

        // Check if the input Matrix has the same amount of columns as the number of rows in layer data
        void Model::checkRowColDim(std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& layerData) const
        {
            if(inData.cols() != layerData.rows())
            {
                std::cout << fName << ": ";
                throw std::runtime_error("Input data Matrix does not have the same amount of rows as the number of columns in layer data!");
            }        
        }

        // Initialize all layers coefficients
        void Model::initializeLayers()
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
                std::shared_ptr<Eigen::MatrixXd> layerWGradients = (*it)->get_mLayerWGradients();
                std::shared_ptr<Eigen::MatrixXd> layerBGradients = (*it)->get_mLayerBGradients();

                *layerZ = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);
                *layerZActivated = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);

                // gradients of Weights matrix has the same dimensions as the Weights matrix
                *layerWGradients = Eigen::MatrixXd::Zero(perceptronNo, prevPercNo);
                // gradients of Bias matrix has the same dimensions as the Bias matrix
                *layerBGradients = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);

                // Layer weights matrix construction
                *layerWeights = Eigen::MatrixXd::Zero(perceptronNo, prevPercNo);

                if(INPUT_LAYER_IDX == layerId)
                {
                    // Input layer does not contain Weights, Biases nor Activation
                    *layerBias = Eigen::MatrixXd::Zero(perceptronNo, MATRIX_COL_INIT_VAL);

                    (*it)->set_mLearnableCoeffs(NNFRAMEWORK_ZERO);
                }
                else
                {
                    // initialize layer weights based on the activation function
                    (*mWeightInitializerPtr).initializeWeights(layerWeights, (*it)->mActivationPtr->name());

                    // initialize layer biases                        
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
        void Model::forwardPass(const Eigen::MatrixXd& inputData, const uint32_t rowIdx)
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
                (*layerZActivated) = (*(mLayers[i]->mActivationPtr))(*layerZ);
            } 
        }

        // Back propagation
        void Model::backPropagation(const Eigen::MatrixXd& expData)
        {
            // calculate gradients of the output layer
            std::shared_ptr<Eigen::MatrixXd> layerZActivated = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated();
            std::shared_ptr<Eigen::MatrixXd> layerWGradients = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerWGradients();
            std::shared_ptr<Eigen::MatrixXd> layerBGradients = mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerBGradients();
            std::shared_ptr<Eigen::MatrixXd> prevLayerZActivated = mLayers[PREVIOUS_LAYER_IDX(mLayersNo - 1)]->get_mLayerZActivated();

            // calculate derivative of the loss based on the output activation
            Eigen::VectorXd lossDerivative = ((*mModelConfigPtr->mLossPtr))(expData.transpose(), *layerZActivated, true);

            // calculate derivative of the activated values of output layer
            Eigen::VectorXd layerZActivationDer = (*(mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->mActivationPtr))(*layerZActivated);

            // calculate elementwise product dL/dY * dA / dZ, which is equal to dL/dB
            lossDerivative = lossDerivative.cwiseProduct(layerZActivationDer);

            // calculate overall gradient of the output layer
            // dL/dW = dL/dY * dY/dZ * dZ/dW
            // .rowwise() assignment will boradcast prevLayerZActivated column vector 
            // to all rows of layerWGradients
            (*layerWGradients).rowwise() = (*prevLayerZActivated).reshaped().transpose();
            (*layerWGradients) = (*layerWGradients).array().colwise() * lossDerivative.array();

            // calculate gradient of the bias term in output layer
            // dL/dB = dL/dY * dY/dZ * 1
            // we stored the loss derivative in respect to the output layer Z activated derivative 
            // in variable lossDerivative
            (*layerBGradients) = lossDerivative;

            // calculate gradients of the rest of the layers
            // skip first and last layer
            for (uint32_t i = (mLayersNo - 2); i > NNFRAMEWORK_ZERO; --i)
            {
                std::shared_ptr<Eigen::MatrixXd> nextLayerWeights = mLayers[NEXT_LAYER_IDX(i)]->get_mLayerWeights();
                layerZActivated = mLayers[i]->get_mLayerZActivated();
                layerWGradients = mLayers[i]->get_mLayerWGradients();
                layerBGradients = mLayers[i]->get_mLayerBGradients();
                prevLayerZActivated = mLayers[PREVIOUS_LAYER_IDX(i)]->get_mLayerZActivated();

                // dL/dA = delta^T * nextLayerWeights
                lossDerivative = lossDerivative.transpose() * (*nextLayerWeights);
            
                // calculate layerZActivationDer
                layerZActivationDer = (*(mLayers[i]->mActivationPtr))(*layerZActivated);

                // dL/dB = dL/dA (dotprod) layerZActivationDer
                (*layerBGradients) = lossDerivative.cwiseProduct(layerZActivationDer);

                // dL/dW = dL/dA (dotprod) layerZActivationDer (dotprod) prevLayerZActivated
                (*layerWGradients).colwise() = (*layerBGradients).reshaped();
                (*layerWGradients) = (*layerWGradients).array().rowwise() * (*prevLayerZActivated).reshaped().array().transpose();

                // loss derivative for the next layer in backpropagation algorithm
                lossDerivative = (*layerBGradients);
            }
        }

        // Return values: tuple[0] = loss, tuple[1] = metrics
        std::tuple<Eigen::VectorXd, double> Model::calculateLossAndMetrics(const Eigen::MatrixXd& expectedData, const uint32_t rowIdx)
        {
            // CHECK MATRICES AND VECTORS AS INPUT VALUES 
            Eigen::MatrixXd outputLayerZActivated = *(mLayers[OUTPUT_LAYER_IDX(mLayersNo)]->get_mLayerZActivated());
            Eigen::VectorXd modelOutput(Eigen::Map<Eigen::VectorXd>(outputLayerZActivated.data(), outputLayerZActivated.cols() * outputLayerZActivated.rows()));
            
            Eigen::MatrixXd expectedOutput = expectedData.row(rowIdx);
            Eigen::VectorXd loss = ((*mModelConfigPtr->mLossPtr))(expectedOutput, outputLayerZActivated);
            double metrics = ((*mModelConfigPtr->mMetricsPtr))(modelOutput, expectedOutput);       

            return std::make_tuple(loss, metrics);    
        }
    }
}