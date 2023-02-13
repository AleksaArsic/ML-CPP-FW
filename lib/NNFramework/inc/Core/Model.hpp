#ifndef MODEL_CORE_HPP
#define MODEL_CORE_HPP

#include <memory>
#include <vector>
#include <tuple>
#include "../Eigen/Dense"
#include "Layers.hpp"
#include "Activations.hpp"
#include "ModelConfiguration.hpp"
#include "WeightInitializer.hpp"
#include "../Utilities/DataHandler.hpp"
#include "../Common/Common.hpp"

namespace NNFramework
{
    namespace Model
    {
        class Model final
        {
            public:

                Model() : mLearnableCoeffs(0), mLayersNo(0), mIsCompiled(false) { }

                // Add new layer to the NN Model
                // check what happens when sent by reference
                bool addLayer(Layers::Layer layer);

                // Compile model with added layers, optimizer, loss function and metrics 
                bool compileModel(ModelConfiguration::ModelConfiguration& modelConfig);

                // Save model weights to desired location
                bool saveModel(std::string modelPath = "./model.csv") const;

                // Load model weights from desired location
                bool loadModel();
                
                // Train desired model
                // Expected inputData format:
                // Eigen::MatrixXd
                // Data:
                // 1)   [x11, x12, ..., x1m]
                // 2)   [x21, x22, ..., x2m]
                // ...
                // n)   [xn1, xn2, ..., xnm]
                // 
                // Expected expectedData format:
                // Eigen::MatrixXd
                // Data:
                // 1)   [y11, y12, ..., y1m]
                // 2)   [y21, y22, ..., y2m]
                // ...
                // n)   [yn1, yn2, ..., ynm]
                //         
                void modelFit(const Eigen::MatrixXd& inData, const Eigen::MatrixXd& expData, const uint16_t epochs);

                // Trained model predict on provided input data
                // Expected inputData format:
                // Eigen::MatrixXd
                // Data:
                // 1)   [x11, x12, ..., x1m]
                // 2)   [x21, x22, ..., x2m]
                // ...
                // n)   [xn1, xn2, ..., xnm]
                // 
                // Return value data format:
                // Eigen::MatrixXd
                // Data:
                // 1)   [y11, y12, ..., y1m]
                // 2)   [y21, y22, ..., y2m]
                // ...
                // n)   [yn1, yn2, ..., ynm]
                //
                Eigen::MatrixXd modelPredict(Eigen::MatrixXd& inputData);

                // Show model summary by printing it on std::cout
                void modelSummary() const;

                // Getters
                uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }
                uint16_t get_mLayersNo() const noexcept { return this->mLayersNo; }
                bool get_mIsCompiled() const noexcept { return this->mIsCompiled; }

                // get ModelHistory
                auto get_mModelHistory() const noexcept { return this->mHistory; }

            private:
                // saves model training history
                // Loss, Validation Loss, Accuracy and Validation Accuracy
                struct ModelHistory final
                {
                    Eigen::VectorXd hLoss;
                    Eigen::VectorXd hAccuracy;
                };

                ModelHistory mHistory; // Model history container

                std::unique_ptr<ModelConfiguration::ModelConfiguration> mModelConfigPtr; // Model configuration container
                std::unique_ptr<WeightInitializer::WeightInitializer> mWeightInitializerPtr; // Layer weights initializer based on the activation function of the layer

                std::vector<std::unique_ptr<Layers::Layer>> mLayers; // Number of Layers is not known in advance thus, std::vector is more suitable for storing Layers
                uint32_t mLearnableCoeffs;
                uint16_t mLayersNo;
                bool mIsCompiled;

                // Check if model is compiled
                void checkIsModelCompiled(std::string fName) const;

                // Check if data matrix (Eigen::MatrixXd) is empty
                // throws an exception if data matrix is empty
                void isDataEmpty(const std::string fName, const Eigen::MatrixXd& data) const;

                // Check if input data and expected data have the same amount of rows
                // Check if there is a pair for each input data tensor in expected data and vice versa
                void checkInExpRowDim(const std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& expData) const;

                // Check if the input Matrix has the same amount of columns as the number of rows in layer data
                void checkRowColDim(const std::string fName, const Eigen::MatrixXd& inData, const Eigen::MatrixXd& layerData) const;

                // Initialize all layers coefficients
                void initializeLayers();

                // Forward pass
                void forwardPass(const Eigen::MatrixXd& inputData, const uint32_t rowIdx);

                // Back propagation
                void backPropagation(const Eigen::MatrixXd& expData);

                // Calculate loss
                // Return values: tuple[0] = loss, tuple[1] = metrics
                std::tuple<Eigen::MatrixXd, double> calculateLossAndMetrics(const Eigen::MatrixXd& expectedData, const uint32_t rowIdx);

        };
    }
}

#endif