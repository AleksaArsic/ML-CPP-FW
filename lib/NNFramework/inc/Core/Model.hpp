#ifndef MODEL_CORE_HPP
#define MODEL_CORE_HPP

#include <memory>
#include <vector>
#include <iostream>
#include "../Eigen/Dense"
#include "Layers.hpp"
#include "Activations.hpp"
#include "Loss.hpp"

namespace NNFramework
{
    class Model final
    {
        public:
            
            // Loss function unique_ptr
            std::unique_ptr<Loss::LossFunctor> mLossPtr;

            Model() : mLearnableCoeffs(0), mLayersNo(0), mIsCompiled(false) { }

            ~Model() = default;

            // Add new layer to the NN Model
            bool addLayer(Layers::Layer layer);

            // Compile model with added layers, optimizer, loss function and metrics 
            template<class T>
            bool compileModel(Loss::LossType<T>)
            {
                // bind loss functor to the neural network model
                mLossPtr = std::make_unique<T>();

                // initialize all layers coefficients
                this->__initializeLayers();

                // set model compiled 
                this->mIsCompiled = true;

                return this->mIsCompiled;
            }

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
            void modelFit(Eigen::MatrixXd& inputData, Eigen::MatrixXd& expectedData, const uint32_t epochs);

            // Trained model predict on provided input data
            // Expected output:
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
            uint8_t get_mLayersNo() const noexcept { return this->mLayersNo; }

        private:
            // saves model training history
            // Loss, Validation Loss, Accuracy and Validation Accuracy
            struct ModelHistory
            {
                std::vector<double> hLoss;
                std::vector<double> hValLoss;
                std::vector<double> hAccuracy;
                std::vector<double> hValAccuracy;
            };

            ModelHistory mHistory;
            std::vector<std::unique_ptr<Layers::Layer>> mLayers; // Number of Layers is not known in advance thus, std::vector is more suitable for storing Layers
            uint32_t mLearnableCoeffs;
            uint8_t mLayersNo;
            bool mIsCompiled;

            // Initialize all layers coefficients
            void __initializeLayers();

            // Forward pass
            void __forwardPass(const Eigen::MatrixXd inputData, const uint32_t rowId);
    };
}

#endif