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
    class Model
    {
        public:
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
                this->initializeLayers();

                // set model compiled 
                this->mIsCompiled = true;

                return this->mIsCompiled;
            }

            // Save model weights to desired location
            bool saveModel(std::string modelPath = "./model.csv") const;

            // Load model weights from desired location
            bool loadModel();
            
            // Train desired model
            void modelFit();

            // Trained model predict on provided input data
            void modelPredict() const;

            // Show model summary by printing it on std::cout
            void modelSummary() const;

            // Getters
            uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }
            uint8_t get_mLayersNo() const noexcept { return this->mLayersNo; }

            // Loss function unique_ptr
            std::unique_ptr<Loss::LossFunctor> mLossPtr;

        private:
            std::vector<std::unique_ptr<Layers::Layer>> mLayers; // Number of Layers is not known in advance thus, std::vector is more suitable for storing Layers
            uint32_t mLearnableCoeffs;
            uint8_t mLayersNo;
            bool mIsCompiled;

            // Initialize all layers coefficients
            void initializeLayers();
    };
}

#endif