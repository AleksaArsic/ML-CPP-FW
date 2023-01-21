#ifndef MODEL_CORE_HPP
#define MODEL_CORE_HPP

#include <memory>
#include <vector>
#include <iostream>
#include "../Eigen/Dense"
#include "Activations.hpp"
#include "Loss.hpp"

class Layer
{
    public:
        Layer(const uint8_t perceptronNo) : mLayerId(0), mPerceptronNo(perceptronNo), mLearnableCoeffs(0)
        {
            mInstances++;
            mLayerWeights = std::make_shared<Eigen::MatrixXd>();
            mLayerZ = std::make_shared<Eigen::VectorXd>();
            mLayerBias = std::make_shared<Eigen::VectorXd>();
            mActivationPtr = nullptr; // if the activation functor type is not specified, mActivationPtr = nullptr
        }

        template<class T>
        Layer(const uint8_t perceptronNo, Activations::ActivationType<T>) : Layer(perceptronNo)
        {
            mActivationPtr = std::make_unique<T>();
        }

        Layer(Layer& l) : mLayerId(l.mLayerId), mPerceptronNo(l.mPerceptronNo), mLearnableCoeffs(l.mLearnableCoeffs)
        {
            mLayerWeights = std::move(l.mLayerWeights);
            mLayerZ = std::move(l.mLayerZ);
            mLayerBias = std::move(l.mLayerBias);
            mActivationPtr = std::move(l.mActivationPtr);
            l.mLayerId = 0;
            l.mPerceptronNo = 0;
            l.mLearnableCoeffs = 0;
        }

        // Getters
        uint8_t get_mPerceptronNo() const noexcept { return this->mPerceptronNo; }
        uint8_t get_mLayerId() const noexcept { return this->mLayerId; }
        uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }

        std::shared_ptr<Eigen::MatrixXd> get_mLayerWeights() const noexcept { return this->mLayerWeights; }
        std::shared_ptr<Eigen::VectorXd> get_mLayerZ() const noexcept { return this->mLayerZ; }
        std::shared_ptr<Eigen::VectorXd> get_mLayerBias() const noexcept { return this->mLayerBias; }

        // Setters
        void set_mLayerId(uint8_t id) { this->mLayerId = id; }
        void set_mLearnableCoeffs(uint32_t coeffsNo) { this->mLearnableCoeffs = coeffsNo; }

        // Activation function unique_ptr
        std::unique_ptr<Activations::ActivationFunctor> mActivationPtr;

    private:
        inline static uint8_t mInstances = 0;

        std::shared_ptr<Eigen::MatrixXd> mLayerWeights;
        std::shared_ptr<Eigen::VectorXd> mLayerZ;
        std::shared_ptr<Eigen::VectorXd> mLayerBias;
        
        uint8_t mLayerId;
        uint8_t mPerceptronNo;
        uint32_t mLearnableCoeffs; 
};

class Model
{
    public:
        Model() : mLearnableCoeffs(0), mLayersNo(0), mIsCompiled(false) { }

        ~Model() = default;

        // Add new layer to the NN Model
        bool addLayer(Layer layer);

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
        std::vector<std::unique_ptr<Layer>> mLayers; // Number of Layers is not known in advance thus, std::vector is more suitable for storing Layers
        uint32_t mLearnableCoeffs;
        uint8_t mLayersNo;
        bool mIsCompiled;

        // Initialize all layers coefficients
        void initializeLayers();
};

#endif