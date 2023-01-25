#ifndef LAYERS_CORE_HPP
#define LAYERS_CORE_HPP

#include <memory>
#include "../Eigen/Dense"
#include "Activations.hpp"

#include <iostream>

namespace NNFramework
{
    namespace Layers
    {
        // The idea is to have base class that will consist of all common things for different layer types:
        // Dense, Convolutive, Flatten, ...
        // With this we achieve greater modularity of the code as we will have a possibility to simply
        // inherit base functionality of the one layer.
        // This also gives a possibility to the NN Model to have a pointer to the base class 
        // where layers of different types will be instantiated.
        class Layer
        {
            public:
                Layer() = delete;
                Layer(Layer& l) = delete;

                Layer(const uint8_t perceptronNo);
                Layer(Layer&& l);

                // Getters
                uint8_t get_mPerceptronNo() const noexcept { return this->mPerceptronNo; }
                uint8_t get_mLayerId() const noexcept { return this->mLayerId; }
                uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }

                std::shared_ptr<Eigen::MatrixXd> get_mLayerWeights() const noexcept { return this->mLayerWeights; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerZ() const noexcept { return this->mLayerZ; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerBias() const noexcept { return this->mLayerBias; }

                // Setters
                void set_mLayerId(uint8_t id) { this->mLayerId = id; }
                void set_mLearnableCoeffs(uint32_t coeffsNo) { this->mLearnableCoeffs = coeffsNo; }

                // Activation function unique_ptr
                std::unique_ptr<Activations::ActivationFunctor> mActivationPtr;

            protected:
                std::shared_ptr<Eigen::MatrixXd> mLayerWeights;
                std::shared_ptr<Eigen::MatrixXd> mLayerZ;
                std::shared_ptr<Eigen::MatrixXd> mLayerBias;
                
                uint8_t mLayerId;
                uint8_t mPerceptronNo;
                uint32_t mLearnableCoeffs; 
        };

        // Demonstration of how greater modularity could be achieved
        // in this particular case of Dense layer we don't have much additional functionality
        // in addition to the base class Layer, but for some other layer types there could be a lot more.
        class Dense : public Layer
        {
            public:
                Dense(const uint8_t perceptronNo);

                template<class T>
                Dense(const uint8_t perceptronNo, Activations::ActivationType<T>) : Dense(perceptronNo)
                {
                    mActivationPtr = std::make_unique<T>();
                }

                // don't allow copy
                // Dense(Dense& l) { /* Nothing to do, just call parent copy constructor */ };
                Dense(Dense&& l) : Layer(std::move(l)) { /* Nothing to do, just call parent move constructor */ };

            private:
                inline static uint8_t mInstances = 0;
        };
    }
}
#endif