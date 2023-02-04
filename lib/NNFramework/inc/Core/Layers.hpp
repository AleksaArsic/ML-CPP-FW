#ifndef LAYERS_CORE_HPP
#define LAYERS_CORE_HPP

#include <memory>
#include "../Eigen/Dense"
#include "Activations.hpp"

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
                // Activation function unique_ptr
                std::unique_ptr<Activations::ActivationFunctor> mActivationPtr;

                Layer() = delete;
                Layer(Layer& l) = delete;

                Layer(Layer&& l);

                // Delete copy assignment operator
                Layer& operator=(const Layer& l) = delete;

                // Delete move assignment operator
                Layer& operator=(const Layer&& l) = delete;

                // Getters
                uint8_t get_mPerceptronNo() const noexcept { return this->mPerceptronNo; }
                uint8_t get_mLayerId() const noexcept { return this->mLayerId; }
                uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }

                std::shared_ptr<Eigen::MatrixXd> get_mLayerWeights() const noexcept { return this->mLayerWeights; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerZ() const noexcept { return this->mLayerZ; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerBias() const noexcept { return this->mLayerBias; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerZActivated() const noexcept { return this->mLayerZActivated; }
                std::shared_ptr<Eigen::MatrixXd> get_mLayerWGradients() const noexcept { return this->mLayerWGradients; }

                // Setters
                void set_mLayerId(uint8_t id) { this->mLayerId = id; }
                void set_mLearnableCoeffs(uint32_t coeffsNo) { this->mLearnableCoeffs = coeffsNo; }

            protected:
                std::shared_ptr<Eigen::MatrixXd> mLayerWeights;
                std::shared_ptr<Eigen::MatrixXd> mLayerZ;
                std::shared_ptr<Eigen::MatrixXd> mLayerBias;
                
                std::shared_ptr<Eigen::MatrixXd> mLayerZActivated;
                std::shared_ptr<Eigen::MatrixXd> mLayerWGradients;

                uint8_t mLayerId;
                uint8_t mPerceptronNo;
                uint32_t mLearnableCoeffs; 

                Layer(const uint8_t perceptronNo); // Hide constructor from outside world, only classes inheriting Layer can construct Layers::Layer
        };

        // Demonstration of how greater modularity could be achieved
        // in this particular case of Dense layer we don't have much additional functionality
        // in addition to the base class Layer, but for some other layer types there could be a lot more.
        class Dense : public Layer
        {
            public:
                Dense() = delete;
                Dense(Dense& d) = delete;

                Dense(const uint8_t perceptronNo);

                template<class T>
                Dense(const uint8_t perceptronNo, Activations::ActivationType<T>) : Dense(perceptronNo)
                {
                    mActivationPtr = std::make_unique<T>();
                }

                Dense(Dense&& l) : Layer(std::move(l)) { /* Nothing to do, just call parent move constructor */ };

                // Delete copy assignment operator
                Dense& operator=(const Dense& d) = delete;

                // Delete move assignment operator
                Dense& operator=(const Dense&& d) = delete;

            private:
                inline static uint8_t mInstances = 0;
        };
    }
}
#endif