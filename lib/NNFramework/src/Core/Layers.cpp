#include "Core/Layers.hpp"

namespace NNFramework
{
    namespace Layers
    {
        Layer::Layer(const uint8_t perceptronNo) : mLayerId(0), mPerceptronNo(perceptronNo), mLearnableCoeffs(0)
        {
            mLayerWeights = std::make_shared<Eigen::MatrixXd>();
            mLayerZ = std::make_shared<Eigen::MatrixXd>();
            mLayerBias = std::make_shared<Eigen::MatrixXd>();
            mLayerZActivated = std::make_shared<Eigen::MatrixXd>();
            mLayerWGradients = std::make_shared<Eigen::MatrixXd>();
            mLayerBGradients = std::make_shared<Eigen::MatrixXd>();
            // This way we are sure we are having "Passtrough" activation for the input layer 
            // and as this constructor is protected only classes that are inheriting Layers::Layer
            // can construct base functionality of the Layer class
            mActivationPtr = std::make_unique<Activations::InputActivation>(); 
        }

        Layer::Layer(Layer&& l) : mLayerId(l.mLayerId), mPerceptronNo(l.mPerceptronNo), mLearnableCoeffs(l.mLearnableCoeffs)
        {
            mLayerWeights = std::move(l.mLayerWeights);
            mLayerZ = std::move(l.mLayerZ);
            mLayerBias = std::move(l.mLayerBias);
            mLayerZActivated = std::move(l.mLayerZActivated);
            mLayerWGradients = std::move(l.mLayerWGradients);
            mLayerBGradients = std::move(l.mLayerBGradients);
            mActivationPtr = std::move(l.mActivationPtr);
            
            l.mLayerId = 0;
            l.mPerceptronNo = 0;
            l.mLearnableCoeffs = 0;
        }

        Dense::Dense(const uint8_t perceptronNo) : Layer(perceptronNo)
        {
            ++mInstances;
        }

    }
}