#include "Core/Layers.hpp"

namespace NNFramework
{
    namespace Layers
    {
        Layer::Layer(const uint8_t perceptronNo) : mLayerId(0), mPerceptronNo(perceptronNo), mLearnableCoeffs(0)
        {
            mLayerWeights = std::make_shared<Eigen::MatrixXd>();
            mLayerZ = std::make_shared<Eigen::VectorXd>();
            mLayerBias = std::make_shared<Eigen::VectorXd>();
            mActivationPtr = nullptr;
        }

        Layer::Layer(Layer&& l) : mLayerId(l.mLayerId), mPerceptronNo(l.mPerceptronNo), mLearnableCoeffs(l.mLearnableCoeffs)
        {
            mLayerWeights = std::move(l.mLayerWeights);
            mLayerZ = std::move(l.mLayerZ);
            mLayerBias = std::move(l.mLayerBias);
            mActivationPtr = std::move(l.mActivationPtr);
            
            l.mLayerId = 0;
            l.mPerceptronNo = 0;
            l.mLearnableCoeffs = 0;
        }

        Dense::Dense(const uint8_t perceptronNo) : Layer(perceptronNo)
        {
            mInstances++;
        }

    }
}