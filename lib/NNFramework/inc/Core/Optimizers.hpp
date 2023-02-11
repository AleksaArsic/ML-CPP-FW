#ifndef OPTIMIZERS_CORE_HPP
#define OPTIMIZERS_CORE_HPP

#include <memory>
#include "Layers.hpp"
#include "../Eigen/Dense"
#include "../Common/Common.hpp"

namespace NNFramework
{
    namespace Optimizers
    {
        template<class TypeName> struct OptimizersType { typedef TypeName T; }; 

        struct OptimizersFunctor
        {
            double learningRate = 0.5;
            
            virtual std::string name() const = 0;
            virtual void operator()(const std::vector<std::unique_ptr<Layers::Layer>>& layers) const = 0;
        };

        struct GradientDescent final : OptimizersFunctor
        {

            std::string name() const override
            {
                return "GradientDescent";
            }

            void operator()(const std::vector<std::unique_ptr<Layers::Layer>>& layers) const override
            {
                std::shared_ptr<Eigen::MatrixXd> layerWeights;
                std::shared_ptr<Eigen::MatrixXd> layerWeightsGradients;
                std::shared_ptr<Eigen::MatrixXd> layerBias;
                std::shared_ptr<Eigen::MatrixXd> layerBiasGradients;

                // skip first layer as there are no gradients calculated for the pass trough layer
                for(uint32_t i = (INPUT_LAYER_IDX + 1L); i < layers.size(); ++i)
                {
                    layerWeights = layers[i]->get_mLayerWeights();
                    layerWeightsGradients = layers[i]->get_mLayerWGradients();
                    layerBias = layers[i]->get_mLayerBias();
                    layerBiasGradients = layers[i]->get_mLayerBGradients();

                    // w(t+1) = w(t) - lr * dL/dW -> t = epoch
                    (*layerWeights) = (*layerWeights) - ((*layerWeightsGradients) * learningRate);

                    // b(t+1) = b(t) - lr * dL/dB -> t = epoch
                    Eigen::MatrixXd biasGradSum = Eigen::MatrixXd::Zero((*layerBiasGradients).rows(), (*layerBiasGradients).cols());
                    biasGradSum.array() += ((*layerBiasGradients).sum() / (*layerBiasGradients).rows());
                    
                    (*layerBias) = (*layerBias) - ((biasGradSum) * learningRate);
                }
            }
        };
    }
}

#endif