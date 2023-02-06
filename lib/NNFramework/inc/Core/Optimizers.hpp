#ifndef OPTIMIZERS_CORE_HPP
#define OPTIMIZERS_CORE_HPP

#include <memory>
#include "Layers.hpp"
#include "../Eigen/Dense"

namespace NNFramework
{
    namespace Optimizers
    {
        template<class TypeName> struct OptimizersType { typedef TypeName T; }; 

        struct OptimizersFunctor
        {
            virtual std::string name() const = 0;
            virtual void operator()(const std::vector<std::shared_ptr<Layers::Layer>> layers) const = 0;
        };

        struct GradientDescent final : OptimizersFunctor
        {
            double learningRate = 0.1;

            std::string name() const override
            {
                return "GradientDescent";
            }

            void operator()(const std::vector<std::shared_ptr<Layers::Layer>> layers) const override
            {
                std::shared_ptr<Eigen::MatrixXd> layerWeights;
                std::shared_ptr<Eigen::MatrixXd> layerWeightsGradients;
                std::shared_ptr<Eigen::MatrixXd> layerBias;
                std::shared_ptr<Eigen::MatrixXd> layerBiasGradients;

                // skip first layer as there are no gradients calculated for the pass trough layer
                for(uint32_t i = 1; i < layers.size(); ++i)
                {
                    layerWeights = layers[i]->get_mLayerWeights();
                    layerWeightsGradients = layers[i]->get_mLayerWGradients();
                    layerBias = layers[i]->get_mLayerBias();
                    layerBiasGradients = layers[i]->get_mLayerBGradients();

                    // w(t+1) = w(t) - lr * dL/dW -> t = epoch
                    (*layerWeights) = (*layerWeights) - ((*layerWeightsGradients) * learningRate);
                    // b(t+1) = b(t) - lr * dL/dB -> t = epoch
                    (*layerBias) = (*layerBias) - ((*layerBiasGradients) * learningRate);

                }
            }
        };

        struct StohasticGradientDescent final : OptimizersFunctor
        {
            std::string name() const override
            {
                return "StohasticGradientDescent";
            }

            void operator()(const std::vector<std::shared_ptr<Layers::Layer>> layers) const override
            {

            }
        };
    }
}

#endif