#ifndef MODELCONFIGURATION_CORE_HPP
#define MODELCONFIGURATION_CORE_HPP

#include "Loss.hpp"
#include "Metrics.hpp"
#include "Optimizers.hpp"

namespace NNFramework
{
    namespace Model
    {
        namespace ModelConfiguration
        {
            // class specific for defining model configuration such as:
            // Loss function
            // Metrics
            // Optimization algorithm
            class ModelConfiguration final
            {
                public: 
                    // Loss functor unique_ptr
                    std::unique_ptr<Loss::LossFunctor> mLossPtr;

                    // Metrics functor unique_ptr
                    std::unique_ptr<Metrics::MetricsFunctor> mMetricsPtr;

                    // Optimizer functor unique_ptr
                    std::unique_ptr<Optimizers::OptimizersFunctor> mOptimizerPtr;

                    template<class X, class Y, class Z>
                    ModelConfiguration(Loss::LossType<X>, Metrics::MetricsType<Y>, Optimizers::OptimizersType<Z>)
                    {
                        // bind loss functor to the model configuration
                        mLossPtr = std::make_unique<X>();

                        // bind metrics functor to the model configuration
                        mMetricsPtr = std::make_unique<Y>();

                        // bind optimizer functor to the model configurations
                        mOptimizerPtr = std::make_unique<Z>();
                    }

                    // Delete default constructor
                    ModelConfiguration() = delete;

                    // Default destructor
                    ~ModelConfiguration() = default;

                    // Delete copy constructor
                    ModelConfiguration(ModelConfiguration& m) = delete;

                    // Move constructor
                    ModelConfiguration(ModelConfiguration&& m) : mLossPtr(std::move(m.mLossPtr)), mMetricsPtr(std::move(m.mMetricsPtr)), mOptimizerPtr(std::move(m.mOptimizerPtr))
                    { }
                    
                    // Delete copy assignment operator
                    ModelConfiguration& operator=(const ModelConfiguration& d) = delete;

                    // Move assignment operator
                    ModelConfiguration& operator=(const ModelConfiguration&& d) = delete;
            };

        }
    }
}

#endif