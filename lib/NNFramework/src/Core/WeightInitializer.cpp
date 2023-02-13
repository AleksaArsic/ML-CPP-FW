#include "Core/WeightInitializer.hpp"

namespace NNFramework
{
    namespace Model
    {
        namespace WeightInitializer
        {
            // Initialize weights based on the activation function
            void WeightInitializer::initializeWeights(const std::shared_ptr<Eigen::MatrixXd>& weights, std::string activationName)
            {

                // Activations::ActivationTypeEnum won't work here as we are having pointers to the 
                // Activations::ActivationFunctor in the actual layers
                // room for future improvement
                if("Sigmoid" == activationName)
                {
                    set_XavierGlorotParameters((*weights).cols(), (*weights).rows());
                    *weights = (*weights).unaryExpr([this](double x){ return mUniformDistribution(mGenerator); });
                }
                else
                {
                    set_KaimingHeParameters((*weights).cols());
                    *weights = (*weights).unaryExpr([this](double x){ return mNormalDistribution(mGenerator); });
                }
            }

            // set normal distribution parameters
            void WeightInitializer::set_XavierGlorotParameters(const double& prevPercNo, const double& perceptronNo)
            {
                std::uniform_real_distribution<double>::param_type distParam(-1 * std::sqrt(2.0) / std::sqrt((prevPercNo + perceptronNo)), std::sqrt(2.0) / std::sqrt((prevPercNo + perceptronNo)));
                mUniformDistribution.param(distParam);
            }
            
            // set uniform distribution parameters
            void WeightInitializer::set_KaimingHeParameters(const double& prevPercNo)
            {
                std::normal_distribution<double>::param_type distParam(0.0, std::sqrt(2.0 / prevPercNo));
                mNormalDistribution.param(distParam);
            }

        }
    }
}