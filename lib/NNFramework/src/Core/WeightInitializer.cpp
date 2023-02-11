#include "Core/WeightInitializer.hpp"

namespace NNFramework
{
    namespace Model
    {
        namespace WeightInitializer
        {
            // Initialize weights based on the activation function
            void WeightInitializer::initializeWeights(std::shared_ptr<Eigen::MatrixXd> weights, std::string activationName)
            {
                // maybe use actual type instead of the name?
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
            void WeightInitializer::set_XavierGlorotParameters(double prevPercNo, double perceptronNo)
            {
                std::uniform_real_distribution<double>::param_type distParam(-1 * std::sqrt(2.0) / std::sqrt((prevPercNo + perceptronNo)), std::sqrt(2.0) / std::sqrt((prevPercNo + perceptronNo)));
                mUniformDistribution.param(distParam);
            }
            // set uniform distribution parameters
            void WeightInitializer::set_KaimingHeParameters(double prevPercNo)
            {
                std::normal_distribution<double>::param_type distParam(0.0, std::sqrt(2.0 / prevPercNo));
                mNormalDistribution.param(distParam);
            }

        }
    }
}