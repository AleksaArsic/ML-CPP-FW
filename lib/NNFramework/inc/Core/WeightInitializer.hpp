#ifndef WEIGHTINITIALIZER_CORE_HPP
#define WEIGHTINITIALIZER_CORE_HPP

#include <memory>
#include <random>
#include "../Eigen/Dense"

namespace NNFramework
{
    namespace Model
    {
        namespace WeightInitializer
        {
            class WeightInitializer final
            {
                public:
                    // Define default constructor
                    WeightInitializer() = default;

                    // Delete copy constructor
                    WeightInitializer(WeightInitializer& wInitializer) = delete;
                    // Delete move constructor
                    WeightInitializer(WeightInitializer&& wInitializer) = delete;

                    // Delete copy assignment operator
                    WeightInitializer& operator=(const WeightInitializer& wInitializer) = delete;
                    // Delete move assignment operator
                    WeightInitializer& operator=(WeightInitializer&& wInitializer) = delete;

                    // Initialize weights based on the activation function
                    void initializeWeights(const std::shared_ptr<Eigen::MatrixXd>& weights, std::string activationName);

                private:
                    // Initialization distributions
                    std::default_random_engine mGenerator;
                    std::normal_distribution<double> mNormalDistribution;
                    std::uniform_real_distribution<double> mUniformDistribution;

                    // set normal distribution parameters
                    void set_XavierGlorotParameters(const double& prevPercNo, const double& perceptronNo);
                    // set uniform distribution parameters
                    void set_KaimingHeParameters(const double& prevPercNo);
            };
        }
    }
}

#endif