#ifndef OPTIMIZERS_CORE_HPP
#define OPTIMIZERS_CORE_HPP


namespace NNFramework
{
    namespace Optimizers
    {
        template<class TypeName> struct OptimizersType { typedef TypeName T; }; 

        struct OptimizersFunctor
        {
            virtual std::string name() const = 0;
            virtual double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const = 0;
        };

        struct GradientDescent final : OptimizersFunctor
        {
            std::string name() const override
            {
                return "GradientDescent";
            }

            double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                return 0.0;
            }
        };

        struct StohasticGradientDescent final : OptimizersFunctor
        {
            std::string name() const override
            {
                return "StohasticGradientDescent";
            }

            double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                return 0.0;
            }
        };
    }
}

#endif