#ifndef METRICS_CORE_HPP
#define METRICS_CORE_HPP

#include <string>
#include "../Eigen/Dense"

namespace NNFramework
{
    namespace Metrics
    {
        template<class TypeName> struct MetricsType { typedef TypeName T; }; 

        struct MetricsFunctor
        {
            virtual std::string name() const = 0;
            virtual double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const = 0;
        };

        struct ClassificationAccuracy : MetricsFunctor
        {
                double threshold = 0.1;

                ClassificationAccuracy() = default;
                ClassificationAccuracy(const double thr) : threshold(thr) {}

                std::string name() const override
                {
                    return "ClassificationAccuracy";
                }

                // param: x -> expected
                // param: y -> predicted
                // acc = correct / noofpred
                double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
                {
                    Eigen::VectorXd diffBool = x - y;
                    diffBool = diffBool.cwiseAbs();
                    diffBool = diffBool.unaryExpr([this](double x){ return (x <= threshold) ? 1.0 : 0.0; });

                    double correct = diffBool.sum();

                    return correct / diffBool.size();
                }
        };

        struct MeanSquaredError : MetricsFunctor
        {
            std::string name() const override
            {
                return "MeanSquaredError";
            }

            // param: x -> expected
            // param: y -> predicted
            double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                Eigen::VectorXd diffSquared = x - y;
                diffSquared = diffSquared.cwiseProduct(diffSquared);
                
                return (diffSquared.sum() / diffSquared.size());
            }      
        };

        struct MeanAbsoluteError : MetricsFunctor
        {
            std::string name() const override
            {
                return "MeanAbsoluteError";
            }

            // param: x -> expected
            // param: y -> predicted
            double operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                Eigen::VectorXd diffAbs = x - y;
                diffAbs = diffAbs.cwiseAbs();

                return (diffAbs.sum() / diffAbs.size());
            }      
        };

    }
}

#endif