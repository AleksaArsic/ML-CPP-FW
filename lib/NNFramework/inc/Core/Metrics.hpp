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
            virtual double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const = 0;
        };

        struct ClassificationAccuracy : MetricsFunctor
        {
                std::string name()
                {
                    return "ClassificationAccuracy";
                }

                // param: x -> expected
                // param: y -> predicted
                // acc = correct / noofpred
                double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const
                {
                    
                    

                    return 0.0;
                }
        };

        struct MeanSquaredError : MetricsFunctor
        {
            std::string name() const
            {
                return "MeanSquaredError";
            }

            // param: x -> expected
            // param: y -> predicted
            double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const
            {
                Eigen::VectorXd diffSquared = x - y;
                diffSquared = diffSquared.cwiseProduct(diffSquared);
                
                return (diffSquared.sum() / diffSquared.size());
            }      
        };

        struct MeanAbsoluteError : MetricsFunctor
        {
            std::string name() const
            {
                return "MeanAbsoluteError";
            }

            // param: x -> expected
            // param: y -> predicted
            double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const
            {
                Eigen::VectorXd diffAbs = x - y;
                diffAbs = diffAbs.cwiseAbs();

                return (diffAbs.sum() / diffAbs.size());
            }      
        };

    }
}

#endif