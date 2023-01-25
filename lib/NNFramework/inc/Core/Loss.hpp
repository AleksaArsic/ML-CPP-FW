#ifndef LOSS_CORE_HPP
#define LOSS_CORE_HPP

#include <string>
#include "../Eigen/Dense"

namespace NNFramework
{
    namespace Loss
    {
        template<class TypeName> struct LossType { typedef TypeName T; }; 

        struct LossFunctor
        {
            virtual std::string name() const = 0;
            
            // param: x -> expected
            // param: y -> predicted
            virtual double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const = 0;
        };

        struct MeanSquaredError : LossFunctor
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

        struct MeanAbsoluteError : LossFunctor
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

        struct BinaryCrossEntropy : LossFunctor
        {
            std::string name() const
            {
                return "BinaryCrossEntropy";
            }

            // param: x -> expected
            // param: y -> predicted
            // BCELoss = (1/n) * Sum_of( −(x * log(y) + (1−x) * log(1−y) ) )
            double operator()(const Eigen::VectorXd x, const Eigen::VectorXd y) const
            {
                Eigen::VectorXd predLogFirst = y.array().log10().matrix();
                Eigen::VectorXd predLogSecond = ((Eigen::VectorXd::Ones(y.size()) - y).array().log10()).matrix();
                Eigen::VectorXd expectedDiff = (Eigen::VectorXd::Ones(x.size()) - x);

                double productSum = ((x.cwiseProduct(predLogFirst) + expectedDiff.cwiseProduct(predLogSecond))).sum();

                return ((-productSum) / x.size());
            }      
        };
    }
}

#endif