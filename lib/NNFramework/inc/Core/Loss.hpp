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
            virtual Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const = 0;
        };

        struct MeanSquaredError final : LossFunctor
        {
            std::string name() const override
            {
                return "MeanSquaredError";
            }

            // param: x -> expected
            // param: y -> predicted
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                Eigen::VectorXd diffSquared = x - y;
                diffSquared = diffSquared.cwiseProduct(diffSquared);

                return diffSquared;
            }      
        };

        struct MeanAbsoluteError final : LossFunctor
        {
            std::string name() const override
            {
                return "MeanAbsoluteError";
            }

            // param: x -> expected
            // param: y -> predicted
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                Eigen::VectorXd diffAbs = x - y;
                diffAbs = diffAbs.cwiseAbs();

                return diffAbs;
            }      
        };

        struct BinaryCrossEntropy final : LossFunctor
        {
            std::string name() const override
            {
                return "BinaryCrossEntropy";
            }

            // param: x -> expected
            // param: y -> predicted
            // BCELoss =  −(x * log(y) + (1−x) * log(1−y))
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const override
            {
                Eigen::VectorXd predLogFirst = y.array().log10().matrix();
                Eigen::VectorXd predLogSecond = ((Eigen::VectorXd::Ones(y.size()) - y).array().log10()).matrix();
                Eigen::VectorXd expectedDiff = (Eigen::VectorXd::Ones(x.size()) - x);

                Eigen::VectorXd loss = -1 * (x.cwiseProduct(predLogFirst) + expectedDiff.cwiseProduct(predLogSecond));

                return loss;
            }      
        };
    }
}

#endif