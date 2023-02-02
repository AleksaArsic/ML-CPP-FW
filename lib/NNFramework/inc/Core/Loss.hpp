#ifndef LOSS_CORE_HPP
#define LOSS_CORE_HPP

#include <string>
#include "../Eigen/Dense"
#include <math.h>

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
            Eigen::VectorXd operator()(const Eigen::VectorXd& x, const Eigen::VectorXd& y, const bool derive = false) const
            {
                if(derive)
                {
                    return derivative(x, y);
                }
                else
                {
                    return loss(x, y);
                }
            }
            
            virtual Eigen::VectorXd loss(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const = 0;
            virtual Eigen::VectorXd derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const = 0;
        };

        struct MeanSquaredError final : LossFunctor
        {
            std::string name() const override
            {
                return "MeanSquaredError";
            }

            // param: x -> expected
            // param: y -> predicted  
            Eigen::VectorXd loss(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd diffSquared = x - y;
                diffSquared = diffSquared.cwiseProduct(diffSquared);

                return 0.5 *  diffSquared;
            }

            Eigen::VectorXd derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd diff = y - x;

                return diff;
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
            Eigen::VectorXd loss(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd diffAbs = x - y;
                diffAbs = diffAbs.cwiseAbs();

                return diffAbs;
            }

            // derivative of MeanAbsoluteError is not defined in 0
            Eigen::VectorXd derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd result(x.size());

                for(uint32_t i = 0; i < result.size(); ++i)
                {
                    result[i] = (y[i] > x[i] ? 1.0 : (y[i] < x[i] ? -1.0 : NAN));
                }

                return result;
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
            Eigen::VectorXd loss(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd predLogFirst = y.array().log10().matrix();
                Eigen::VectorXd predLogSecond = ((Eigen::VectorXd::Ones(y.size()) - y).array().log10()).matrix();
                Eigen::VectorXd expectedDiff = (Eigen::VectorXd::Ones(x.size()) - x);

                Eigen::VectorXd retLoss = -1 * (x.cwiseProduct(predLogFirst) + expectedDiff.cwiseProduct(predLogSecond));

                return retLoss;
            }
            Eigen::VectorXd derivative(const Eigen::VectorXd& x, const Eigen::VectorXd& y) const
            {
                Eigen::VectorXd result(x.size());
                
                for(uint32_t i = 0; i < result.size(); ++i)
                {
                    result[i] = (y[i] - x[i]) / (y[i] * (1 - y[i]));
                }

                return result;
            }  
        };
    }
}

#endif