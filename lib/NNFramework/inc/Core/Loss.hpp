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
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const bool derive = false) const
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
            
            virtual Eigen::MatrixXd loss(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
            virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const = 0;
        };

        struct MeanSquaredError final : LossFunctor
        {
            std::string name() const override
            {
                return "MeanSquaredError";
            }

            // param: x -> expected
            // param: y -> predicted  
            Eigen::MatrixXd loss(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd diffSquared = x - y;
                diffSquared = diffSquared.cwiseProduct(diffSquared);

                return 0.5 *  diffSquared;
            }

            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd diff = y - x;

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
            Eigen::MatrixXd loss(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd diffAbs = x - y;
                diffAbs = diffAbs.cwiseAbs();

                return diffAbs;
            }

            // derivative of MeanAbsoluteError is not defined in 0
            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd result(x.rows(), x.cols());

                for(uint32_t i = 0; i < result.rows(); ++i)
                {
                    for(uint32_t j = 0; j < result.cols(); ++j)
                    {
                        result(i, j) = (y(i, j) > x(i, j) ? 1.0 : (y(i, j) < x(i, j) ? -1.0 : NAN));
                    }
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
            Eigen::MatrixXd loss(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd predLogFirst = y.array().log10().matrix();
                Eigen::MatrixXd predLogSecond = ((Eigen::MatrixXd::Ones(y.rows(), y.cols()) - y).array().log10()).matrix();
                Eigen::MatrixXd expectedDiff = (Eigen::MatrixXd::Ones(x.rows(), x.cols()) - x);

                Eigen::MatrixXd retLoss = -1 * (x.cwiseProduct(predLogFirst) + expectedDiff.cwiseProduct(predLogSecond));

                return retLoss;
            }
            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) const
            {
                Eigen::MatrixXd result(x.rows(), x.cols());

                for(uint32_t i = 0; i < result.rows(); ++i)
                {
                    for(uint32_t j = 0; j < result.cols(); ++j)
                    {
                        result(i, j) = (y(i, j) - x(i, j)) / (y(i, j) * (1 - y(i, j)));
                    }
                }

                return result;
            }  
        };
    }
}

#endif