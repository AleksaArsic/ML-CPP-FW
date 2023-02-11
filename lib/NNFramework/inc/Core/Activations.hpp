#ifndef ACTIVATIONS_CORE_HPP
#define ACTIVATIONS_CORE_HPP

#include <cmath>
#include <string>
#include <iostream>

namespace NNFramework
{
    namespace Activations
    {

        template<class TypeName> struct ActivationType { typedef TypeName T; }; 

        struct ActivationFunctor
        {
            virtual std::string name() const = 0;
            
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& x, const bool derive = false) const
            {
                if(derive)
                {
                    return derivative(x);
                }
                else
                {
                    return activate(x);
                }
            }
            
            virtual Eigen::MatrixXd activate(const Eigen::MatrixXd& x) const = 0;
            virtual Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) const = 0;
        };

        struct InputActivation final : ActivationFunctor 
        {
            std::string name() const override
            {
                return "InputActivation";
            }

            Eigen::MatrixXd activate(const Eigen::MatrixXd& x) const override 
            { 
                return x;
            }

            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) const override 
            {
                Eigen::MatrixXd retVec = Eigen::MatrixXd::Zero(x.rows(), x.cols());
                return retVec;
            }
        };

        struct Sigmoid final : ActivationFunctor
        {
            std::string name() const override
            {
                return "Sigmoid";
            }

            Eigen::MatrixXd activate(const Eigen::MatrixXd& x) const override 
            { 
                Eigen::MatrixXd retVec = x;
                
                retVec *= -1.0;
                retVec = retVec.unaryExpr([](const double& el) { return std::exp(el); });

                retVec = retVec + Eigen::MatrixXd::Ones(retVec.rows(), retVec.cols());

                return retVec.unaryExpr([](const double& el){ return 1.0 / el; });
            }

            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) const override 
            {
                Eigen::MatrixXd activated = activate(x);
                return activated.cwiseProduct(Eigen::MatrixXd::Ones(activated.rows(), activated.cols()) - activated);
            }
        };

        struct Relu final : ActivationFunctor
        {
            std::string name() const override
            {
                return "Relu";
            }

            Eigen::MatrixXd activate(const Eigen::MatrixXd& x) const override 
            { 
                Eigen::MatrixXd retVec = x.unaryExpr([](const double& el){ return std::max(0.0, el); });

                return retVec;
            }

            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) const override 
            {
                std::string fname = __FUNCTION__;

                Eigen::MatrixXd retVec = x.unaryExpr(
                    [fname](const double& el)
                    {
                        if(0.0 > el)
                        {
                            return 0.0;
                        }
                        else if (0.0 < el)
                        {
                            return 1.0;
                        }
                        else
                        {
                            std::cout << fname << ": ";
                            throw std::runtime_error("Activation derivative undefined in zero!");    
                        }
                    }
                );

                return retVec;
            }
        };

        struct LeakyRelu final : ActivationFunctor
        {
            double factor = 0.01; // f(y) = a*y -> when a is not 0.01 than it's called Randomized ReLU as per: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

            std::string name() const override
            {
                return "LeakyRelu";
            }

            Eigen::MatrixXd activate(const Eigen::MatrixXd& x) const override 
            { 
                Eigen::MatrixXd retVec = x.unaryExpr([this](const double& el) { return (el >= 0) ? el : factor * el; } );

                return retVec;
            }

            Eigen::MatrixXd derivative(const Eigen::MatrixXd& x) const override 
            {
                std::string fname = __FUNCTION__;

                Eigen::MatrixXd retVec = x.unaryExpr(
                    [this, fname](const double& el)
                    {
                        if(0.0 < el)
                        {
                            return 1.0;
                        }
                        else if (0.0 > el)
                        {
                            return factor;
                        }
                        else
                        {
                            std::cout << fname << ": ";
                            throw std::runtime_error("Activation derivative undefined in zero!");    
                        }
                    }
                );

                return retVec;
            }
        };
    }
}
#endif