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
            
            double operator()(const double x, const bool derive = false) const
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
            
            virtual double activate(const double x) const = 0;
            virtual double derivative(const double x) const = 0;
        };

        struct InputActivation final : ActivationFunctor 
        {
            std::string name() const override
            {
                return "InputActivation";
            }

            double activate(const double x) const override 
            { 
                return x;
            }

            double derivative(const double x) const override 
            {
                return 0.0;
            }
        };

        struct Sigmoid final : ActivationFunctor
        {
            std::string name() const override
            {
                return "Sigmoid";
            }

            double activate(const double x) const override 
            { 
                return (1.0 / (1.0 + std::exp(-x)));
            }

            double derivative(const double x) const override 
            {
                return activate(x) * (1 - activate(x));
            }
        };

        struct Relu final : ActivationFunctor
        {
            std::string name() const override
            {
                return "Relu";
            }

            double activate(const double x) const override 
            { 
                return std::max(0.0, x);
            }

            double derivative(const double x) const override 
            {
                if(0.0 > x)
                {
                    return 0.0;
                }
                else if (0.0 < x)
                {
                    return 1.0;
                }
                else
                {
                    std::cout << __FUNCTION__ << ": ";
                    throw std::runtime_error("Activation derivative undefined in zero!");    
                }
            }
        };

        struct LeakyRelu final : ActivationFunctor
        {
            double factor = 0.01; // f(y) = a*y -> when a is not 0.01 than it's called Randomized ReLU as per: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

            std::string name() const override
            {
                return "LeakyRelu";
            }

            double activate(const double x) const override 
            { 
                if(x >= 0)
                {
                    return x;
                }
                else
                {
                    return factor * x; 
                }
            }

            double derivative(const double x) const override 
            {
                if(0.0 < x)
                {
                    return 1.0;
                }
                else if (0.0 > x)
                {
                    return factor;
                }
                else
                {
                    std::cout << __FUNCTION__ << ": ";
                    throw std::runtime_error("Activation derivative undefined in zero!");    
                }
            }
        };
    }
}
#endif