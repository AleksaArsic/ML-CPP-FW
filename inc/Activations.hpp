#ifndef ACTIVATIONS_HPP
#define ACTIVATIONS_HPP

#include <cmath>
#include <string>
#include <algorithm>

namespace Activations
{

    template<class TypeName> struct ActivationType { typedef TypeName T; }; 

    struct ActivationFunctor
    {
        virtual std::string name() const = 0;
        virtual double operator()(const double x) const = 0;
    };

    struct InputActivation : ActivationFunctor
    {
        std::string name() const override
        {
            return "inputlayer";
        }

        double operator()(const double x) const override 
        {
            return x; // input layer does not have activation as it's purpose is to delegate the input signal to the deep layers
        }
    };

    struct Sigmoid : ActivationFunctor
    {
        std::string name() const override
        {
            return "sigmoid";
        }

        double operator()(const double x) const override
        {
            return (1.0 / (1.0 + std::exp(-x)));
        }
    };

    struct Relu : ActivationFunctor
    {
        std::string name() const override
        {
            return "relu";
        }

        double operator()(const double x) const override
        {
            return std::max(0.0, x);
        }
    };

    struct LeakyRelu : ActivationFunctor
    {
        std::string name() const override
        {
            return "leakyrelu";
        }

        double operator()(const double x) const override
        {
            if(x >= 0)
            {
                return x;
            }
            else
            {
                return 0.01 * x; // f(y) = a*y -> when a is not 0.01 than it's called Randomized ReLU as per: https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6
            }
        }
    };
}

#endif