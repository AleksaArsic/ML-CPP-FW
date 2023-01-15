#ifndef LOSS_HPP
#define LOSS_HPP

#include <string>

namespace Loss
{
    template<class TypeName> struct LossType { typedef TypeName T; }; 

    struct LossFunctor
    {
        virtual std::string name() const = 0;
        virtual double operator()(const double x, const double y) const = 0;
    };

    struct MeanSquareError : LossFunctor
    {
        virtual std::string name() const
        {
            return "meansquareerror";
        }

        virtual double operator()(const double x, const double y) const
        {
            return 0.0;
        }      
    }
}

#endif