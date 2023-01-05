#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "Model.hpp"

int main()
{
    Model model;

    model.addLayer(Layer(3));
    model.addLayer(Layer(4));
    model.addLayer(Layer(3));

    model.compileModel();
    model.modelSummary();

    model.saveModel();

    return 0;
}