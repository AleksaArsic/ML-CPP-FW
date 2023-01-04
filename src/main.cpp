#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "Model.hpp"

int main()
{
    Model model;

    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;

    std::cout << "m = " << std::endl << m << std::endl;

    return 0;
}