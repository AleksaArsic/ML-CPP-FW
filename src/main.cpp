#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include "test.hpp"

int main()
{
    std::vector<Test> testVector {Test{}, Test{}, Test{}, Test{}, Test{}, Test{}};

    std::cout << testVector[0] << std::endl;

    Eigen::MatrixXd m(2, 2);
    m(0, 0) = 1;
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;

    std::cout << "m = " << std::endl << m << std::endl;

    return 0;
}