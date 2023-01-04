#include <iostream>
#include <vector>
#include "test.hpp"

int main()
{
    std::vector<Test> testVector {Test{}, Test{}, Test{}, Test{}, Test{}, Test{}};

    std::cout << testVector[0] << std::endl;

    return 0;
}