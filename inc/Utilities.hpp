#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include "matplot/matplot.h"
#include "../src/NNFramework/inc/Eigen/Dense" // ugly ? 

void plotTest()
{
    using namespace matplot;
    std::vector<double> x = linspace(0, 2 * pi);
    std::vector<double> y = transform(x, [](auto x) { return sin(x); });

    plot(x, y, "-o");
    hold(on);
    plot(x, transform(y, [](auto y) { return -y; }), "--xr");
    plot(x, transform(x, [](auto x) { return x / pi - 1.; }), "-:gs");
    plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

    show();
}

std::tuple<Eigen::VectorXd, Eigen::VectorXd> loadData(const std::string path)
{
    std::fstream inputFile;

    inputFile.open(path, std::ios::in);

    if (inputFile.is_open())
    { 
        std::string line;
        std::vector<double> inDataVec;
        std::vector<double> expDataVec;

        while(getline(inputFile, line)) // parse one line of the file
        {
            std::stringstream s(line);
            double inputVal;
            double expectedVal;

            s >> inputVal >> expectedVal;

            inDataVec.push_back(inputVal);
            expDataVec.push_back(expectedVal);
        }

        inputFile.close(); // close the file object.

        // convert std::vector<double> to Eigen::VectorXd
        Eigen::VectorXd inEigenV = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(inDataVec.data(), inDataVec.size());
        Eigen::VectorXd expEigenV = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(expDataVec.data(), expDataVec.size());

        return std::make_tuple(inEigenV, expEigenV);
    }

    return std::make_tuple(Eigen::VectorXd(), Eigen::VectorXd());
}