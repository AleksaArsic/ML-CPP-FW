#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include "matplot/matplot.h"
#include "NNFramework/inc/Eigen/Dense"

//void plotData(std::tuple<Eigen::VectorXd, Eigen::VectorXd> data, std::string savePath = "img/graph")
void plotData()
{
    using namespace matplot;
    auto f = figure(true);

    //plot(std::get<0>(data), std::get<1>(data));

    plot({0, 1, 2, 3}, {0, 1, 2, 3});

    std::string filename = "./test.pdf";
    save(filename); // currently is not supported by the matplot++ library
    //show();               // use show instead
}

// used for sorting pairs of (xi, yi)
std::tuple<Eigen::VectorXd, Eigen::VectorXd> sortData(Eigen::VectorXd inData, Eigen::VectorXd outData)
{

    if(inData.size() != outData.size())
    {
        throw "Size of the input Eigen::VectorXd's are not the same! Data cannot be sorted.";
    }
    else
    {  
        std::map<double, double> data;
        Eigen::VectorXd inDataSorted(inData.size());
        Eigen::VectorXd outDataSorted(outData.size());

        for (uint32_t i = 0; i < inData.size(); i++)
        {   
            // map is automatically sorted by key value
            data.insert(std::make_pair(inData[i], outData[i]));
        }

        uint32_t i = 0;
        for (const auto& [key, value] : data)
        {
            inData[i] = key;
            outData[i++] = value;
        }

        return std::make_tuple(inData, outData);
    }
}

// Load data from the input file to Eigen::VectorXd's
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
    else
    {
        throw "Input file is not opened.";
    }
}