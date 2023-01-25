#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <tuple>
#include "matplot/matplot.h"
#include "NNFramework/NNFramework"

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
// Function is specific for the given input format in data/input_data.txt
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> sortData(Eigen::VectorXd inData, Eigen::VectorXd outData)
{

    if(inData.size() != outData.size())
    {
        throw "Size of the input Eigen::VectorXd's are not the same! Data cannot be sorted.";
    }
    else
    {  
        std::map<double, double> data;
        Eigen::MatrixXd inDataSorted(inData.rows(), inData.cols());
        Eigen::MatrixXd outDataSorted(outData.rows(), outData.cols());

        for (uint32_t i = 0; i < inData.size(); ++i)
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

// Load data from the input file to Eigen::MatrixXd's
// Function is specific for the given input format in data/input_data.txt
// where delimiter is blank space character ' '
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> loadData(const std::string path, uint32_t sampleNo, uint8_t inCol, uint8_t expCol)
{
    std::fstream inputFile;

    inputFile.open(path, std::ios::in);

    if (inputFile.is_open())
    { 
        std::string line;
        Eigen::MatrixXd inData = Eigen::MatrixXd::Zero(sampleNo, inCol);
        Eigen::MatrixXd expData = Eigen::MatrixXd::Zero(sampleNo, expCol);

        uint32_t i = 0;
        while(getline(inputFile, line)) // parse one line of the file
        {
            std::stringstream s(line);

            for(uint32_t j = 0; j < (inCol + expCol); ++j)
            {
                if(j < inCol)
                {
                    s >> inData(i, j);
                }
                else
                {
                    s >> expData(i, j - inCol);
                }
            }
            i++;
        }

        inputFile.close(); // close the file object.

        return std::make_tuple(inData, expData);
    }
    else
    {
        throw "Input file is not opened.";
    }
}