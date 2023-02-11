#ifndef DATAHANDLER_UTILITIES_HPP
#define DATAHANDLER_UTILITIES_HPP

#include <memory>
#include <iostream>
#include <random>
#include "../Eigen/Dense"


namespace NNFramework
{
    namespace DataHandler
    {
        // Class DataHandler does not store any kind of that 
        // it is a simple interface for handling and manipulating with provided data
        // such as: Data normalization and denormalization, data shuffle, etc.
        // It is constructed as a Singleton Class as it is "container" for data manipulation methods.
        class DataHandler final
        {
            public:
                // Delete copy constructor
                DataHandler(DataHandler& dHandler) = delete;
                // Delete move constructor
                DataHandler(DataHandler&& dHandler) = delete;

                // Delete copy assignment operator
                DataHandler& operator=(const DataHandler& dHandler) = delete;
                // Delete move assignment operator
                DataHandler& operator=(DataHandler&& dHandler) = delete;

                // Retrieve singleton object reference
                static std::unique_ptr<DataHandler>& getInstance();

                // Normalize data in range [0, 1]
                Eigen::MatrixXd& normalizeData(Eigen::MatrixXd& data);

                // Denormalize data from range [0, 1] to [min, max]
                Eigen::MatrixXd& denormalizeData(Eigen::MatrixXd& data, double min, double max);

                // Shuffle data matrices
                // shuffleData() shuffles data where original indices in inData matrix will match indices in expData matrix
                // i.e. shuffleData() considers inData[0] and expData[0] to be a pair (inData[0], expData[0])
                // works with original matrices 
                void shuffleData(Eigen::MatrixXd& inData, Eigen::MatrixXd& expData);

            private:
                DataHandler() { }
        };
    }
}
#endif