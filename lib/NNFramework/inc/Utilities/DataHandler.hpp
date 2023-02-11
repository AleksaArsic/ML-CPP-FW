#ifndef DATAHANDLER_UTILITIES_HPP
#define DATAHANDLER_UTILITIES_HPP

#include <memory>
#include "../Eigen/Dense"
#include <iostream>

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
                int x;

                // Delete copy constructor
                DataHandler(DataHandler& wInitializer) = delete;
                // Delete move constructor
                DataHandler(DataHandler&& wInitializer) = delete;

                // Delete copy assignment operator
                DataHandler& operator=(const DataHandler& wInitializer) = delete;
                // Delete move assignment operator
                DataHandler& operator=(DataHandler&& wInitializer) = delete;

                static std::unique_ptr<DataHandler>& getInstance();

                void method()
                {
                    std::cout << "DataHandler" << std::endl;
                }

            private:
                DataHandler() { }
        };
    }
}
#endif