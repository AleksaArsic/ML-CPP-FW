#include "Utilities/DataHandler.hpp"

namespace NNFramework
{
    namespace DataHandler
    {
        // Retrieve singleton object reference
        std::unique_ptr<DataHandler>& DataHandler::getInstance()
        {
            static std::unique_ptr<DataHandler> instance;
            
            if (nullptr == instance.get()) 
            {
                instance = std::unique_ptr<DataHandler>(new DataHandler());
            }
            else
            {
                /* Do nothing. */
            }

            return instance;
        }

        // Normalize data in range [0, 1]
        Eigen::MatrixXd& DataHandler::normalizeData(Eigen::MatrixXd& data)
        {
            double min = data.minCoeff();
            double max = data.maxCoeff();

            data = data.unaryExpr([min, max](double x){ return (x - min) / (max - min); });

            return data;
        }

        // Denormalize data from range [0, 1] to [min, max]
        Eigen::MatrixXd& DataHandler::denormalizeData(Eigen::MatrixXd& data, double min, double max)
        {
            data = data.unaryExpr([min, max](double x){ return (x * (max - min) + min); });

            return data;
        }

        // Shuffle data matrices
        // shuffleData() shuffles data where original indices in inData matrix will match indices in expData matrix
        // i.e. shuffleData() considers inData[0] and expData[0] to be a pair (inData[0], expData[0])
        // works with original matrices 
        void DataHandler::shuffleData(Eigen::MatrixXd& inData, Eigen::MatrixXd& expData)
        {
            std::random_device randDevice;
            std::seed_seq rngSeed{randDevice(), randDevice(), randDevice(), randDevice(), randDevice(), randDevice(), randDevice(), randDevice()};

            // Create random engines with the rng seed
            std::mt19937 engine(rngSeed);

            // Create permutation Matrix with the size of the rows
            Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permMat(inData.rows());

            permMat.setIdentity();

            std::shuffle(permMat.indices().data(), permMat.indices().data() + permMat.indices().size(), engine);

            inData = permMat * inData;   // Shuffle row wise
            expData = permMat * expData; // Shuffle row wise
        }
    }
}