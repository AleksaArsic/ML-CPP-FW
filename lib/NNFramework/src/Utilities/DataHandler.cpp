#include "Utilities/DataHandler.hpp"

namespace NNFramework
{
    namespace DataHandler
    {
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
    }
}