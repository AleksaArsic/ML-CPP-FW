#include "Model.hpp"
#include <fstream>
#include <iostream>

// Add new layer to the NN Model
bool Model::addLayer(Layer layer)
{
    try
    {
        layer.set_mLayerId(this->mLayersNo++);

        this->mLayers.push_back(std::move(std::make_unique<Layer>(layer)));

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Model.addLayer() operation failed: ";
        std::cerr << e.what() << std::endl;
        return false;
    }
    
}

// Compile model with added layers, optimizer, loss function and metrics 
bool Model::compileModel()
{
    // iterate trough layers
    for(auto it = mLayers.begin(); it != mLayers.end(); it++)
    {
        uint8_t layerId = (*it)->get_mLayerId();
        uint8_t perceptronNo = (*it)->get_mPerceptronNo();
        uint8_t prevPercNo = (0 == layerId ? perceptronNo : mLayers[layerId - 1]->get_mPerceptronNo());

        // calculate learnable coefficients
        // learnableCoeffs = noOfPerceptrons * (noOfWeights + noOfInputs) + 1 (bias)
        uint32_t noOfCoeffs = (perceptronNo * (2 * prevPercNo)) + 1;

        (*it)->set_mLearnableCoeffs(noOfCoeffs);
        mLearnableCoeffs += noOfCoeffs;

        // iterate trough Perceptrons
        for(uint8_t i = 0; i < perceptronNo; i++)
        {
            // initialize W, x, b for each Perceptron in each Layer
            // bias - b - can be common for layer instead of having it in each Perceptron
            Perceptron& p = (*it)->get_Perceptron(i);
            p.w = Eigen::MatrixXd::Random(1, prevPercNo);
            p.x = Eigen::MatrixXd::Random(prevPercNo, 1);
            p.b = 1.0;

#if 0
            std::cout << "LayerId: " << static_cast<int>(layerId) << std::endl;
            std::cout << "(" << (*it)->get_Perceptron(i).w.rows() << ", " << (*it)->get_Perceptron(i).w.cols() << ")" << std::endl;
            std::cout << "(" << (*it)->get_Perceptron(i).x.rows() << ", " << (*it)->get_Perceptron(i).x.cols() << ")" << std::endl;
            std::cout << "Learnable coefficients: " << (*it)->get_mLearnableCoeffs() << std::endl;
            std::cout << std::endl;
#endif
        }
    }
    
    this->mIsCompiled = true;

    return this->mIsCompiled;
}

// Save model weights to desired location
bool Model::saveModel(std::string modelPath) const
{
    try
    {
        std::ofstream modelFile;
        modelFile.open (modelPath, std::ios::out);

        for(auto it = mLayers.begin(); it != mLayers.end(); it++)
        {
            modelFile << "Placeholder.\n"; 
        }

        modelFile.close();

        return true;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Model.saveModel() failed: ";
        std::cerr << e.what() << '\n';
        return false;
    }

}

// Load model weights from desired location
bool Model::loadModel()
{
    return false;
}

// Train desired model
void Model::modelFit()
{

}

// Trained model predict on provided input data
void Model::modelPredict() const
{

}

// Show model summary by printing it on std::cout
void Model::modelSummary() const
{
    if(this->mIsCompiled)
    {
        std::cout << "**************************************" << std::endl;
        std::cout << "Model summary: " << std::endl;
        std::cout << "**************************************" << std::endl;

        for (auto it = mLayers.begin(); it != mLayers.end(); it++)
        {
            std::cout << "Layer: " << static_cast<uint32_t>((*it)->get_mLayerId()) << std::endl;
            std::cout << "\t Perceptrons = " << static_cast<uint32_t>((*it)->get_mPerceptronNo()) << std::endl;
            std::cout << "\t Coeffs = " << (*it)->get_mLearnableCoeffs() << std::endl;
            std::cout << "**************************************" << std::endl;
        }

        std::cout << "Total learnable coefficients = " << this->mLearnableCoeffs << std::endl;
        std::cout << "**************************************" << std::endl;
    }
    else
    {
        std::cout << "Model is not compiled!" << std::endl;
    }
}