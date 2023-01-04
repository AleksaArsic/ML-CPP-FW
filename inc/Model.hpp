#include <memory>
#include <vector>

struct Perceptron
{
    // y = wx + b
    double w;
    double x;
    double b;
};

class Model
{
    public:
        Model() = default;
        ~Model() = default;

        // Add new layer to the NN Model
        bool addLayer();

        // Compile model with added layers, optimizer, loss function and metrics 
        bool compileModel();

        // Save model weights to desired location
        bool saveModel();

        // Load model weights from desired location
        bool loadModel();
        
        // Train desired model
        void modelFit();

        // Trained model predict on provided input data
        void modelPredict();

        // Show model summary by printing it on std::cout
        void modelSummary();

    private:
        std::unique_ptr<std::vector<Perceptron>> mLayers; 
};