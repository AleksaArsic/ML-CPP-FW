#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

struct Perceptron
{
    // y = wx + b
    Eigen::MatrixXd w;
    Eigen::MatrixXd x;
    double b;
};

class Layer
{
    public:
        Layer(const uint8_t perceptronNo) : mLayerId(0), mPerceptronNo(perceptronNo), mLearnableCoeffs(0)
        {
            this->mInstances++;
            this->mLayer = std::make_unique<Perceptron[]>(this->mPerceptronNo);
        }

        Layer(Layer& l) : mLayer(std::move(l.mLayer)), mLayerId(l.mLayerId), mPerceptronNo(l.mPerceptronNo), mLearnableCoeffs(l.mLearnableCoeffs)
        {
            l.mLayer = nullptr;
            l.mLayerId = 0;
            l.mPerceptronNo = 0;
            l.mLearnableCoeffs = 0;
        }

        ~Layer()
        {
            mLayer.release();
        }
    
        uint8_t get_mPerceptronNo() const noexcept { return this->mPerceptronNo; }
        uint8_t get_mLayerId() const noexcept { return this->mLayerId; }
        uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }
        Perceptron& get_Perceptron(uint8_t perceptronId) const noexcept { return this->mLayer[perceptronId]; }

        void set_mLayerId(uint8_t id) { this->mLayerId = id; }
        void set_mLearnableCoeffs(uint32_t coeffsNo) { this->mLearnableCoeffs = coeffsNo; }
    private:
        inline static uint8_t mInstances = 0;
        std::unique_ptr<Perceptron[]> mLayer; // Number of Perceptrons is known in advance thus it is easier to dinamically allocate required number of Perceptrons
        uint8_t mLayerId;
        uint8_t mPerceptronNo;
        uint32_t mLearnableCoeffs; 
};

class Model
{
    public:
        Model() : mLearnableCoeffs(0), mLayersNo(0), mIsCompiled(false) { }

        ~Model() = default;

        // Add new layer to the NN Model
        bool addLayer(Layer layer);

        // Compile model with added layers, optimizer, loss function and metrics 
        bool compileModel();

        // Save model weights to desired location
        bool saveModel(std::string modelPath = "./model.csv") const;

        // Load model weights from desired location
        bool loadModel();
        
        // Train desired model
        void modelFit();

        // Trained model predict on provided input data
        void modelPredict() const;

        // Show model summary by printing it on std::cout
        void modelSummary() const;

        uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }
        uint8_t get_mLayersNo() const noexcept { return this->mLayersNo; }
    private:
        std::vector<std::unique_ptr<Layer>> mLayers; // Number of Layers is not known in advance thus, std::vector is more suitable for storing Layers
        uint32_t mLearnableCoeffs;
        uint8_t mLayersNo;
        bool mIsCompiled;
};