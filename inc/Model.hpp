#include <memory>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

class Layer
{
    public:
        Layer(const uint8_t perceptronNo) : mLayerId(0), mPerceptronNo(perceptronNo), mLearnableCoeffs(0)
        {
            this->mInstances++;
            this->mLayerWeights = std::make_shared<Eigen::MatrixXd>();
            this->mLayerX = std::make_shared<Eigen::MatrixXd>();
            this->mLayerBias = std::make_shared<Eigen::MatrixXd>();

        }

        Layer(Layer& l) :  mLayerId(l.mLayerId), mPerceptronNo(l.mPerceptronNo), mLearnableCoeffs(l.mLearnableCoeffs)
        {
            mLayerWeights = std::move(l.mLayerWeights);
            mLayerX = std::move(l.mLayerX);
            mLayerBias = std::move(l.mLayerBias);
            l.mLayerId = 0;
            l.mPerceptronNo = 0;
            l.mLearnableCoeffs = 0;
        }

        uint8_t get_mPerceptronNo() const noexcept { return this->mPerceptronNo; }
        uint8_t get_mLayerId() const noexcept { return this->mLayerId; }
        uint32_t get_mLearnableCoeffs() const noexcept { return this->mLearnableCoeffs; }

        std::shared_ptr<Eigen::MatrixXd> get_mLayerWeights() const noexcept { return this->mLayerWeights; }
        std::shared_ptr<Eigen::MatrixXd> get_mLayerX() const noexcept { return this->mLayerX; }
        std::shared_ptr<Eigen::MatrixXd> get_mLayerBias() const noexcept { return this->mLayerBias; }

        void set_mLayerId(uint8_t id) { this->mLayerId = id; }
        void set_mLearnableCoeffs(uint32_t coeffsNo) { this->mLearnableCoeffs = coeffsNo; }
    private:
        inline static uint8_t mInstances = 0;

        std::shared_ptr<Eigen::MatrixXd> mLayerWeights;
        std::shared_ptr<Eigen::MatrixXd> mLayerX;
        std::shared_ptr<Eigen::MatrixXd> mLayerBias;
        
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