#include <iostream>
#include <vector>
#include "NNFramework/inc/Eigen/Dense"
#include "NNFramework/Activations"
#include "NNFramework/Loss"
#include "NNFramework/Model"

#include <cmath>

//#include <matplot/matplot.h>
#if 0
void plotTest()
{
    using namespace matplot;
    std::vector<double> x = linspace(0, 2 * pi);
    std::vector<double> y = transform(x, [](auto x) { return sin(x); });

    plot(x, y, "-o");
    hold(on);
    plot(x, transform(y, [](auto y) { return -y; }), "--xr");
    plot(x, transform(x, [](auto x) { return x / pi - 1.; }), "-:gs");
    plot({1.0, 0.7, 0.4, 0.0, -0.4, -0.7, -1}, "k");

    show();
}
#endif

int main()
{
    Model model;

    model.addLayer(Layer(3)); // or -> model.addLayer(Layer(3, Activations::ActivationType<Activations::InputActivation>()));
    model.addLayer(Layer(4, Activations::ActivationType<Activations::Relu>()));
    model.addLayer(Layer(3, Activations::ActivationType<Activations::LeakyRelu>()));

    model.compileModel(Loss::LossType<Loss::MeanAbsoluteError>());
    model.modelSummary();

    Eigen::VectorXd predicted = Eigen::VectorXd(4);
    Eigen::VectorXd expected = Eigen::VectorXd(4);

    predicted << 0.8,
                0.7,
                0.3,
                0.9;

    expected << 1,
                0,
                1,
                1;

    Loss::BinaryCrossEntropy bce;
    std::cout << bce(expected, predicted) << std::endl;

    //plotTest();

#if 0
    Eigen::MatrixXd m = Eigen::MatrixXd::Random(3, 2);
    Eigen::VectorXd layerOutputZ = Eigen::VectorXd(2);
    layerOutputZ << 3.23,
                    -3.23;

    std::cout << m << std::endl;
    std::cout << m * layerOutputZ << std::endl;

    std::cout << "Sigmoid: " << layerOutputZ.unaryExpr(Activations::Sigmoid()) << std::endl;
#endif

    // unfinished
    //model.saveModel();

    return 0;
}