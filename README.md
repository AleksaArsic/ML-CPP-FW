# ML-CPP-FW
Simple machine learning framework implemented using C++20 programming language.

<a name="toc"></a>
## 1. Table of contents
1. [ Table of contents ](#toc)
2. [ Introduction ](#intro)
3. [ Repository structure ](#struct)
4. [ Clone the project ](#clone)
5. [ Prerequisites ](#prereq)
6. [ Build the project ](#build)
7. [ Run the project ](#run)
8. [ Include NNFramework in CMake project ](#cmakeinclude)
9. [ Example usecase ](#cheatsheet)
10. [ List of supported Layer and Model Configuration parameters ](#modelconfig)
11. [ Headers description ](#headerdesc)

<a name="intro"></a>
## 2. Introduction
This repository contains test application for simple machine learning framework implemented using C++20 programming language. The idea is to have a modular framework that is easly exapnded with more complex and advanced machine learning principals.  
Currently the Neural Network Framework **supports** several functionalities:
* Construct feed forward neural network model
* Configure hyper-parameters of the neural network model
* Train configured NN model
* Predict on trained NN model
* Present the information of the constructed NN model to the end user

**Unsuported** features that are likely to be implemented in the future:
* Save trained NN model localy 
* Load saved NN model 
* Continue training of the loaded NN model
* Add more optimizers (AdaDelta, Adam, etc.)
* Add more activation functions (softmax, tanh, ... )
* Add more layer types (Convolutive, Flatten, ... )

It is worth to mention that all of the mathematical background is implemented from scratch. That includes Feedforward algorithm, Backpropagation algorithm, Losses, Metrics and Optimizers. As for the linear algebra operations Neural Network Framework comes with built-in functionality provided trough [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page). Eigen library is built as a part of NN Framework and serves as a linear algebra "backend".

Neural Network Framework is buildable as a static library and can easly be included into other projects trough CMake build system infrastructure.

This repository provides test application that is covering complete workflow - design, configure, train and predict - on the constructed model. Test application is located under ./src/main.cpp path and incorporates NNFramework as it's core part from ./lib/ directory.  
***Neural Network Framework itself is located under ./lib/NNFramework directory.***

<a name="struct"></a>
## 3. Repository structure

* data - contains input data used for training of the neural network
* inc - contains include files for the test application
* lib
  *  ***NNFramework*** - root for the actual Neural Network Framework
  * matplotplusplus - containes external matplotplusplus library used for ploting graphs
* src - containes the main.cpp of the test application
* tools - contains third party tools used for the project build as well as external libraries (nothing from this directory is included in the build of the project)

<a name="clone"></a>
## 4. Clone the project
``` sh
$ git clone git@github.com:AleksaArsic/ML-CPP-FW.git
```

<a name="prereq"></a>
## 5. Prerequisites

As this project is developed on Windows OS, for building and running the project several prerequisites need to be satisfied:
* Installation of msys2
* Installation of dependencies inside msys2 mingw64 environment
* Setting environment variables

### Installation of msys2
The project comes with the msys2 version that was used for development of the Neural Network Framework. After cloning the project installation executable is provided inside of ./tools/windows/msys2-x86_64-20221216. Simply extract the .zip file and install it on desired location.

### Installation of dependencies inside msys2 mingw64 environment
After successfully installing msys2 tooling open MSYS2 MINGW64 terminal from start menu and run the following command to install all neccessary dependancies needed for building g++20 projects:
``` sh
$ pacman -S --needed base-devel mingw-w64-x86_64-toolchain
```

*Note: If asked which mingw tools to install, choose all (19) packages*

Also, install cmake for mingw64 with the command:
``` sh
pacman -S mingw-w64-x86_64-cmake
```

After the command is executed run the following command to install gnuplot needed for the external library matplotplusplus:
``` sh
$ pacman -S mingw-w64-x86_64-gnuplot
```

**Close MSYS2 MINGW64 terminal and proceed to setup of environment variables**

*Note: if you encounter problems with key-ring during any of the previous steps, this command should fix it:*
``` sh
$ pacman -S msys2-keyring 
```

### Setting environment variables
Edit your PATH environment variables to include paths to the msys2 mingw64 toolchain e.g:
```
C:\msys64\mingw64  
C:\msys64\mingw64\bin  
C:\msys64\mingw64\lib  
```

With out this step MSYS2 MINGW64 environment won't be able to find g++20 compiler, make, etc.

***After setting up environment variables close all instances of SYS2 MINGW64 terminals for changes to be applied.***

<a name="build"></a>
## 6. Build the project

1. Inside of MSYS2 MING64 environment position yourself to the project root.
2. Run ./build.sh shell script with the following arguments:
``` sh
$ ./build.sh --build       # for build without clean procedure
```
``` sh
$ ./build.sh --clean-build # for clean build
```

Either of these commands will trigger the build and in the project root ./build directory will be created with the NNFramework_test.exe executable.

<a name="run"></a>
## 7. Run the project

After successfull build of the project, test executable can be ran as simple as:
``` sh
$ ./build/NNFramework_test.exe
```

<a name="cmakeinclude"></a>
## 8. Include NNFramework in CMake project

Including NNFramework in CMake project is as simple as:
``` cmake
target_link_libraries(NNFramework_test PUBLIC NNFramework) # link NNFramework library 
```

Minimal version of cmake is: **3.15** with CXX_STANDARD 20

<a name="cheatsheet"></a>
## 9. Example usecase

Following chapters contain "cheatsheet" on how to use the Neural Network Framework and is representing complete workflow.  
Complete code snippet can be found in ./src/main.cpp

### Include NNFramework

Including of the Neural Network framework alongside with all of its functionality can be done as simple as:

```cpp
#include "NNFramework/NNFramework
```

### Declare Model object

Declaration of Neural Network model is done using:

```cpp
Model::Model model;
```
At this stage the model cannot be trained as it internal architecture and configuration is not defined. This will only declare object of class Model with 0 layers and no configuration.

### Add Layers to the Model object

Adding layers to neural network model is done by invoking Model.addLayer() method:

```cpp
model.addLayer(Layers::Dense(1)); // or -> model.addLayer(Layers::Dense(3, Activations::ActivationType<Activations::InputActivation>()));
model.addLayer(Layers::Dense(20, Activations::ActivationType<Activations::LeakyRelu>()));
model.addLayer(Layers::Dense(1, Activations::ActivationType<Activations::Sigmoid>()));
```

At this stage we added 3 layers to our neural network model with one input, one hidden and one output layer with the following configuration:
* Input layer with one artificial neuron
* Hidden layer with 20 artificial neurons and LeakyRelu activation function
* Output layer with one artificial neuron and Sigmoid activation function

*For supported layers and Model configuration parameters refer to chapter 9.*

### Add Configuration of the Neural Network Model

Loss, Metrics, Optimizer and additional configuration of the Model object is configured trough instance of Model::ModelConfiguration::ModelConfiguration object:

```cpp
Model::ModelConfiguration::ModelConfiguration modelConfig { Loss::LossType<Loss::MeanSquaredError>(), 
                                                            Metrics::MetricsType<Metrics::MeanSquaredError>(), 
                                                            Optimizers::OptimizersType<Optimizers::GradientDescent>(),
                                                            Model::ModelConfiguration::ShuffleData { true, 5 } };
```

We can also change the values of the relevant configuration fields:

```cpp
modelConfig.mOptimizerPtr->learningRate = 0.1;
modelConfig.mShuffleData->mShuffleStep = 10;
```

*For supported layers and Model configuration parameters refer to chapter 9.*

### Compile Model object

After we defined ModelConfiguration we can now bind it to our neural network model trough Model.modelCompile() method:

```cpp
model.compileModel(modelConfig);
```

Now, our neural network model is ready for training (with assumption that we are in possesion of desired training dataset).

### [Optional] Show model summary

If we want to log model summary with all the information regarding individual layers, loss, metric, optimizer, learnable parameters, etc. we can invoke Model.modelSummary() method that will log all mentioned data to standard output:

```cpp
model.modelSummary();
```

This way we can be aware of the model architecture and configuration.

### Train the model

Training of the model is invoked trough Model.fit() method:

```cpp
model.modelFit(inputData, expectedData, numberOfEpochs);
```

### [Optional] Retrieve Model.fit() history

We can also retrieve Model.fit() history buffers:

```cpp
auto modelHistory = model.get_mModelHistory();
```

### Predict output on the trained model

Predicting the output on the trained model can be invoked trough Model.predict() method:

```cpp
Eigen::MatrixXd predictedData = model.modelPredict(inputData);
```

As a result we are geting Eigen::MatrixXd of predicted data.

<a name="modelconfig"></a>
## 10. List of supported Layer and Model Configuration parameters

The following configuration parameters are currently supported by Neural Network Framework:
* [**Activations and activation derivatives**](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Activations.hpp)
    * InputActivation (Pass trough)
    * Sigmoid
    * Relu
    * LeakyRelu
* [**Losses and loss derivatives**](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Loss.hpp)
    * MeanSquaredError
    * MeanAbsoluteError
    * BinaryCrossEntropy
* [**Metrics**](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Metrics.hpp)
    * ClassificationAccuracy
    * MeanSquaredError
    * MeanAbsoluteError
* [**Optimizers**](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Optimizers.hpp)
    * GradientDescent

<a name="headerdesc"></a>
## 10. Headers description
As mentioned earlier, root of the NNFramework is located under ./lib/NNFramework. For the user to have a little more knowledge of the framework in the following section brief descriptions of each header file in respect to NNFramework root are provided:
* [./NNFramework](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/NNFramework) - header file whose purpose is to enable easy inclusion of the NNFramework into the end user project
* [./inc/Common/Common.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Common/Common.hpp) - header file with common code used by NNFramework
* [./inc/Core/Activations.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Activations.hpp) - holds activation functors and their derivations
* [./inc/Core/Layers.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Layers.hpp) - holds Layer classes
* [./inc/Core/Loss.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Loss.hpp) - holds loss functors and their derivations
* [./inc/Core/Metrics.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Metrics.hpp) - holds metric functors
* [./inc/Core/Model.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Model.hpp) - holds Model class definition
* [./inc/Core/ModelConfiguration](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/ModelConfiguration.hpp) - holds MoldeConfiguration class used for defining Model configuration parameters
* [./inc/Core/Optimizers.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/Optimizers.hpp) - holds optimizer functors
* [./inc/Core/WeightInitializer.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Core/WeightInitializer.hpp) - holds WeightInitializer class used for initialization of the Layer weights at the Model.compile() time
* [./inc/Eigen/*](https://github.com/AleksaArsic/ML-CPP-FW/tree/main/lib/NNFramework/inc/Eigen) - holds linear algebra "backend" library of the NNFramework
* [./inc/Utilities/DataHandler.hpp](https://github.com/AleksaArsic/ML-CPP-FW/blob/main/lib/NNFramework/inc/Utilities/DataHandler.hpp) - holds DataHandler class that is used for data manipulation (normalization, denormalization, data shuffle) 

Each of the header file serves as an entry point for potential development and is structured in a way that is development friendly for future implementations and extensions of NNFramework.

*For better description of each Class refere to the source code.*
