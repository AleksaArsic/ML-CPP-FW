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

<a name="intro"></a>
## 2. Introduction
This repository contains test application for simple machine learning framework implemented using C++20 programming language. The idea is to have a modular framework that is easly exapnded with more complex and advanced machine learning principals.  
Currently the Neural Network Framework **supports** several functionalities:
* Construct feed forward neural network model
* Configure hyper-parameters of the neural network model
* Train configured NN model
* Predict on trained NN model
* Present the information of the constructed NN model to the end user

**Unsuported** features that are likly to be implemented in the future:
* Save trained NN model localy 
* Load saved NN model 
* Continue training of the loaded NN model

It is worth to mention that all of the mathemaical background is implemented from scratch. That includes Feedforward algorithm, Backpropagation algorithm, Losses, Metrics and Optimizers. As for the linear algebra operations Neural Network Framework comes with built-in functionality provided trough [Eigen library](https://eigen.tuxfamily.org/index.php?title=Main_Page). Eigen library is built as a part of NN Framework and serves as a linear algebra "back-end".

Neural Network Framework is buildable as a static library and can easly be included into other projects trough CMake build system infrastructure.

<a name="struct"></a>
## 3. Repository structure

* data - contains input data used for training of the neural network
* inc - contains include files for the test application
* lib
  *  ***NNFramework*** - root for the actuall Neural Network Framework
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

After the command is executed run the following command to install gnuplot needed for the external library matplotplusplus:
``` sh
$ pacman -S mingw-w64-x86_64-gnuplot
```

**Close MSYS2 MINGW64 terminal and proceed to setup of environment variables**

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
