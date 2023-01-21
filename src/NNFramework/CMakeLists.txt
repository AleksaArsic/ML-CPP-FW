# Project information
cmake_minimum_required(VERSION 3.15)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(NNFramework VERSION 1.0.0)

include_directories(inc)
add_library(NNFramework src/Model.cpp)
