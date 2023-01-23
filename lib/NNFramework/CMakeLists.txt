# Project information
cmake_minimum_required(VERSION 3.15)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(NNFramework VERSION 1.0.0)

include_directories(inc)

file(GLOB_RECURSE _HDRS "inc/*.hpp")
file(GLOB_RECURSE _SRCS "src/*.[hc]pp")

add_library(NNFramework ${_HDRS} ${_SRCS}
            Activations
            Loss
            Metrics
            Model
            )

target_include_directories(NNFramework INTERFACE .. )
