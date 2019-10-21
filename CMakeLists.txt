cmake_minimum_required(VERSION 3.13)
project(Lighty)

set(CMAKE_CXX_STANDARD 17)

add_executable(Lighty main.cpp Layers.hpp linal/thensor.hpp linal/thensor.hpp)