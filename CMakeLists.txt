cmake_minimum_required(VERSION 3.13)
project(Lighty)

set(CMAKE_CXX_STANDARD 17)

add_executable(Lighty main.cpp tests/base.cpp Nets.hpp Optimizers.hpp linal/algebra.hpp)