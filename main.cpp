#include <iostream>


#include "tests/base.hpp"
#include <chrono>
int main()
{
    test::dense_layer();
    test::matmul3x3(true);
    return 0;
}