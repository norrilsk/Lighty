#include <iostream>
#include "linal/thensor.hpp"
int main()
{
    //Dense1D<float,float> D1(2,3);
    {
        linal::thensor<float, 2> t({3, 3});
        t[0][0] = 1.0;
        t[1][1] = 2.0;
        t[2][2] = 3.0;
        std::cout << t << std::endl;
        linal::thensor<float, 2> t1({3, 3});
        t1[0][0] = 3.0f;
        t1[1][1] = -4.0f;
        t1[2][2] = 7.0f;
        t1[2][0] = 5.0f;
        std::cout << t1 << std::endl;
        linal::thensor<float, 2> t2 = t1 + t;
        t2 -= 5 * t2;
        if (t2 != t1)
        {
            std::cout << t2.dot(t2) << std::endl;
        }
        std::cout << t2 << "\n" << std::endl;
    }
    {
        linal::thensor<float, 1> t(3);
        t[0] = 1.0;
        t[1] = 2.0;
        t[2] = 3.0;
        std::cout << t << std::endl;
        linal::thensor<float, 1> t1(3);
        t1[0] = 3.0f;
        t1[1] = -4.0f;
        t1[2] = 7.0f;
        std::cout << t1 << std::endl;
        linal::thensor<float, 1> t2 = t1 + t;
        t2 -= 5 * t2;
        if (t2 != t1)
        {
            std::cout << t2.dot(t2) << std::endl;
        }
    }
    std::cout << "Hello, World!" << std::endl;
 
    return 0;
}