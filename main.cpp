#include <iostream>
#include "linal/thensor.hpp"
#include "Layers.hpp"
#include "Losses.hpp"
#include <chrono>
int main()
{
    //Dense1D<float,float> D1(2,3);
//    {
//        linal::thensor<float, 2> t({3, 3});
//        t[0][0] = 1.0;
//        t[1][1] = 2.0;
//        t[2][2] = 3.0;
//        std::cout << t << std::endl;
//        linal::thensor<float, 2> t1({3, 3});
//        t1[0][0] = 3.0f;
//        t1[1][1] = -4.0f;
//        t1[2][2] = 7.0f;
//        t1[2][0] = 5.0f;
//        std::cout << t1 << std::endl;
//        linal::thensor<float, 2> t2 = t1 + t;
//        t2 -= 5 * t2;
//        if (t2 != t1)
//        {
//            std::cout << t2.dot(t2) << std::endl;
//        }
//        std::cout << t2 << "\n" << std::endl;
//    }
//    {
//        linal::thensor<float, 1> t(3);
//        t[0] = 1.0;
//        t[1] = 2.0;
//        t[2] = 3.0;
//        std::cout << t << std::endl;
//        linal::thensor<float, 1> t1(3);
//        t1[0] = 3.0f;
//        t1[1] = -4.0f;
//        t1[2] = 7.0f;
//        std::cout << t1 << std::endl;
//        linal::thensor<float, 1> t2 = t1 + t;
//        t2 -= 5 * t2;
//        if (t2 != t1)
//        {
//            std::cout << t2.dot(t2) << std::endl;
//        }
//    }
//    linal::thensor<float,2> M1({3,2});
//    linal::thensor<float,1> V1(2);
//    M1[0][0]=1;
//    M1[0][1]=2;
//    M1[1][0]=3;
//    M1[1][1]=4;
//    M1[2][0]=5;
//    M1[2][1]=6;
//    V1[0] =2;
//    V1[1] = 3;
//
//    std::cout <<M1<<'\n'<<std::endl;
//
//    std::cout << M1*V1 <<'\n' <<std::endl;
//    std::cout << "Hello, World!" << std::endl;
    std::vector<fvec>y(9, fvec(1)); //y = x*2 + 1
    y[0][0] = 15;
    y[1][0] = 10;
    y[2][0] = 5;
    y[3][0] = 2;
    y[4][0] = 1;
    y[5][0] = 2;
    y[6][0] = 5;
    y[7][0] = 10;
    y[8][0] = 15;
    
    std::vector<fvec>x(9, fvec(1));
    x[0][0] = -4;
    x[1][0] = -3;
    x[2][0] = -2;
    x[3][0] = -1;
    x[4][0] = 0;
    x[5][0] = 1;
    x[6][0] = 2;
    x[7][0] = 3;
    x[8][0] = 4;
    
    int N = 10;
    Dense1D<float, float> f1(1, N);
    Relu1D<float, float> relu;
    Dense1D<float, float> f2(N, 1);
    MSE<float, float> mse;
    
  
    fvec hidden(N), res(1), delta2(1), delta1(N), delta0(1);
    for (int ep = 0; ep <1000; ++ep){
        float tmse = 0;
        for(int i = 0; i < 9; ++i){
            f1.forward(x[i], hidden);
            //std::cout <<"hid: " << hidden << std::endl;
            relu.forward(hidden, hidden);
            //std::cout <<"rel: " << hidden << std::endl;
            f2.forward(hidden, res);
            //std::cout <<"res: " << res << std::endl;
            tmse += mse(res, y[i]);
            
            delta2 = mse.grad(res, y[i]);
            f2.backward(delta2, delta1);
            relu.backward(delta1, delta1);
            f1.backward(delta1, delta0);
        }
        std::cout << tmse/ 9 << "\n";
    }
    
    std::cout << std::endl << "______________" <<"\n";
    
    std::vector<fvec>xt(8, fvec(1));
    xt[0][0] = -3.5f;
    xt[1][0] = -2.5f;
    xt[2][0] = -1.5f;
    xt[3][0] = -0.5f;
    xt[4][0] = 0.5f;
    xt[5][0] = 1.5f;
    xt[6][0] = 2.5f;
    xt[7][0] = 3.5f;
    
    
    std::vector<fvec>yt(8, fvec(1));
    yt[0][0] = xt[0][0]*xt[0][0] + 1;
    yt[1][0] = xt[1][0]*xt[1][0] + 1;
    yt[2][0] = xt[2][0]*xt[2][0] + 1;
    yt[3][0] = xt[3][0]*xt[3][0] + 1;
    yt[4][0] = xt[4][0]*xt[4][0] + 1;
    yt[5][0] = xt[5][0]*xt[5][0] + 1;
    yt[6][0] = xt[6][0]*xt[6][0] + 1;
    yt[7][0] = xt[7][0]*xt[7][0] + 1;
    
    float tmse = 0;
    
    for(int i = 0; i < 9; ++i)
    {
        f1.forward(x[i], hidden);
        relu.forward(hidden, hidden);
        f2.forward(hidden, res);
        std::cout << res << "|" << y[i] <<"; ";
        tmse += mse(res, y[i]);
    }
    std::cout <<"\n"<< tmse/ 9 << std::endl;
    tmse = 0;
    for(int i = 0; i < 8; ++i)
    {
        f1.forward(xt[i], hidden);
        relu.forward(hidden, hidden);
        f2.forward(hidden, res);
        std::cout << res << "|" << yt[i] <<"; ";
        tmse += mse(res, yt[i]);
    }
    std::cout <<"\n"<< tmse/ 8 << std::endl;
    return 0;
}