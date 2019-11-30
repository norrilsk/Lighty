//
// Created by norrilsk on 30.11.19.
//
#include"base.hpp"
#include "../linal/thensor.hpp"
#include "../Layers.hpp"
#include "../Losses.hpp"
#include "base.hpp"

namespace test
{
  void dense_layer(bool verbose)
  {
      std ::cout<< "DENSE LAYER TEST" <<std::endl;
      std ::cout<< "9 point interpolation 2 layer dense-net" <<std::endl;
      std::vector<fvec> y(9, fvec({1})); //y = x*2 + 1
      y[0][0] = 15;
      y[1][0] = 10;
      y[2][0] = 5;
      y[3][0] = 2;
      y[4][0] = 1;
      y[5][0] = 2;
      y[6][0] = 5;
      y[7][0] = 10;
      y[8][0] = 15;
      
      std::vector<fvec> x(9, fvec({1}));
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
      for (int ep = 0; ep < 1000; ++ep)
      {
          float tmse = 0;
          for (int i = 0; i < 9; ++i)
          {
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
          if (verbose)
              std::cout << tmse / 9 << "\n";
      }
      if (verbose)
        std::cout << std::endl << "______________" << "\n";
      
      std::vector<fvec> xt(8, fvec(1));
      xt[0][0] = -3.5f;
      xt[1][0] = -2.5f;
      xt[2][0] = -1.5f;
      xt[3][0] = -0.5f;
      xt[4][0] = 0.5f;
      xt[5][0] = 1.5f;
      xt[6][0] = 2.5f;
      xt[7][0] = 3.5f;
      
      std::vector<fvec> yt(8, fvec(1));
      yt[0][0] = xt[0][0] * xt[0][0] + 1;
      yt[1][0] = xt[1][0] * xt[1][0] + 1;
      yt[2][0] = xt[2][0] * xt[2][0] + 1;
      yt[3][0] = xt[3][0] * xt[3][0] + 1;
      yt[4][0] = xt[4][0] * xt[4][0] + 1;
      yt[5][0] = xt[5][0] * xt[5][0] + 1;
      yt[6][0] = xt[6][0] * xt[6][0] + 1;
      yt[7][0] = xt[7][0] * xt[7][0] + 1;
      
      float tmse = 0;
      
      for (int i = 0; i < 9; ++i)
      {
          f1.forward(x[i], hidden);
          relu.forward(hidden, hidden);
          f2.forward(hidden, res);
          if (verbose)
              std::cout << res << "|" << y[i] << "; ";
          tmse += mse(res, y[i]);
      }
      
      std::cout << "\n Test mse:  " << tmse / 9 << std::endl;
      tmse = 0;
      for (int i = 0; i < 8; ++i)
      {
          f1.forward(xt[i], hidden);
          relu.forward(hidden, hidden);
          f2.forward(hidden, res);
          if (verbose)
            std::cout << res << "|" << yt[i] << "; ";
          tmse += mse(res, yt[i]);
      }
      std::cout << "\n Train mse: " << tmse / 8 << std::endl;
  }
  void matmul3x3(bool verbose)
  {
      std::cout <<"MATMUL TEST 3x3 int" << std::endl;
      int l[3][3] = {{5,6,-1},{8,12,22},{11,-1,3024}};
      int m[3][3] = {{22,65,11},{-5,-88,-93},{75,0,42}};
      int r[3][3] = {{5,-203,-545},{1766,-536,-104},{227047,803,127222}};
      linal::thensor<int,2> L,M,R_true,R_pred;
      L.wrap(&l[0][0],{3,3});
      M.wrap(&m[0][0],{3,3});
      R_true.wrap(&r[0][0],{3,3});
      R_pred = linal::matmul(L,M);
      if (verbose)
      {
          std::cout << "\nleft:\n" << L
                    << "\nright:\n" << M
                    << "\nres:\n" << R_pred
                    << "\ncorrect:\n" << R_true
                    <<"\n" <<std::endl;
      }
      if (R_true == R_pred)
      {
          std::cout << "OK" << std::endl;
      }
      else
      {
          std::cout <<"FAILED" <<std::endl;
      }
      
  }
}