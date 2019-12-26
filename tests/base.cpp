//
// Created by norrilsk on 30.11.19.
//
#include"base.hpp"
#include "../linal/thensor.hpp"
#include "../Layers.hpp"
#include "../Losses.hpp"
#include "base.hpp"
#include "../Nets.hpp"
#include<fstream>
namespace test
{
  void dense_layer(bool verbose)
  {
      std ::cout<< "DENSE LAYER TEST" <<std::endl;
      std ::cout<< "9 point interpolation 2 layer dense-net" <<std::endl;
      fmat y({9,1}); //y = x*2 + 1
      y[0][0] = 15;
      y[1][0] = 10;
      y[2][0] = 5;
      y[3][0] = 2;
      y[4][0] = 1;
      y[5][0] = 2;
      y[6][0] = 5;
      y[7][0] = 10;
      y[8][0] = 15;
    
      fmat x({9,1});
      x[0][0] = -4;
      x[1][0] = -3;
      x[2][0] = -2;
      x[3][0] = -1;
      x[4][0] = 0;
      x[5][0] = 1;
      x[6][0] = 2;
      x[7][0] = 3;
      x[8][0] = 4;
      
      int N = 1000;
      Dense1D<float, float> f1(1, N);
      Relu1D<float, float> relu;
      Dense1D<float, float> f2(N, 1);
      MSE<float, float> mse;
      
      //fvec res(1), delta2(1), delta1(N), delta0(1);
      for (int ep = 0; ep < 10000; ++ep)
      {
          float tmse = 0;
          auto&& hidden  = f1.forward(x);
          //std::cout <<"hid: " << hidden << std::endl;
          auto&& hidden2 = relu.forward(hidden);
          //std::cout <<"rel: " << hidden << std::endl;
          auto&& hidden3  = f2.forward(hidden2);
          //std::cout <<"res: " << res << std::endl;
          tmse += mse(dynamic_cast<const fmat &>(hidden3), y);
          
          auto&& delta2 = mse.grad(dynamic_cast<const fmat &>(hidden3), y);
          auto&& delta1 = f2.backward(delta2);
          auto&& delta0 = relu.backward(delta1);
          f1.backward(delta0);
          
          if (verbose)
              std::cout << tmse / 9 << "\n";
      }
      if (verbose)
        std::cout << std::endl << "______________" << "\n";
      
      fmat xt({8,1});
      xt[0][0] = -3.5f;
      xt[1][0] = -2.5f;
      xt[2][0] = -1.5f;
      xt[3][0] = -0.5f;
      xt[4][0] = 0.5f;
      xt[5][0] = 1.5f;
      xt[6][0] = 2.5f;
      xt[7][0] = 3.5f;
    
      fmat yt({8,1});
      yt[0][0] = xt[0][0] * xt[0][0] + 1;
      yt[1][0] = xt[1][0] * xt[1][0] + 1;
      yt[2][0] = xt[2][0] * xt[2][0] + 1;
      yt[3][0] = xt[3][0] * xt[3][0] + 1;
      yt[4][0] = xt[4][0] * xt[4][0] + 1;
      yt[5][0] = xt[5][0] * xt[5][0] + 1;
      yt[6][0] = xt[6][0] * xt[6][0] + 1;
      yt[7][0] = xt[7][0] * xt[7][0] + 1;
    
      {
          auto &&hidden = f1.forward(x);
          //std::cout <<"hid: " << hidden << std::endl;
          auto &&hidden2 = relu.forward(hidden);
          //std::cout <<"rel: " << hidden << std::endl;
          fmat res = dynamic_cast<const fmat &>(f2.forward(hidden2)).copy();
          //std::cout <<"res: " << res << std::endl;
          if (verbose)
          {
              for (int i = 0; i < 9; ++i)
              {
                  std::cout << res[i][0] << "|" << y[i] << "; ";
              }
          }
          std::cout << "\n Train mse:  " << mse(res, y) << std::endl;
      }
      {
          auto &&hidden = f1.forward(xt);
          //std::cout <<"hid: " << hidden << std::endl;
          auto &&hidden2 = relu.forward(hidden);
          //std::cout <<"rel: " << hidden << std::endl;
          fmat res = dynamic_cast<const fmat &>(f2.forward(hidden2)).copy();
          //std::cout <<"res: " << res << std::endl;
          if (verbose)
          {
              for (int i = 0; i < 8; ++i)
              {
                  std::cout << res[i][0] << "|" << yt[i] << "; ";
              }
          }
          std::cout << "\n Test mse:  " << mse(res, yt) << std::endl;
      }
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
  void sequention_net(bool verbose)
  {
      Sequential net;
      net.addDense1D<float,float>(2, 10);
      net.addRelu1D<float,float>();
      net.addDense1D<float,float>(10, 15);
      net.addRelu1D<float,float>();
      net.addDense1D<float,float>(15, 20);
      net.addRelu1D<float,float>();
      net.addDense1D<float,float>(20, 1);
      float step = 0.04;
      fmat data({2500,2});
      fmat labels({2500,1});
      int i = 0;
      for ( float x = step/2 -1; x <1 ; x+=step)
      {
          for ( float y = step/2 -1; y <1 ; y+=step)
          {
              data[i][0] = x;
              data[i][1] = y;
			  labels[i][0] = ((x*x + y*y) < 1.f ) ? 1.f : 0.f;
			  //std::cout << "x = " << x << "  y = " << y << "  x ^2+ y^2 = " << (x * x + y * y) << " label = " << labels[i][0] << "\n";
			 // std::cout << "x = " << data[i][0] << "  y = " << data[i][1] << "  x ^2+ y^2 = " << (x * x + y * y) << " label = " << labels[i][0] << "\n";
              i++;
          }
      }
      
      step =0.01;
      fmat data_test({40000,2});
      fmat labels_test({40000,1});
      i = 0;
      for ( float x = step/2 -1; x <1 ; x+=step)
      {
          for ( float y = step/2 -1; y <1 ; y+=step)
          {
              data_test[i][0] = x;
              data_test[i][1] = y;
              labels_test[i][0] = ((x*x + y*y) < 1.f ) ? 1.f : 0.f;
              i++;
          }
      }
      
      MSE<float,float> mse;
      net.train<fmat,fmat, MSE<float,float>,true>(data,labels,mse,10,20);
      fmat ans = net.predict_batch<fmat,fmat>(data_test);
      std::cout<<mse(ans,labels_test);
      std::ofstream out_train;
      out_train.open("../train.txt");
      //out_train<< data;
      out_train<< " \n";
	  out_train << labels;// -net.predict_batch<fmat, fmat>(data);
      
      std::ofstream out;
      out.open("../predict.txt");
     // out<< data_test;
      out<< " \n";
      out << labels_test - ans;
  }
}