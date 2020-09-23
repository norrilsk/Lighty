//
// Created by norrilsk on 30.11.19.
//
#include"base.hpp"
#include "../linal/thensor.hpp"
#include "../Layers.hpp"
#include "../Losses.hpp"
#include "../linal/algebra.hpp"
#include "base.hpp"
#include "../Nets.hpp"
#include <fstream>
#include <chrono>
#include <ctime>
#include<cmath>
#include <iostream>

namespace test
{
  template<typename T , int32_t _dim>
  void fill(linal::thensor<T,_dim> &th, std::function<T()> initializer);
  
  template<typename T, int32_t _dim>
  void fill(linal::thensor<T, _dim> &th, std::function<T()> initializer)
  {
	T* ptr = th.data();
	for (int32_t i = 0; i < th.size(); i++)
	{
		ptr[i] = initializer();
	}
  }
  
  bool dense_net(bool verbose)
    {
        std::cout<< "DENSE NET TEST" <<std::endl;
        std::cout<< "COPY OF DENSE LAYER TEST WITH NET API"<<std::endl;
        fmat y({ 9,1 }); //y = x*2 + 1
        y[0][0] = 15;
        y[1][0] = 10;
        y[2][0] = 5;
        y[3][0] = 2;
        y[4][0] = 1;
        y[5][0] = 2;
        y[6][0] = 5;
        y[7][0] = 10;
        y[8][0] = 15;
    
        fmat x({ 9,1 });
        x[0][0] = -4;
        x[1][0] = -3;
        x[2][0] = -2;
        x[3][0] = -1;
        x[4][0] = 0;
        x[5][0] = 1;
        x[6][0] = 2;
        x[7][0] = 3;
        x[8][0] = 4;
        
        int32_t N = 1000;
        Sequential net;
        net.addDense1D<float,float>(1,N);
        net.addRelu1D<float,float>();
        net.addDense1D<float,float>(N,1);
        MSE<float, float> mse;
        
        net.train<fmat, fmat, MSE<float, float> >(x, y, 1, 10000,verbose);
        
    
        fmat xt({ 8,1 });
        xt[0][0] = -3.5f;
        xt[1][0] = -2.5f;
        xt[2][0] = -1.5f;
        xt[3][0] = -0.5f;
        xt[4][0] = 0.5f;
        xt[5][0] = 1.5f;
        xt[6][0] = 2.5f;
        xt[7][0] = 3.5f;
    
        fmat yt({ 8,1 });
        yt[0][0] = xt[0][0] * xt[0][0] + 1;
        yt[1][0] = xt[1][0] * xt[1][0] + 1;
        yt[2][0] = xt[2][0] * xt[2][0] + 1;
        yt[3][0] = xt[3][0] * xt[3][0] + 1;
        yt[4][0] = xt[4][0] * xt[4][0] + 1;
        yt[5][0] = xt[5][0] * xt[5][0] + 1;
        yt[6][0] = xt[6][0] * xt[6][0] + 1;
        yt[7][0] = xt[7][0] * xt[7][0] + 1;
        
		
    
        {
			fmat res = net.predict_batch<fmat, fmat>(x);
            if (verbose)
            {
                for (int32_t i = 0; i < 9; ++i)
                {
                    std::cout << res[i][0] << "|" << y[i] << "; ";
                }
            }
            std::cout << "\n Train mse:  " << mse(res, y) << std::endl;
        }
        {
			fmat res = net.predict_batch<fmat, fmat>(xt);
            if (verbose)
            {
                for (int32_t i = 0; i < 8; ++i)
                {
                    std::cout << res[i][0] << "|" << yt[i] << "; ";
                }
            }
            std::cout << "\n Test mse:  " << mse(res, yt) << std::endl;
        }
        return true;
    }
	bool dense_layer(bool verbose)
	{
		std::cout << "DENSE LAYER TEST" << std::endl;
		std::cout << "9 point interpolation 2 layer dense-net" << std::endl;
		fmat y({ 9,1 }); //y = x*2 + 1
		y[0][0] = 15;
		y[1][0] = 10;
		y[2][0] = 5;
		y[3][0] = 2;
		y[4][0] = 1;
		y[5][0] = 2;
		y[6][0] = 5;
		y[7][0] = 10;
		y[8][0] = 15;

		fmat x({ 9,1 });
		x[0][0] = -4;
		x[1][0] = -3;
		x[2][0] = -2;
		x[3][0] = -1;
		x[4][0] = 0;
		x[5][0] = 1;
		x[6][0] = 2;
		x[7][0] = 3;
		x[8][0] = 4;

		int32_t N = 1000;
		Dense1D<float, float> f1(1, N);
		Relu1D<float, float> relu;
		Dense1D<float, float> f2(N, 1);
		MSE<float, float> mse;

		//fvec res(1), delta2(1), delta1(N), delta0(1);
		for (int32_t ep = 0; ep < 10000; ++ep)
		{
			float tmse = 0;
			auto&& hidden = f1.forward(x);
			//std::cout <<"hid: " << hidden << std::endl;
			auto&& hidden2 = relu.forward(hidden);
			//std::cout <<"rel: " << hidden << std::endl;
			auto&& hidden3 = f2.forward(hidden2);
			//std::cout <<"res: " << res << std::endl;
			tmse += mse(dynamic_cast<const fmat&>(hidden3), y);

			auto&& delta2 = mse.grad(dynamic_cast<const fmat&>(hidden3), y);
			auto&& delta1 = f2.backward(delta2);
			auto&& delta0 = relu.backward(delta1);
			f1.backward(delta0);

			if (verbose)
				std::cout << tmse / 9 << "\n";
		}
		if (verbose)
			std::cout << std::endl << "______________" << "\n";

		fmat xt({ 8,1 });
		xt[0][0] = -3.5f;
		xt[1][0] = -2.5f;
		xt[2][0] = -1.5f;
		xt[3][0] = -0.5f;
		xt[4][0] = 0.5f;
		xt[5][0] = 1.5f;
		xt[6][0] = 2.5f;
		xt[7][0] = 3.5f;

		fmat yt({ 8,1 });
		yt[0][0] = xt[0][0] * xt[0][0] + 1;
		yt[1][0] = xt[1][0] * xt[1][0] + 1;
		yt[2][0] = xt[2][0] * xt[2][0] + 1;
		yt[3][0] = xt[3][0] * xt[3][0] + 1;
		yt[4][0] = xt[4][0] * xt[4][0] + 1;
		yt[5][0] = xt[5][0] * xt[5][0] + 1;
		yt[6][0] = xt[6][0] * xt[6][0] + 1;
		yt[7][0] = xt[7][0] * xt[7][0] + 1;

		{
			auto&& hidden = f1.forward(x);
			//std::cout <<"hid: " << hidden << std::endl;
			auto&& hidden2 = relu.forward(hidden);
			//std::cout <<"rel: " << hidden << std::endl;
			fmat res = dynamic_cast<const fmat&>(f2.forward(hidden2)).copy();
			//std::cout <<"res: " << res << std::endl;
			if (verbose)
			{
				for (int32_t i = 0; i < 9; ++i)
				{
					std::cout << res[i][0] << "|" << y[i] << "; ";
				}
			}
			std::cout << "\n Train mse:  " << mse(res, y) << std::endl;
		}
		{
			auto&& hidden = f1.forward(xt);
			//std::cout <<"hid: " << hidden << std::endl;
			auto&& hidden2 = relu.forward(hidden);
			//std::cout <<"rel: " << hidden << std::endl;
			fmat res = dynamic_cast<const fmat&>(f2.forward(hidden2)).copy();
			//std::cout <<"res: " << res << std::endl;
			if (verbose)
			{
				for (int32_t i = 0; i < 8; ++i)
				{
					std::cout << res[i][0] << "|" << yt[i] << "; ";
				}
			}
			std::cout << "\n Test mse:  " << mse(res, yt) << std::endl;
		}
		return true;
	}
	bool matmul3x3(bool verbose)
	{
		std::cout << "MATMUL TEST 3x3 int32_t" << std::endl;
		int32_t l[3][3] = { {5,6,-1},{8,12,22},{11,-1,3024} };
		int32_t m[3][3] = { {22,65,11},{-5,-88,-93},{75,0,42} };
		int32_t r[3][3] = { {5,-203,-545},{1766,-536,-104},{227047,803,127222} };
		linal::thensor<int32_t, 2> L, M, R_true, R_pred;
		L.wrap(&l[0][0], { 3,3 });
		M.wrap(&m[0][0], { 3,3 });
		R_true.wrap(&r[0][0], { 3,3 });
		R_pred = linal::matmul(L, M);
		if (verbose)
		{
			std::cout << "\nleft:\n" << L
				<< "\nright:\n" << M
				<< "\nres:\n" << R_pred
				<< "\ncorrect:\n" << R_true
				<< "\n" << std::endl;
		}
		if (R_true == R_pred)
		{
			std::cout << "OK" << std::endl;
			return true;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
			return false;
		}

	}
	
    bool dense_net_sin(bool verbose)
	{
    	fmat x({100,1}), y({100,1}),
    		 xt({100,1}), yt({100,1});
		static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 42 );
		std::uniform_real_distribution<float> distribution{-3.141592f,3.141592f};
    	for (int32_t i = 0 ; i < 100 ; i++)
		{
    		x[i][0] = distribution(generator);
    		y[i][0] = std::sin(x[i][0]);
			xt[i][0] = distribution(generator);
			yt[i][0] = std::sin(xt[i][0]);
		}
		Sequential net;
    	int32_t N1 = 50;
    	int32_t N2 = 500;
    	int32_t N3 = 100;
    	net.addDense1D<float,float>(1,N1);
    	net.addRelu1D<float,float>();
    	net.addDense1D<float,float>(N1,N2);
    	net.addRelu1D<float,float>();
		net.addDense1D<float,float>(N2,N3);
		net.addSigmoid1D<float,float>();
    	net.addDense1D<float,float>(N3,1);
    	
    	net.set_optimizers(optim::OPTIMIZER_ADAM,2e-2);
    	net.train<fmat,fmat,MSE<float,float> >(x,y,10,3000,verbose);
		MSE<float,float> mse;
		{
			fmat res = net.predict_batch<fmat, fmat>(x);
			if (verbose)
			{
				for (int32_t i = 0; i < 10; ++i)
				{
					std::cout << res[i][0] << "|" << y[i] << "; ";
				}
			}
			std::cout << "\n Train mse:  " << mse(res, y) << std::endl;
		}
		{
			fmat res = net.predict_batch<fmat, fmat>(xt);
			if (verbose)
			{
				for (int32_t i = 0; i < 10; ++i)
				{
					std::cout << res[i][0] << "|" << yt[i] << "; ";
				}
			}
			std::cout << "\n Test mse:  " << mse(res, yt) << std::endl;
		}
		return true;
	}
	
	bool sequention_net(bool verbose)
	{
		int32_t Npoints = 10000;
		fmat x({Npoints,2}), y({Npoints,1}),
			xt({Npoints,2}), yt({Npoints,1});
		static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 42 );
		std::uniform_real_distribution<float> distribution{-1.f,1.f};
		for (int32_t i = 0 ; i < Npoints ; i++)
		{
			float x1 = distribution(generator),
			      x2 = distribution(generator);
			x[i][0] = x1;
			x[i][1] = x2;
			y[i][0] = ((x1*x1 + x2*x2) < 1.f) ? 1.f : 0.f;
			
			x1 = distribution(generator),
			x2 = distribution(generator);
			xt[i][0] = x1;
			xt[i][1] = x2;
			yt[i][0] = ((x1*x1 + x2*x2) < 1.f) ? 1.f : 0.f;
		}
		Sequential net;
		int32_t N1 = 40;
		int32_t N2 = 30;
		int32_t N3 = 20;
		int32_t N4 = 30;
		int32_t N5 = 40;
		net.addDense1D<float,float>(2,N1);
		net.addRelu1D<float,float>();
		net.addDense1D<float,float>(N1,N2);
        net.addRelu1D<float,float>();
		net.addDense1D<float,float>(N2,N3);
		net.addRelu1D<float,float>();
		net.addDense1D<float,float>(N3,N4);
		net.addRelu1D<float,float>();
		net.addDense1D<float,float>(N4,N5);
		net.addSigmoid1D<float,float>();
		net.addDense1D<float,float>(N5,1);
		//net.addSigmoid1D<float,float>();
		net.set_optimizers(optim::OPTIMIZER_ADAM,1e-3);
		net.train<fmat,fmat,MSE<float,float> >(x, y, 100, 300, verbose);
		MSE<float,float> mse;
		
		fmat ans = net.predict_batch<fmat, fmat>(x);
		if (verbose)
		{
			for (int32_t i = 0 ; i < 10; i++)
			{
					std::cout<<"("<< x[i][0]<<';'<<x[i][1]<<") " << ans[i] << " | " << y[i][0] <<";  ";
			}
			std::cout<<'\n';
			std::cout << "MSE train " << mse(ans, y) << "\n";
			
			ans = net.predict_batch<fmat, fmat>(xt);
			for (int32_t i = 0 ; i < 10; i++)
			{
				std::cout<<"("<< xt[i][0]<<';'<<xt[i][1]<<") " << ans[i] << " | " << yt[i][0] <<";  ";
			}
			std::cout<<'\n';
			std::cout << "MSE test " << mse(ans, yt) << std::endl;
			
		}
		return true;
	}

	bool thensor_test1(bool verbose)
	{
		std::cout << "THENSOR TEST 1\t";
		linal::thensor<int32_t,1> vec({3});
		linal::thensor<int32_t,2> mat({3,3});
		vec[0] = 0;
		vec[1] = 1;
		vec[2] = 2;

		mat[0][0] = 3;
		mat[0][1] = 4;
		mat[0][2] = 5;

		mat[1][0] = 6;
		mat[1][1] = 7;
		mat[1][2] = 8;

		mat[2][0] = 9;
		mat[2][1] = 10;
		mat[2][2] = 11;

		if (verbose)
		{
			std::cout << "\n\nvector\n" << vec << "\n\nmatrix\n" << mat;
		}

		vec = mat[0];

		vec[0] = 22;
		mat[0][1] = 33;

		if (verbose)
		{
			std::cout << "\n\nvec = mat[0] \n vec[0] = 22\nmat[0][1] = 33 ";
		}
		linal::thensor<int32_t, 1> vec_checker({ 3 });
		linal::thensor<int32_t, 2> mat_checker({ 3,3 });
		vec_checker[0] = 22;
		vec_checker[1] = 33;
		vec_checker[2] = 5;

		mat_checker[0][0] = 22;
		mat_checker[0][1] = 33;
		mat_checker[0][2] = 5;

		mat_checker[1][0] = 6;
		mat_checker[1][1] = 7;
		mat_checker[1][2] = 8;

		mat_checker[2][0] = 9;
		mat_checker[2][1] = 10;
		mat_checker[2][2] = 11;

		bool check = true;

		if (verbose)
		{
			std::cout << "\n\nafter test:\n\n\nvector : \n" << vec << "\n\nexpected\n" << vec_checker;
			std::cout << "\n\nmatrix : \n" << mat << "\n\nexpected\n" << mat_checker;
		}
		check &= mat == mat_checker;
		check &= vec == vec_checker;
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
	}

	bool thensor_test2(bool verbose)
	{
		std::cout << "THENSOR TEST 2\t";
		linal::thensor<int32_t, 1> vec({ 3 });
		linal::thensor<int32_t, 2> mat({ 3,3 });
		vec[0] = 0;
		vec[1] = 1;
		vec[2] = 2;

		mat[0][0] = 3;
		mat[0][1] = 4;
		mat[0][2] = 5;

		mat[1][0] = 6;
		mat[1][1] = 7;
		mat[1][2] = 8;

		mat[2][0] = 9;
		mat[2][1] = 10;
		mat[2][2] = 11;

		if (verbose)
		{
			std::cout << "\n\nvector\n" << vec << "\n\nmatrix\n" << mat;
		}

		vec = mat[0].copy();

		vec[0] = 22;
		mat[0][1] = 33;

		if (verbose)
		{
			std::cout << "\n\nvec = mat[0].copy() \n vec[0] = 22\nmat[0][1] = 33 ";
		}
		linal::thensor<int32_t, 1> vec_checker({ 3 });
		linal::thensor<int32_t, 2> mat_checker({ 3,3 });
		vec_checker[0] = 22;
		vec_checker[1] = 4;
		vec_checker[2] = 5;

		mat_checker[0][0] = 3;
		mat_checker[0][1] = 33;
		mat_checker[0][2] = 5;

		mat_checker[1][0] = 6;
		mat_checker[1][1] = 7;
		mat_checker[1][2] = 8;

		mat_checker[2][0] = 9;
		mat_checker[2][1] = 10;
		mat_checker[2][2] = 11;

		bool check = true;

		if (verbose)
		{
			std::cout << "\n\nafter test:\n\n\nvector : \n" << vec << "\n\nexpected\n" << vec_checker;
			std::cout << "\n\nmatrix : \n" << mat << "\n\nexpected\n" << mat_checker;
		}
		check &= mat == mat_checker;
		check &= vec == vec_checker;
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
	}

	bool thensor_test3(bool verbose)
	{
		std::cout << "THENSOR TEST 3\t";
		linal::thensor<int32_t, 1> vec({ 3 });
		linal::thensor<int32_t, 2> mat({ 3,3 });
		vec[0] = 0;
		vec[1] = 1;
		vec[2] = 2;

		mat[0][0] = 3;
		mat[0][1] = 4;
		mat[0][2] = 5;

		mat[1][0] = 6;
		mat[1][1] = 7;
		mat[1][2] = 8;

		mat[2][0] = 9;
		mat[2][1] = 10;
		mat[2][2] = 11;

		if (verbose)
		{
			std::cout << "\n\nvector\n" << vec << "\n\nmatrix\n" << mat;
		}

		mat[0] = vec;

		vec[0] = 22;
		mat[0][1] = 33;

		if (verbose)
		{
			std::cout << "\n\nmat[0] = vec\n vec[0] = 22\nmat[0][1] = 33 ";
		}
		linal::thensor<int32_t, 1> vec_checker({ 3 });
		linal::thensor<int32_t, 2> mat_checker({ 3,3 });
		vec_checker[0] = 22;
		vec_checker[1] = 1;
		vec_checker[2] = 2;

		mat_checker[0][0] = 0;
		mat_checker[0][1] = 33;
		mat_checker[0][2] = 2;

		mat_checker[1][0] = 6;
		mat_checker[1][1] = 7;
		mat_checker[1][2] = 8;

		mat_checker[2][0] = 9;
		mat_checker[2][1] = 10;
		mat_checker[2][2] = 11;

		bool check = true;

		if (verbose)
		{
			std::cout << "\n\nafter test:\n\n\nvector : \n" << vec << "\n\nexpected\n" << vec_checker;
			std::cout << "\n\nmatrix : \n" << mat << "\n\nexpected\n" << mat_checker;
		}
		check &= mat == mat_checker;
		check &= vec == vec_checker;
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
	}

	bool thensor_test4(bool verbose)
	{
		std::cout << "THENSOR TEST 4\t";
		linal::thensor<int32_t, 1> vec({ 3 });
		linal::thensor<int32_t, 2> mat({ 3,3 });
		vec[0] = 0;
		vec[1] = 1;
		vec[2] = 2;

		mat[0][0] = 3;
		mat[0][1] = 4;
		mat[0][2] = 5;

		mat[1][0] = 6;
		mat[1][1] = 7;
		mat[1][2] = 8;

		mat[2][0] = 9;
		mat[2][1] = 10;
		mat[2][2] = 11;

		if (verbose)
		{
			std::cout << "\n\nvector\n" << vec << "\n\nmatrix\n" << mat;
		}

		mat[0] = vec.copy();

		vec[0] = 22;
		mat[0][1] = 33;

		if (verbose)
		{
			std::cout << "\n\nmat[0] = vec.copy() \n vec[0] = 22\nmat[0][1] = 33 ";
		}
		linal::thensor<int32_t, 1> vec_checker({ 3 });
		linal::thensor<int32_t, 2> mat_checker({ 3,3 });
		vec_checker[0] = 22;
		vec_checker[1] = 1;
		vec_checker[2] = 2;

		mat_checker[0][0] = 0;
		mat_checker[0][1] = 33;
		mat_checker[0][2] = 2;

		mat_checker[1][0] = 6;
		mat_checker[1][1] = 7;
		mat_checker[1][2] = 8;

		mat_checker[2][0] = 9;
		mat_checker[2][1] = 10;
		mat_checker[2][2] = 11;

		bool check = true;

		if (verbose)
		{
			std::cout << "\n\nafter test:\n\n\nvector : \n" << vec << "\n\nexpected\n" << vec_checker;
			std::cout << "\n\nmatrix : \n" << mat << "\n\nexpected\n" << mat_checker;
		}
		check &= mat == mat_checker;
		check &= vec == vec_checker;
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
	}
  
    bool linal_conv2d(bool verbose )
    {
        std::cout << "CONVOLUTION TEST 1\t";
        linal::thensor<int32_t,2> mat({3,3});
        mat[0][0] = 3;
        mat[0][1] = 4;
        mat[0][2] = 5;
        
        mat[1][0] = 6;
        mat[1][1] = 7;
        mat[1][2] = 8;
        
        mat[2][0] = 9;
        mat[2][1] = 10;
        mat[2][2] = 11;
    
        linal::thensor<int32_t,2> kernel({2,2});
        
        kernel[0][0] = 1;
        kernel[0][1] = 2;
        kernel[1][0] = -1;
        kernel[1][1] = 3;
        

        linal::thensor<int32_t,2> checker({2,2});
        checker[0][0] = 26;
        checker[0][1] = 31;
        checker[1][0] = 41;
        checker[1][1] = 46;
        bool check = false;
        linal::thensor<int32_t,2> res = linal::conv2d(mat,kernel);
        check = res==checker;
        if (verbose)
        {
            std::cout << "\n\nafter test:\n\n\n result : \n" << res << "\n\nexpected\n" << checker<<"\n\n";
        }
        if (check)
        {
            std::cout << "OK" << std::endl;
        }
        else
        {
            std::cout << "FAILED" << std::endl;
        }
        return check;
        
    }

	bool linal_conv2d_2(bool verbose)
	{
		std::cout << "CONVOLUTION TEST 2\t";
		linal::thensor<int32_t, 2> mat({ 3,3 });
		mat[0][0] = 3;
		mat[0][1] = 4;
		mat[0][2] = 5;

		mat[1][0] = 6;
		mat[1][1] = 7;
		mat[1][2] = 8;

		mat[2][0] = 9;
		mat[2][1] = 10;
		mat[2][2] = 11;

		linal::thensor<int32_t, 2> kernel({ 2,2 });

		kernel[0][0] = 1;
		kernel[0][1] = 2;
		kernel[1][0] = -1;
		kernel[1][1] = 3;


		linal::thensor<int32_t, 2> checker({ 2,2 });
		checker[0][0] = 9;
		checker[0][1] = 11;
		checker[1][0] = 39;
		checker[1][1] = 46;
		bool check = false;
		linal::thensor<int32_t, 2> res = linal::conv2d(mat, kernel,2,1,0);
		check = res == checker;
		if (verbose)
		{
			std::cout << "\n\nafter test:\n\n\n result : \n" << res << "\n\nexpected\n" << checker << "\n\n";
		}
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;

	}
  
  bool linal_conv2d_3(bool verbose)
  {
	  std::cout << "CONVOLUTION TEST 3\t";
	  linal::thensor<int32_t, 3> mat({ 3,3,3});
	  mat[0][0][0] = 1;   mat[0][1][0] = 2;   mat[0][2][0] = 0;
	  mat[1][0][0] = 1;   mat[1][1][0] = 1;   mat[1][2][0] = 3;
	  mat[2][0][0] = 0;   mat[2][1][0] = 2;   mat[2][2][0] = 2;
	
	  mat[0][0][1] = 0;   mat[0][1][1] = 2;   mat[0][2][1] = 1;
	  mat[1][0][1] = 0;   mat[1][1][1] = 3;   mat[1][2][1] = 2;
	  mat[2][0][1] = 1;   mat[2][1][1] = 1;   mat[2][2][1] = 0;
	
	  mat[0][0][2] = 1;   mat[0][1][2] = 2;   mat[0][2][2] = 1;
	  mat[1][0][2] = 0;   mat[1][1][2] = 1;   mat[1][2][2] = 3;
	  mat[2][0][2] = 3;   mat[2][1][2] = 3;   mat[2][2][2] = 2;
	  
	  
	  //input channels,  rows , cols output_channels,
	  linal::thensor<int32_t, 4> kernel({ 3, 2, 2, 2 });
	  
	  kernel[0][0][0][0] = 1;   kernel[0][0][0][1] = 1;
	  kernel[0][0][1][0] = 2;   kernel[0][0][1][1] = 2;
	
	  kernel[0][1][0][0] = 1;   kernel[0][1][0][1] = 0;
	  kernel[0][1][1][0] = 0;   kernel[0][1][1][1] = 1;
	  
	  
	  kernel[1][0][0][0] = 1;   kernel[1][0][0][1] = 1;
	  kernel[1][0][1][0] = 1;   kernel[1][0][1][1] = 1;
	
	  kernel[1][1][0][0] = 2;   kernel[1][1][0][1] = 1;
	  kernel[1][1][1][0] = 2;   kernel[1][1][1][1] = 1;
	  
	
	  kernel[2][0][0][0] = 0;   kernel[2][0][0][1] = 1;
	  kernel[2][0][1][0] = 1;   kernel[2][0][1][1] = 0;
	
	  kernel[2][1][0][0] = 1;   kernel[2][1][0][1] = 2;
	  kernel[2][1][1][0] = 2;   kernel[2][1][1][1] = 0;
	  
	  linal::thensor<int32_t, 3> checker({ 2, 2, 2});
	  
	  checker[0][0][0] = 14;   checker[0][1][0] = 20;
	  checker[1][0][0] = 15;   checker[1][1][0] = 24;

	  checker[0][0][1] = 12;   checker[0][1][1] = 24;
	  checker[1][0][1] = 17;   checker[1][1][1] = 26;
	  bool check = false;
	  linal::thensor<int32_t, 3> res = linal::conv2d(mat, kernel);
	  check = res == checker;
	  if (verbose)
	  {
		  std::cout << "\n\nafter test:\n\n\n result : \n" << res << "\n\nexpected\n" << checker << "\n\n";
	  }
	  if (check)
	  {
		  std::cout << "OK" << std::endl;
	  }
	  else
	  {
		  std::cout << "FAILED" << std::endl;
	  }
	  return check;
  }
  bool linal_conv_unroll(bool verbose)
  {
	  std::cout << "UNROLL TEST\t";
	  linal::thensor<int32_t, 2> mat({ 3,3});
	  mat[0][0] = 1;   mat[0][1] = 2;   mat[0][2] = 0;
	  mat[1][0] = 1;   mat[1][1] = 1;   mat[1][2] = 3;
	  mat[2][0] = 0;   mat[2][1] = 2;   mat[2][2] = 2;
	
	  linal::thensor<int32_t, 2> checker({ 4,4});
	  checker[0][0] = 1;   checker[0][1] = 2;   checker[0][2] = 1;   checker[0][3] = 1;
	  checker[1][0] = 2;   checker[1][1] = 0;   checker[1][2] = 1;   checker[1][3] = 3;
	  checker[2][0] = 1;   checker[2][1] = 1;   checker[2][2] = 0;   checker[2][3] = 2;
	  checker[3][0] = 1;   checker[3][1] = 3;   checker[3][2] = 2;   checker[3][3] = 2;
	
	  bool check = false;
	  linal::thensor<int32_t, 2> res = linal::unroll_image( mat,2, 2, 1,1);
	  check = res == checker;
	  if (verbose)
	  {
		  std::cout << "\n\nafter test:\n\n\n result : \n" << res << "\n\nexpected\n" << checker << "\n\n";
	  }
	  if (check)
	  {
		  std::cout << "OK" << std::endl;
	  }
	  else
	  {
		  std::cout << "FAILED" << std::endl;
	  }
	  return check;
  }
  
  bool linal_conv2d_experimental(bool verbose)
  {
	    std::cout << "CONVOLUTION EXPERIMENTAL TEST 1\t";
  		linal::thensor<int32_t,2> src({50,50});
    	linal::thensor<int32_t,2> kernel({7,7});
    	linal::thensor<int32_t,2> res1, checker;
    	
    	std::default_random_engine gen(static_cast<unsigned long>(std::time(nullptr)));
    	std::uniform_int_distribution<int32_t> dist(-100,100);
    	fill<int32_t,2>(src, [&]{return dist(gen);});
    	fill<int32_t,2>(kernel, [&]{return dist(gen);});
    	auto start = std::chrono::high_resolution_clock::now();
    	checker = linal::depricated::conv2d(src,kernel,2,3,4);
    	auto p1 = std::chrono::high_resolution_clock::now();
    	res1 = linal::conv2d(src,kernel,2,3,4);
    	auto p2 = std::chrono::high_resolution_clock::now();
		bool check = checker == res1;
    	
    	if (verbose)
		{
    		std::cout << "base time : "
    		<< std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - start).count() * 1e-6
    		<< " ms\n";
    		std::cout << "experimental time : "
			<< std::chrono::duration_cast<std::chrono::nanoseconds>(p2 - p1).count() * 1e-6
			<< " ms\n";
    		
    		/*std::cout << "src:\n" <<src;
    		std::cout << "\n\nkernel:\n"<<kernel;
    		std::cout << "\n\nres:\n" << res1;
    		std::cout << "\n\ncontrol:\n " <<checker;*/
		}
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
  }
  
  bool linal_conv2d_2_experimental(bool verbose)
  {
	  std::cout << "CONVOLUTION EXPERIMENTAL TEST 2\t";
      linal::thensor<int32_t,3> src({128,128,10});
      linal::thensor<int32_t,4> kernel({10,20,3,3});
      linal::thensor<int32_t,3> res1, checker;
      
      std::default_random_engine gen(static_cast<unsigned long>(std::time(nullptr)));
      std::uniform_int_distribution<int32_t> dist(-100,100);
      fill<int32_t,3>(src, [&]{return dist(gen);});
      fill<int32_t,4>(kernel, [&]{return dist(gen);});
      auto start = std::chrono::high_resolution_clock::now();
      checker = linal::depricated::conv2d(src,kernel,2,3,4);
      auto p1 = std::chrono::high_resolution_clock::now();
      res1 = linal::conv2d(src,kernel,2,3,4);
      auto p2 = std::chrono::high_resolution_clock::now();
	  bool check = checker == res1;
      
      if (verbose)
      {
          std::cout << "base time : "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(p1 - start).count() * 1e-6
                    << " ms\n";
          std::cout << "experimental time : "
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(p2 - p1).count() * 1e-6
                    << " ms\n";
      }
      if (check)
      {
          std::cout << "OK" << std::endl;
      }
      else
      {
          std::cout << "FAILED" << std::endl;
      }
	  return check;
  }



  bool linal_unroll_and_back(bool verbose)
  {
	  std::cout << "LINAL UNROLL AND BACK  TEST\t";
	  bool check = false;
	  for (int32_t i = 0; i < 1000; i++)
	  {
		  std::default_random_engine gen(static_cast<unsigned long>(std::time(nullptr)));
		  std::uniform_int_distribution<int32_t> dist(-1000, 1000);
		  std::uniform_int_distribution<int32_t> stride_dist(1, 5);
		  std::uniform_int_distribution<int32_t> channel_dist(1, 10);
		  std::uniform_int_distribution<int32_t> kernel_dist(1, 7);
		  std::uniform_int_distribution<int32_t> mult_dist(1, 20);
		  int32_t ker_x = kernel_dist(gen);
		  int32_t ker_y = kernel_dist(gen);
		  int32_t stride_v = std::min(stride_dist(gen), ker_y);
		  int32_t stride_h = std::min(stride_dist(gen), ker_x);
		  int32_t channels = channel_dist(gen);
		  int32_t filters = channel_dist(gen);
		  int32_t mul1 = mult_dist(gen);
		  int32_t mul2 = mult_dist(gen);
		  std::vector<int32_t> kernel_shape = { channels, filters, ker_y, ker_x };
		  ithensor3 checker({ mul1 * stride_v + kernel_shape[2], mul2 * stride_h + kernel_shape[3] , kernel_shape[0] });

		  

		  fill<int32_t, 3>(checker, [&] {return dist(gen); });

		  check = (checker == linal::backward_unroll_image(
			  linal::unroll_image(checker, kernel_shape, stride_v, stride_h),
			  kernel_shape, checker.shape(), stride_v, stride_h, 0, 0));
		  if (!check)
		  {
			  if (verbose)
			  {
				  std::cout << "failed with params : \n";
				  std::cout << "stride_v = " << stride_v << "\nstride_h = " << stride_h
					  << "\nchannels = " << channels << "\nfilters = " << filters
					  << "\nker_x = " << ker_x << "\nker_y = " << ker_y
					  << "\nmul1 = " << mul1 << "\n mul2 = " << mul2 << std::endl;
				  break;
			  }
		  }
	  }
	  if (check)
	  {
		  std::cout << "OK" << std::endl;
	  }
	  else
	  {
		  std::cout << "FAILED" << std::endl;
	  }
	  return check;
  }

  bool linal_unroll_test1(bool verbose )
  {
	  std::cout << "LINAL UNROLL TEST1 \t";
	  int32_t stride_v = 2;
	  int32_t stride_h = 1;
	  int32_t channels = 3;
	  int32_t filters = 2;
	  int32_t ker_x = 1;
	  int32_t ker_y = 2;
	  int32_t mul1 = 1;
	  int32_t mul2 = 1;
	  std::vector<int32_t> kernel_shape = { channels, filters, ker_y, ker_x  };
	  ithensor3 checker({ mul1 * stride_v + kernel_shape[2], mul2 * stride_h + kernel_shape[3] , kernel_shape[0] });

	  bool check = false;
	  std::default_random_engine gen(static_cast<unsigned long>(std::time(nullptr)));
	  std::uniform_int_distribution<int32_t> dist(0, 5);
	  fill<int32_t, 3>(checker, [&] {return dist(gen); });

	  check = (checker == linal::backward_unroll_image(
		  linal::unroll_image(checker, kernel_shape, stride_v, stride_h),
		  kernel_shape, checker.shape(), stride_v, stride_h, 0, 0));

	  if (verbose)
	  {
		  std::cout << "checker:\n "<<checker << std::endl;
		  std::cout << "unrolled:\n" << linal::unroll_image(checker, kernel_shape, stride_v, stride_h) << std::endl;
		  std::cout << "res:\n" << linal::backward_unroll_image(
			  linal::unroll_image(checker, kernel_shape, stride_v, stride_h),
			  kernel_shape, checker.shape(), stride_v, stride_h, 0, 0) << std::endl;
	  }
	  if (check)
	  {
		  std::cout << "OK" << std::endl;
	  }
	  else
	  {
		  std::cout << "FAILED" << std::endl;
	  }
	  return check;
  }

  bool linal_conv2d_net(bool verbose)
  {
	  std::cout << "CONV2D NET TEST" << std::endl;
	  int32_t N_train = 30000;
	  int32_t N_test = 1000;
	  int32_t size = 20;
	  int32_t epochs = 100;
	  int32_t batch_size = 100;

	  /* generate 2D normal distributed data white pixels on black background 
	  * and try to predict disspersion */
	  std::default_random_engine gen(static_cast<unsigned long>(std::time(nullptr)));
	  std::uniform_real_distribution<float> random(0, 1);
	  auto normal_vec = [](int32_t size, float m, float sigm)
	  {
		  fvec x({ size });
		  for (int32_t i = 0; i < size; i++)
		  {
			  x[i] = std::exp(-(i - m) * (i - m) / (2 * sigm * sigm));
		  }
		  return x;
	  };
	  auto generate = [&normal_vec, &random, &gen](int32_t size, float x, float y, float sx, float sy)
	  {
		  fmat density = linal::matmul(linal::reshape<float, 1, 2>(normal_vec(size, y, sy), { size, 1 }),
			  linal::reshape<float, 1, 2>(normal_vec(size, x, sx), { 1, size }));
		  float * density_d = density.data();
		  for (int32_t i = 0; i < size; i++)
		  {
			  for (int32_t j = 0; j < size; j++)
			  {
				  *(density_d++) = static_cast<float>(random(gen) < *density_d);
			  }
		  }
		  return density;
	  };
	  auto generate_batch = [&generate, &gen, &random](int32_t bathch_size, int32_t size)
	  {
		  fthensor2 labels({ bathch_size, 2 });
		  fthensor3 res({bathch_size,size,size});
		  for (int32_t i = 0; i < bathch_size; i++)
		  {
			  
			  float x = random(gen) * size;
			  float y = random(gen) * size;
			  float sx = random(gen) * size / 2 + 1;
			  float sy = random(gen) * size / 2 + 1;
			  labels[i][0] = sx;
			  labels[i][1] = sy;
			  res[i] = generate(size, x, y, sx, sy);
		  }
		  return std::make_tuple(linal::reshape<float, 3, 4>(res, {bathch_size, size,size, 1} ), labels);
	  };
	  auto data_train = generate_batch(N_train, size);
	  fthensor4 x_train = std::get<0>(data_train);
	  fthensor2 y_train = std::get<1>(data_train);

	  auto data_test = generate_batch(N_test, size);
	  fthensor4 x_test = std::get<0>(data_test);
	  fthensor2 y_test = std::get<1>(data_test);
	  

	  Sequential net;

	  net.addConv2D<float, float>(1, std::tuple<int32_t, int32_t>{ 5, 5 }, 8, std::tuple<int32_t, int32_t>{ 3, 3 });
      net.addTanh2D<float, float>();
	  net.addConv2D<float, float>(8, std::tuple<int32_t, int32_t>{ 3, 3 }, 16);
	  net.addTanh2D<float, float>();
	  net.addConv2D<float, float>(16, std::tuple<int32_t, int32_t>{ 3, 3 }, 32);
	  net.addTanh2D<float, float>();
	  net.addFlattern4to2<float, float>();
	  net.addDense1D<float, float>(128, 2);

	  net.set_optimizers(optim::OPTIMIZER_ADAM, 2e-3);

	  MSE<float, float> mse;

	  net.train<fthensor4, fmat, MSE<float, float> >(x_train, y_train, batch_size, epochs, verbose);

	  {
		  fmat res = net.predict_batch<fthensor4, fmat>(x_train);
		  std::cout << "\n Train mse:  " << mse(res, y_train) << std::endl;
	  }
	  {
		  fmat res = net.predict_batch<fthensor4, fmat>(x_test);
		  std::cout << "\n Test mse:  " << mse(res, y_test) << std::endl;
	  }
	  return true;
  }

  bool dump_up(bool verbose)
  {
	  std::cout << "DUMP AND LOAD TEST TEST1 \t";
	  fmat x({ 100,1 }), y({ 100,1 }),
		  xt({ 100,1 }), yt({ 100,1 });
	  static std::default_random_engine generator(static_cast<unsigned>(time(nullptr)) + 42);
	  std::uniform_real_distribution<float> distribution{ -3.141592f,3.141592f };
	  for (int32_t i = 0; i < 100; i++)
	  {
		  x[i][0] = distribution(generator);
		  y[i][0] = std::sin(x[i][0]);
		  xt[i][0] = distribution(generator);
		  yt[i][0] = std::sin(xt[i][0]);
	  }
	  Sequential net;
	  int32_t N1 = 50;
	  int32_t N2 = 500;
	  int32_t N3 = 100;
	  net.addDense1D<float, float>(1, N1);
	  net.addRelu1D<float, float>();
	  net.addDense1D<float, float>(N1, N2);
	  net.addRelu1D<float, float>();
	  net.addDense1D<float, float>(N2, N3);
	  net.addSigmoid1D<float, float>();
	  net.addDense1D<float, float>(N3, 1);

	  net.set_optimizers(optim::OPTIMIZER_ADAM, 2e-3);
	  net.train<fmat, fmat, MSE<float, float> >(x, y, 10, 10, verbose);
	  std::ofstream file("TMPTEST1123747.net", std::ios::binary);
	  net.dump(file);
	  file.close();
	  MSE<float, float> mse;
	  
      fmat res_before_1 = net.predict_batch<fmat, fmat>(x);
	  if (verbose)
	  {
		  for (int32_t i = 0; i < 10; ++i)
		  {
			  std::cout << res_before_1[i][0] << "|" << y[i] << "; ";
		  }
		  std::cout << "\n Train mse before loading :  " << mse(res_before_1, y) << std::endl;
	  }
      
        
        
      fmat res_before_2 = net.predict_batch<fmat, fmat>(xt);
	  if (verbose)
	  {
		  for (int32_t i = 0; i < 10; ++i)
		  {
			  std::cout << res_before_2[i][0] << "|" << yt[i] << "; ";
		  }
		  std::cout << "\n Test mse: before loading  " << mse(res_before_2, yt) << std::endl;
	  }
     
    	  

	  Sequential net2;
	  std::ifstream ifile("TMPTEST1123747.net", std::ios::binary);
	  net2.load(ifile);
	  ifile.close();
	  std::remove("TMPTEST1123747.net");
	  
	  
	  
		fmat res1 = net2.predict_batch<fmat, fmat>(x);
		if (verbose)
		{
			for (int32_t i = 0; i < 10; ++i)
			{
				std::cout << res1[i][0] << "|" << y[i] << "; ";
			}
			std::cout << "\n Train mse after loading :  " << mse(res1, y) << std::endl;
		}
		
	  
	  
		fmat res2 = net2.predict_batch<fmat, fmat>(xt);
		if (verbose)
		{
			for (int32_t i = 0; i < 10; ++i)
			{
				std::cout << res2[i][0] << "|" << yt[i] << "; ";
			}
			std::cout << "\n Test mse: after loading  " << mse(res2, yt) << std::endl;
		}
		
		bool check = std::abs(mse(res_before_2, yt) - mse(res2, yt)) < 1e-3;
		if (check)
		{
			std::cout << "OK" << std::endl;
		}
		else
		{
			std::cout << "FAILED" << std::endl;
		}
		return check;
  }
}



