//
// Created by norrilsk on 17.02.20.
//

#ifndef LIGHTY_ALGEBRA_HPP
#define LIGHTY_ALGEBRA_HPP
#include"thensor.hpp"
namespace linal
{
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 2> &left, const thensor<T, 2> &right);
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 1> &left, const thensor<T, 1> &right);
  template<typename T>
  thensor<T,3> conv2d(const thensor<T,3> &src, const thensor<T,4> &kernel, int stride = 1, int padding=0, T padding_val = T());
  template<typename T>
  thensor<T,2> conv2d(const thensor<T,2> &src, const thensor<T,2> &kernel, int stride = 1, int padding=0, T padding_val = T());
  template<typename T>
  thensor<T,2> matmul (const thensor<T,1>& left, const thensor<T,1>& right)
  {
      int shp0 = right.size();
      int shp1 = left.size();
      thensor<T,2> res({shp0, shp1});
      T* dst = res.data();
      const T* l = left.data();
      const T* r = right.data();
      
      for(int i = 0; i < shp0; i++)
      {
          for(int j = 0; j < shp1; j++, dst++)
          {
              *dst = l[j]*r[i];
          }
      }
      
      return res;
  }
  
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 2> &left, const thensor<T, 2> &right)
  {
      const T* lhs = left.data();
      const T* rhs = right.data();
      int M = left.shape()[0];
      int N = right.shape()[1];
      int K = left.shape()[1];
      assert(right.shape()[0] == K);
      thensor<T,2> res({M,N});
      T* dst = res.data();
      for (int i = 0; i < M; i++)
      {
          T * dst_row = dst + i * N;
          const T * lhs_row = lhs + i * K;
          for (int j = 0; j < N; j++)
              dst_row[j] = T();
          for (int k = 0; k < K; k++)
          {
              const T * rhs_row = rhs  + k * N;
              T lhs_val = lhs_row[k];
              for (int j = 0; j < N; ++j)
                  dst_row[j] += lhs_val * rhs_row[j];
          }
      }
      return res;
  }
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int stride , int padding, T padding_val )
  {
      assert(0);
      return thensor<T, 3>();
  }
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel,int stride , int padding, T padding_val )
  {
	  thensor<T, 2> mat;
	  assert(padding >= 0);
	  assert(stride > 0);
      if (padding !=0)
      {
		  const int rows = src.shape()[0] + 2 * padding;
		  const int cols = src.shape()[1] + 2 * padding;
		  mat = thensor<T, 2>({ rows, cols});
		  for (int i = 0; i < padding; i++)
		  {
			  for (int j = 0; j < cols; j++)
			  {
				  mat[i][j] = padding_val;
			  }
		  }
		  for (int i = padding; i < rows - padding; i++)
		  {
			  for (int j = 0; j < padding; j++)
			  {
				  mat[i][j] = padding_val;
			  }
			  for (int j = padding; j < cols - padding; j++)
			  {
				  mat[i][j] = src[i - padding][j - padding];
			  }
			  for (int j = cols - padding; j < cols; j++)
			  {
				  mat[i][j] = padding_val;
			  }
		  }
		  for (int i = rows - padding; i < rows; i++)
		  {
			  for (int j = 0; j < cols; j++)
			  {
				  mat[i][j] = padding_val;
			  }
		  }
	  }
	  else
	  {
		  mat = src;
	  }
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T,2> dst({( s_rows - k_rows + stride)/stride, (s_cols - k_cols + stride)/stride});
      
      for (int m = 0; m < s_rows - k_rows + 1; m+=stride)
      {
          for (int n = 0; n < s_cols - k_cols + 1; n+=stride)
          {
              T tmp = T();
              for (int i = 0 ; i < k_rows; i++)
              {
                  for(int j = 0; j < k_cols; j++)
                  {
                      tmp+=mat[m+i][n+j]*kernel[i][j];
                  }
              }
              dst[m/stride][n/stride] = tmp;
          }
      }
      return dst;
  }
  
}
#endif //LIGHTY_ALGEBRA_HPP