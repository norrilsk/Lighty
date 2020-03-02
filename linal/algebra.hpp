//
// Created by norrilsk on 17.02.20.
//

#ifndef LIGHTY_ALGEBRA_HPP
#define LIGHTY_ALGEBRA_HPP
#include"thensor.hpp"
namespace linal
{
  namespace experimental
  {
    template<typename T>
    thensor<T,2> unroll_kernel(const thensor<T,2> &kernel);
    template<typename T>
    thensor<T,2> unroll_image(const thensor<T,2> &img, int k_rows, int k_cols, int stride);
    template<typename T>
    thensor<T,2> padd_image(const thensor<T,2> &img, int padding, T padding_val = T());
    template<typename T>
    thensor<T,2> conv2d(const thensor<T,2> &src, const thensor<T,2> &kernel, int stride = 1, int padding=0, T padding_val = T());
  }
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
              const T * rhs_row = rhs + k * N;
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
      assert(padding >= 0);
      assert(stride > 0);
      assert(kernel.shape()[0] == src.shape()[0]);
      const int k_rows = kernel.shape()[2], k_cols = kernel.shape()[3];
      const int d_rows = src.shape()[1] + 2 * padding, d_cols = src.shape()[2] + 2 * padding;
      const int d_channels = kernel.shape()[1];
      const int s_channels = src.shape()[0];
      thensor<T,3> dst({d_channels,( d_rows - k_rows + stride) / stride, (d_cols - k_cols + stride) / stride});
      zero_set(dst);
      for (int i = 0 ; i < s_channels; i++)
      {
          thensor<T,2> single_channel = src[i];
          thensor<T,3> channel_kernels = kernel[i];
          for (int j = 0 ; j < d_channels; j++)
          {
              dst[j] += conv2d(single_channel,channel_kernels[j],stride,padding,padding_val);
          }
      }
      
      return dst;
  }
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel,int stride , int padding, T padding_val )
  {
	  thensor<T, 2> mat;
	  assert(padding >= 0);
	  assert(stride > 0);
      if (padding == 0)
      {
          mat = src;
      }
      else
      {
          const int rows = src.shape()[0] + 2 * padding;
          const int cols = src.shape()[1] + 2 * padding;
          mat = thensor<T, 2>({rows, cols});
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
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T,2> dst({( s_rows - k_rows + stride) / stride, (s_cols - k_cols + stride) / stride});
      
      for (int m = 0; m < s_rows - k_rows + 1; m+=stride)
      {
          for (int n = 0; n < s_cols - k_cols + 1; n+=stride)
          {
              T tmp = T();
              for (int i = 0 ; i < k_rows; i++)
              {
                  for(int j = 0; j < k_cols; j++)
                  {
                      tmp += mat[m + i][n + j] * kernel[i][j];
                  }
              }
              dst[m / stride][n / stride] = tmp;
          }
      }
      return dst;
  }
  
  
  template<typename T>
  thensor<T,2> experimental::padd_image(const thensor<T,2> &img, int padding, T padding_val)
  {
      if (padding <= 0)
      {
          return img.copy();
      }
      const int src_rows = img.shape()[0];
      const int src_cols = img.shape()[1];
      const int rows = src_rows + 2 * padding;
      const int cols = src_cols + 2 * padding;
      
      thensor<T, 2> res({ rows, cols});
      
      T* res_d = res.data();
      const T* src_d = img.data();
      for (int j = 0; j < cols; j++)
      {
          res_d[j] = padding_val;
      }
      for (int i = 1; i < padding; i++)
      {
          std::copy(res_d , res_d + cols, res_d + i * cols);
      }
      for (int i = rows - padding; i < rows; i++)
      {
          std::copy(res_d , res_d + cols, res_d + i * cols);
      }
      for (int i = padding; i < rows - padding; i++)
      {
          std::copy(res_d, res_d + padding,res_d + i * cols );
          std::copy(src_d + (i - padding) * src_cols, src_d + (i - padding + 1) * src_cols
              , res_d + i * cols  + padding);
          std::copy(res_d, res_d + padding,res_d + (i + 1) * cols - padding);
      }
      return res;
  }
  
  template<typename T>
  thensor<T,2> experimental::unroll_kernel(const thensor<T,2> &kernel)
  {
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      thensor<T, 2> res({k_rows * k_cols, 1});
      std::copy(kernel.data(),kernel.data() + kernel.size(), res.data());
      return res;
  }
  
  template<typename T>
  thensor<T,2> experimental::unroll_image(const thensor<T,2> &img, int k_rows, int k_cols, int stride){
      const int i_rows = img.shape()[0], i_cols = img.shape()[1];
      const int depth = k_cols * k_rows;
      const int r = ( i_rows - k_rows + stride) / stride, c = (i_cols - k_cols + stride) / stride;
      const int rows = r * c;
      
      thensor<T, 2> res({rows, depth});
      T* d_ptr = res.data();
      const T* src_row = img.data();
      
      for (int m = 0; m < i_rows - k_rows + 1; m+=stride)
      {
          const T* src_col = src_row;
          for (int n = 0; n < i_cols - k_cols + 1; n += stride)
          {
              const T* src =  src_col;
              for (int i = 0 ; i < k_rows; i++)
              {
                  std::copy(src, src + k_cols, d_ptr);
                  d_ptr += k_cols;
                  src += i_rows;
              }
              src_col += stride;
          }
          src_row += i_rows * stride;
      }
      return res;
  }
  
  template<typename T>
  thensor<T, 2> experimental::conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int stride, int padding, T padding_val)
  {
      assert(padding >= 0);
      assert(stride > 0);
      thensor<T, 2> mat;
      if (padding != 0)
          mat = experimental::padd_image(src, padding, padding_val);
      else
          mat = src;
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T, 2> filter = experimental::unroll_kernel(kernel);
      thensor<T, 2> img = experimental::unroll_image(mat, k_rows,k_cols,stride);
      
      return matmul(img, filter);
  }
  
}
#endif //LIGHTY_ALGEBRA_HPP
