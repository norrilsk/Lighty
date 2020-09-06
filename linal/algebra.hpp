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
  }
  namespace depricated
  {
	  template<typename T>
	  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int stride = 1, int padding = 0, T padding_val = T());
	  template<typename T>
	  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int stride = 1, int padding = 0, T padding_val = T());
  }
  template<typename T>
  thensor<T, 2> unroll_kernel(const thensor<T, 2> &kernel);
  template<typename T>
  thensor<T, 2> unroll_image(const thensor<T, 2> &img, int k_rows, int k_cols, int stride_vert, int stride_hor);
  template<typename T>
  thensor<T, 2> padd_image(const thensor<T, 2> &img, int pad_rows, int pad_cols, T padding_val = T());
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, std::tuple<int, int> stride , std::tuple<int, int> padding, T padding_val = T());
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int stride = 1 , int padding = 0 , T padding_val = T());
  template<typename T>
  thensor<T, 2> unroll_kernel(const thensor<T, 4> &kernel);
  template<typename T>
  thensor<T, 2> unroll_image(const thensor<T, 3> &img, const std::vector<int> &new_shape, int stride_vert, int stride_hor);
  template<typename T>
  thensor<T, 3> padd_image(const thensor<T, 3> &img, int pad_rows, int pad_cols, T padding_val = T());
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, std::tuple<int, int> stride , std::tuple<int, int> padding, T padding_val = T());
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int  stride =  1, int padding =  0, T padding_val = T());

  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 2> &left, const thensor<T, 2> &right);
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 1> &left, const thensor<T, 1> &right);

  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int stride, int padding , T padding_val )
  {
	  return conv2d(src, kernel, { stride,stride }, { padding,padding }, padding_val);
  }

  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel,int  stride , int padding , T padding_val )
  {
	  return conv2d(src, kernel, { stride,stride }, { padding,padding }, padding_val);
  }



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
  thensor<T, 3> depricated::conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int stride , int padding, T padding_val )
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
              dst[j] += depricated::conv2d(single_channel,channel_kernels[j],stride,padding,padding_val);
          }
      }
      
      return dst;
  }
  template<typename T>
  thensor<T, 2> depricated::conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int stride, int padding, T padding_val)
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
  thensor<T,2> padd_image(const thensor<T,2> &img, int pad_rows, int pad_cols, T padding_val)
  {
      if (pad_rows <= 0 && pad_cols <= 0)
      {
          return img.copy();
      }
      const int src_rows = img.shape()[0];
      const int src_cols = img.shape()[1];
      const int rows = src_rows + 2 * pad_rows;
      const int cols = src_cols + 2 * pad_cols;
      
      thensor<T, 2> res({ rows, cols});
      
      T* res_d = res.data();
      const T* src_d = img.data();
	  if (pad_rows > 0)
	  {
		  for (int j = 0; j < cols; j++)
		  {
			  res_d[j] = padding_val;
		  }

		  for (int i = 1; i < pad_rows; i++)
		  {
			  std::copy(res_d, res_d + cols, res_d + i * cols);
		  }
		  for (int i = rows - pad_rows; i < rows; i++)
		  {
			  std::copy(res_d, res_d + cols, res_d + i * cols);
		  }
	  }
	  if (pad_cols > 0)
	  {
		  for (int j = 0; j < pad_cols; j++)
		  {
			  res_d[j + cols * pad_rows] = padding_val;
		  }
	  }
	  if (pad_rows < rows - pad_rows)
	  {
		  std::copy(src_d , src_d + src_cols
			  , res_d + pad_rows * cols + pad_cols);
		  std::copy(res_d, res_d + pad_cols, res_d + (pad_rows + 1) * cols - pad_cols);
	  }
      for (int i = pad_rows + 1 ; i < rows - pad_rows; i++)
      {
          std::copy(res_d, res_d + pad_cols,res_d + i * cols );
          std::copy(src_d + (i - pad_rows) * src_cols, src_d + (i - pad_rows + 1 ) * src_cols
              , res_d + i * cols  + pad_cols);
          std::copy(res_d, res_d + pad_cols,res_d + (i + 1) * cols - pad_cols);
      }
      return res;
  }
  
  template<typename T>
  thensor<T,2> unroll_kernel(const thensor<T,2> &kernel)
  {
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      thensor<T, 2> res({k_rows * k_cols, 1});
      std::copy(kernel.data(),kernel.data() + kernel.size(), res.data());
      return res;
  }
  

  template<typename T>
  thensor<T,2> unroll_image(const thensor<T,2> &img, int k_rows, int k_cols, int stride_vert, int stride_hor){
      const int i_rows = img.shape()[0], i_cols = img.shape()[1];
      const int depth = k_cols * k_rows;
      const int r = ( i_rows - k_rows + stride_vert) / stride_vert, c = (i_cols - k_cols + stride_hor) / stride_hor;
      const int rows = r * c;
      
      thensor<T, 2> res({rows, depth});
      T* d_ptr = res.data();
      const T* src_row = img.data();
      
      for (int m = 0; m < i_rows - k_rows + 1; m+= stride_vert)
      {
          const T* src_col = src_row;
          for (int n = 0; n < i_cols - k_cols + 1; n += stride_hor)
          {
              const T* src =  src_col;
              for (int i = 0 ; i < k_rows; i++)
              {
                  std::copy(src, src + k_cols, d_ptr);
                  d_ptr += k_cols;
                  src += i_rows;
              }
              src_col += stride_hor;
          }
          src_row += i_rows * stride_vert;
      }
      return res;
  }
  
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src,
                                     const thensor<T, 2> &kernel,
                                     std::tuple<int,int> stride,
									 std::tuple<int, int> padding,
                                     T padding_val)
  {
	  int pad_rows = std::get<0>(padding);
	  int pad_cols = std::get<1>(padding);
	  int stride_vert = std::get<0>(stride);
	  int stride_hor = std::get<1>(stride);
	  assert(pad_rows >= 0 && pad_cols >= 0);
	  assert(stride_vert > 0 && stride_hor > 0);
      thensor<T, 2> mat;
      if (pad_rows != 0 && pad_cols != 0)
          mat = padd_image(src, pad_rows, pad_cols, padding_val);
      else
          mat = src;
      const int k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T, 2> filter = unroll_kernel(kernel);
      thensor<T, 2> img = unroll_image(mat, k_rows,k_cols, stride_vert, stride_hor);
      
      return matmul(img, filter);
  }
  
  template<typename T>
  thensor<T,3> padd_image(const thensor<T,3> &img, int pad_rows, int pad_cols, T padding_val)
  {
      if (pad_rows <= 0 && pad_cols <= 0)
      {
          return img.copy();
      }
      const int channels = img.shape()[0];
      const int src_rows = img.shape()[1];
      const int src_cols = img.shape()[2];
      const int rows = src_rows + 2 * pad_rows;
      const int cols = src_cols + 2 * pad_cols;
      
      thensor<T, 3> res({channels, rows, cols});
      const int s_ch_size = src_rows * src_cols;
      const int d_ch_size = rows * cols;
      for (int k = 0; k < channels; k++)
      {
          T *res_d = res.data() + d_ch_size * k;
          const T *src_d = img.data() + s_ch_size * k;
          if (0 == k)
          {
			  if (pad_rows > 0)
			  {
				  for (int j = 0; j < cols; j++)
				  {
					  res_d[j] = padding_val;
				  }

				  for (int i = 1; i < pad_rows; i++)
				  {
					  std::copy(res_d, res_d + cols, res_d + i * cols);
				  }
				  std::copy(res_d, res_d + cols * pad_rows, res_d + (rows - pad_rows) * cols);
			  }
			  if (pad_cols > 0)
			  {
				  for (int j = 0; j < pad_cols; j++)
				  {
					  res_d[j + cols * pad_rows] = padding_val;
				  }
			  }
          }
          else
          {
              std::copy(res.data(), res.data() + cols * pad_rows, res_d);
              std::copy(res_d, res_d + cols * pad_rows, res_d + (rows - pad_rows) * cols);
          }

          for (int i = pad_rows; i < rows - pad_rows; i++)
          {
              std::copy(res.data(), res.data() + pad_cols, res_d + i * cols);
              std::copy(src_d + (i - pad_rows) * src_cols,
                        src_d + (i - pad_rows + 1) * src_cols,
                        res_d + i * cols + pad_cols);
              std::copy(res_d, res_d + pad_cols, res_d + (i + 1) * cols - pad_cols);
          }
      }
      return res;
  }
  
  template<typename T>
  thensor<T,2> unroll_kernel(const thensor<T,4> &kernel)
  {
      const int channels = kernel.shape()[0], filters = kernel.shape()[1];
      const int k_size = kernel.shape()[2]* kernel.shape()[3];
      thensor<T, 2> resT({filters, channels*k_size});
      T* d_data = resT.data();
      const T* k_data = kernel.data();
      for(int i = 0; i < filters; ++i)
      {
          for(int j = 0; j < channels; ++j)
          {
              const T* src = k_data + (j*filters + i)* k_size;
              std::copy(src, src + k_size, d_data);
              d_data += k_size;
          }
      }
      return transpose(resT);
  }

  template<typename T>
  thensor<T,2> unroll_image(const thensor<T,3> &img,const std::vector<int>& kernel_shape, int stride_vert, int stride_hor){
      const int i_rows = img.shape()[1], i_cols = img.shape()[2];
      assert(4 == kernel_shape.size());
      assert(img.shape()[0] == kernel_shape[0]);
      const int depth = kernel_shape[0] * kernel_shape[2] * kernel_shape[3];
      const int k_rows = kernel_shape[2], k_cols = kernel_shape[3];
      const int ch_depth = k_rows * k_cols;
      const int ch_size = i_rows * i_cols;
      const int r = ( i_rows - k_rows + stride_vert) / stride_vert, c = (i_cols - k_cols + stride_hor) / stride_hor;
      const int rows = r * c;
      
      thensor<T, 2> res({rows, depth});
      
      for (int k = 0; k < kernel_shape[0]; k++)
      {
          
          const T *src_row = img.data() + ch_size * k;
          T *d_data = res.data() + ch_depth *k;
          for (int m = 0; m < i_rows - k_rows + 1; m += stride_vert)
          {
              const T *src_col = src_row;
              for (int n = 0; n < i_cols - k_cols + 1; n += stride_hor)
              {
                  const T *src = src_col;
                  T* d_ptr = d_data;
                  for (int i = 0; i < k_rows; i++)
                  {
                      std::copy(src, src + k_cols, d_ptr);
                      d_ptr += k_cols;
                      src += i_rows;
                  }
                  d_data += depth;
                  src_col += stride_hor;
              }
              src_row += i_rows * stride_vert;
          }
      }
      return res;
  }
  
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src,
                                     const thensor<T, 4> &kernel,
                                     std::tuple<int,int> stride, // rows, cols
	                                 std::tuple<int, int> padding, // rows, cols (vertical, horizontal)
                                     T padding_val)
  {
	  int pad_rows = std::get<0>(padding);
	  int pad_cols = std::get<1>(padding);
	  int stride_vert = std::get<0>(stride);
	  int stride_hor = std::get<1>(stride);
      assert(pad_rows >= 0 && pad_cols >= 0);
      assert(stride_vert > 0 && stride_hor > 0);
      thensor<T, 3> mat;
	  if (pad_rows != 0 && pad_cols != 0)
          mat = padd_image(src, pad_rows, pad_cols, padding_val); 
      else
          mat = src;
      const int  d_channels = kernel.shape()[1], k_rows = kernel.shape()[2], k_cols = kernel.shape()[3];
      const int s_rows = mat.shape()[1], s_cols = mat.shape()[2];
     
      thensor<T, 2> filter = unroll_kernel(kernel);
      thensor<T, 2> img = unroll_image(mat, kernel.shape(), stride_vert, stride_hor);
      
      
      return reshape<T,2,3>(transpose(matmul(img, filter)), 
          {d_channels,(s_rows - k_rows + stride_vert) / stride_vert, (s_cols - k_cols + stride_hor) / stride_hor });
  }
  
  
}
#endif //LIGHTY_ALGEBRA_HPP
