//
// Created by norrilsk on 17.02.20.
//

#ifndef LIGHTY_ALGEBRA_HPP
#define LIGHTY_ALGEBRA_HPP
#include"thensor.hpp"
#include <tuple>
namespace linal
{

  inline std::tuple<int32_t, int32_t, int32_t> getConv2dOutputShape(const std::vector<int32_t>& img_shape, const std::vector<int32_t>& kernel_shape, std::tuple<int32_t, int32_t> stride, std::tuple<int32_t, int32_t> padding);
  namespace experimental
  {
  }
  namespace depricated
  {
	  template<typename T>
	  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int32_t stride = 1, int32_t padding = 0, T padding_val = T());
	  template<typename T>
	  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int32_t stride = 1, int32_t padding = 0, T padding_val = T());
  }
  template<typename T>
  thensor<T, 2> unroll_kernel(const thensor<T, 2> &kernel);
  template<typename T>
  thensor<T, 2> unroll_image(const thensor<T, 2> &img, int32_t k_rows, int32_t k_cols, int32_t stride_vert, int32_t stride_hor);
  template<typename T>
  thensor<T, 2> padd_image(const thensor<T, 2> &img, int32_t pad_rows, int32_t pad_cols, T padding_val = T());
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, std::tuple<int32_t, int32_t> stride , std::tuple<int32_t, int32_t> padding, T padding_val = T());
  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int32_t stride = 1 , int32_t padding = 0 , T padding_val = T());
  template<typename T>
  thensor<T, 2> unroll_kernel(const thensor<T, 4> &kernel);
  template<typename T>
  thensor<T, 2> unroll_image(const thensor<T, 3> &img, const std::vector<int32_t> &kernel_shape, int32_t stride_vert, int32_t stride_hor);
  template<typename T>
  thensor<T, 3> backward_unroll_image(const thensor<T, 2> &unrolled_img, const std::vector<int32_t> &kernel_shape, const std::vector<int32_t>& input_shape, int32_t stride_vert, int32_t stride_hor, int32_t pad_rows,int32_t pad_cols);
  template<typename T>
  thensor<T, 3> padd_image(const thensor<T, 3> &img, int32_t pad_rows, int32_t pad_cols, T padding_val = T());
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, std::tuple<int32_t, int32_t> stride , std::tuple<int32_t, int32_t> padding, T padding_val = T());
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int32_t  stride =  1, int32_t padding =  0, T padding_val = T());

  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 2> &left, const thensor<T, 2> &right);
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 1> &left, const thensor<T, 1> &right);

  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int32_t stride, int32_t padding , T padding_val )
  {
	  return conv2d(src, kernel, { stride,stride }, { padding,padding }, padding_val);
  }

  template<typename T>
  thensor<T, 2> conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel,int32_t  stride , int32_t padding , T padding_val )
  {
	  return conv2d(src, kernel, { stride,stride }, { padding,padding }, padding_val);
  }



  template<typename T>
  thensor<T,2> matmul (const thensor<T,1>& left, const thensor<T,1>& right)
  {
      int32_t shp0 = right.size();
      int32_t shp1 = left.size();
      thensor<T,2> res({shp0, shp1});
      T* dst = res.data();
      const T* l = left.data();
      const T* r = right.data();
      
      for(int32_t i = 0; i < shp0; i++)
      {
          for(int32_t j = 0; j < shp1; j++, dst++)
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
      int32_t M = left.shape()[0];
      int32_t N = right.shape()[1];
      int32_t K = left.shape()[1];
      assert(right.shape()[0] == K);
      thensor<T,2> res({M,N});
      T* dst = res.data();
      for (int32_t i = 0; i < M; i++)
      {
          T * dst_row = dst + i * N;
          const T * lhs_row = lhs + i * K;
          for (int32_t j = 0; j < N; j++)
              dst_row[j] = T();
          for (int32_t k = 0; k < K; k++)
          {
              const T * rhs_row = rhs + k * N;
              T lhs_val = lhs_row[k];
              for (int32_t j = 0; j < N; ++j)
                  dst_row[j] += lhs_val * rhs_row[j];
          }
      }
      return res;
  }
  template<typename T>
  thensor<T, 3> depricated::conv2d(const thensor<T, 3> &src, const thensor<T, 4> &kernel, int32_t stride , int32_t padding, T padding_val )
  {
      assert(padding >= 0);
      assert(stride > 0);
      assert(kernel.shape()[0] == src.shape()[2]);
      const int32_t k_rows = kernel.shape()[2], k_cols = kernel.shape()[3];
      const int32_t d_rows = src.shape()[0] + 2 * padding, d_cols = src.shape()[1] + 2 * padding;
      const int32_t d_channels = kernel.shape()[1];
      const int32_t s_channels = src.shape()[2];
      thensor<T,3> dst({d_channels,( d_rows - k_rows + stride) / stride, (d_cols - k_cols + stride) / stride});
      zero_set(dst);
	  thensor<T, 2> wrapper;
	  thensor<T, 3> wrapper3;
	  thensor<T, 3> src_copy = src.copy();
	  /*we had channel first structure and then decide to make channel last, yes we are strange guys*/
	  /*we convert here channel last to channel  first */
	  /*we use this code just for verification*/

	  thensor<T, 2> src_transposed = linal::transpose(wrapper.wrap(src_copy.data(), std::vector<int32_t>{src.shape()[0] * src.shape()[1], s_channels}));
	  wrapper3.wrap(src_transposed.data(), std::vector<int32_t>{src.shape()[2], src.shape()[0], src.shape()[1]});
      for (int32_t i = 0 ; i < s_channels; i++)
      {
          thensor<T,2> single_channel = wrapper3[i];
          thensor<T,3> channel_kernels = kernel[i];
          for (int32_t j = 0 ; j < d_channels; j++)
          {
              dst[j] += depricated::conv2d(single_channel,channel_kernels[j],stride,padding,padding_val);
          }
      }
	  thensor<T, 2> dst_transposed = linal::transpose(wrapper.wrap(dst.data(), std::vector<int32_t>{d_channels ,dst.shape()[1] * dst.shape()[2]}));
	  return reshape<T, 2, 3>(dst_transposed, { dst.shape()[1], dst.shape()[2], dst.shape()[0] });
  }
  template<typename T>
  thensor<T, 2> depricated::conv2d(const thensor<T, 2> &src, const thensor<T, 2> &kernel, int32_t stride, int32_t padding, T padding_val)
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
          const int32_t rows = src.shape()[0] + 2 * padding;
          const int32_t cols = src.shape()[1] + 2 * padding;
          mat = thensor<T, 2>({rows, cols});
          for (int32_t i = 0; i < padding; i++)
          {
              for (int32_t j = 0; j < cols; j++)
              {
                  mat[i][j] = padding_val;
              }
          }
          for (int32_t i = padding; i < rows - padding; i++)
          {
              for (int32_t j = 0; j < padding; j++)
              {
                  mat[i][j] = padding_val;
              }
              for (int32_t j = padding; j < cols - padding; j++)
              {
                  mat[i][j] = src[i - padding][j - padding];
              }
              for (int32_t j = cols - padding; j < cols; j++)
              {
                  mat[i][j] = padding_val;
              }
          }
          for (int32_t i = rows - padding; i < rows; i++)
          {
              for (int32_t j = 0; j < cols; j++)
              {
                  mat[i][j] = padding_val;
              }
          }
      }
      const int32_t k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int32_t s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T,2> dst({( s_rows - k_rows + stride) / stride, (s_cols - k_cols + stride) / stride});
      
      for (int32_t m = 0; m < s_rows - k_rows + 1; m+=stride)
      {
          for (int32_t n = 0; n < s_cols - k_cols + 1; n+=stride)
          {
              T tmp = T();
              for (int32_t i = 0 ; i < k_rows; i++)
              {
                  for(int32_t j = 0; j < k_cols; j++)
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
  thensor<T,2> padd_image(const thensor<T,2> &img, int32_t pad_rows, int32_t pad_cols, T padding_val)
  {
      if (pad_rows <= 0 && pad_cols <= 0)
      {
          return img.copy();
      }
      const int32_t src_rows = img.shape()[0];
      const int32_t src_cols = img.shape()[1];
      const int32_t rows = src_rows + 2 * pad_rows;
      const int32_t cols = src_cols + 2 * pad_cols;
      
      thensor<T, 2> res({ rows, cols});
      
      T* res_d = res.data();
      const T* src_d = img.data();
	  if (pad_rows > 0)
	  {
		  for (int32_t j = 0; j < cols; j++)
		  {
			  res_d[j] = padding_val;
		  }

		  for (int32_t i = 1; i < pad_rows; i++)
		  {
			  std::copy(res_d, res_d + cols, res_d + i * cols);
		  }
		  for (int32_t i = rows - pad_rows; i < rows; i++)
		  {
			  std::copy(res_d, res_d + cols, res_d + i * cols);
		  }
	  }
	  if (pad_cols > 0)
	  {
		  for (int32_t j = 0; j < pad_cols; j++)
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
      for (int32_t i = pad_rows + 1 ; i < rows - pad_rows; i++)
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
      const int32_t k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      thensor<T, 2> res({k_rows * k_cols, 1});
      std::copy(kernel.data(),kernel.data() + kernel.size(), res.data());
      return res;
  }
  

  template<typename T>
  thensor<T,2> unroll_image(const thensor<T,2> &img, int32_t k_rows, int32_t k_cols, int32_t stride_vert, int32_t stride_hor){
      const int32_t i_rows = img.shape()[0], i_cols = img.shape()[1];
      const int32_t depth = k_cols * k_rows;
      const int32_t r = ( i_rows - k_rows + stride_vert) / stride_vert, c = (i_cols - k_cols + stride_hor) / stride_hor;
      const int32_t rows = r * c;
      
      thensor<T, 2> res({rows, depth});
      T* d_ptr = res.data();
      const T* src_row = img.data();
      
      for (int32_t m = 0; m < i_rows - k_rows + 1; m+= stride_vert)
      {
          const T* src_col = src_row;
          for (int32_t n = 0; n < i_cols - k_cols + 1; n += stride_hor)
          {
              const T* src =  src_col;
              for (int32_t i = 0 ; i < k_rows; i++)
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
                                     std::tuple<int32_t,int32_t> stride,
									 std::tuple<int32_t, int32_t> padding,
                                     T padding_val)
  {
	  int32_t pad_rows = std::get<0>(padding);
	  int32_t pad_cols = std::get<1>(padding);
	  int32_t stride_vert = std::get<0>(stride);
	  int32_t stride_hor = std::get<1>(stride);
	  assert(pad_rows >= 0 && pad_cols >= 0);
	  assert(stride_vert > 0 && stride_hor > 0);
      thensor<T, 2> mat;
      if (pad_rows != 0 && pad_cols != 0)
          mat = padd_image(src, pad_rows, pad_cols, padding_val);
      else
          mat = src;
      const int32_t k_rows = kernel.shape()[0], k_cols = kernel.shape()[1];
      const int32_t s_rows = mat.shape()[0], s_cols = mat.shape()[1];
      thensor<T, 2> filter = unroll_kernel(kernel);
      thensor<T, 2> img = unroll_image(mat, k_rows,k_cols, stride_vert, stride_hor);
      
      return matmul(img, filter);
  }
  
  template<typename T>
  thensor<T,3> padd_image(const thensor<T,3> &img, int32_t pad_rows, int32_t pad_cols, T padding_val)
  {
      if (pad_rows <= 0 && pad_cols <= 0)
      {
          return img.copy();
      }
      const int32_t channels = img.shape()[2];
      const int32_t src_rows = img.shape()[0];
      const int32_t src_cols = img.shape()[1];
      const int32_t rows = src_rows + 2 * pad_rows;
      const int32_t cols = src_cols + 2 * pad_cols;
	  const int32_t row_stride = cols * channels;
	  const int32_t indent = pad_cols * channels;

      thensor<T, 3> res({rows, cols, channels });
      const int32_t s_ch_size = src_rows * src_cols;
      const int32_t d_ch_size = rows * cols;
   
	  T *res_d = res.data();
	  const T *src_d = img.data();
	  
	  if (pad_rows > 0)
	  {
		  for (int32_t j = 0; j < row_stride; j++)
		  {
			  res_d[j] = padding_val;
		  }

		  for (int32_t i = 1; i < pad_rows; i++)
		  {
			  std::copy(res_d, res_d + row_stride, res_d + i * row_stride);
		  }
		  std::copy(res_d, res_d + row_stride * pad_rows, res_d + (rows - pad_rows) * row_stride);
	  }
	  if (pad_cols > 0)
	  {
		  for (int32_t j = 0; j < indent; j++)
		  {
			  res_d[j + row_stride * pad_rows] = padding_val;
		  }
	  }
	 
      
      for (int32_t i = pad_rows; i < rows - pad_rows; i++)
      {
          std::copy(res.data(), res.data() + indent, res_d + i * row_stride);
          std::copy(src_d + (i - pad_rows) * src_cols * channels,
                  src_d + (i - pad_rows + 1) * src_cols * channels,
                  res_d + i * row_stride + indent);
          std::copy(res_d, res_d + indent, res_d + (i + 1) * row_stride - indent);
      }
        
      return res;
  }
  
  template<typename T>
  thensor<T,2> unroll_kernel(const thensor<T,4> &kernel)
  {
      const int32_t channels = kernel.shape()[0], filters = kernel.shape()[1];
	  const int32_t k_rows = kernel.shape()[2];
	  const int32_t k_cols = kernel.shape()[3];
      thensor<T, 2> res({ channels * k_rows * k_cols, filters});
      T* d_data = res.data();
      const T* k_data = kernel.data();
	  //it is really cache unfriendly but we suppose that this is rare-called function 
	  // we made it slow in order to make fast hot code
	  for (int32_t m = 0; m < k_rows; m++)
	  {
		  for (int32_t n = 0; n < k_cols; n++)
		  {
			  for (int32_t i = 0; i < channels; i++)
			  {
				  for (int32_t j = 0; j < filters; j++)
				  {
					  *(d_data++) = k_data[n + k_cols * ( m + k_rows * ( j + i * filters) )];
				  }
			  }
		  }
	  }
	  return res;
  }

  template<typename T>
  thensor<T,2> unroll_image(const thensor<T,3> &img,const std::vector<int32_t>& kernel_shape, int32_t stride_vert, int32_t stride_hor){
      const int32_t i_rows = img.shape()[0], i_cols = img.shape()[1];
	  const int32_t channels = img.shape()[2];
      assert(4 == kernel_shape.size());
      assert(img.shape()[2] == kernel_shape[0]);
      const int32_t depth = kernel_shape[0] * kernel_shape[2] * kernel_shape[3];
      const int32_t k_rows = kernel_shape[2], k_cols = kernel_shape[3];
      const int32_t ch_depth = k_rows * k_cols;
      const int32_t ch_size = i_rows * i_cols;
      const int32_t r = ( i_rows - k_rows + stride_vert) / stride_vert, c = (i_cols - k_cols + stride_hor) / stride_hor;
      const int32_t rows = r * c;
	  if (!((stride_vert * r) == (i_rows - k_rows + stride_vert)) || !((stride_hor * c) == (i_cols - k_cols + stride_hor)))
		  std::cout << "Warning : image size is not multiple of stride" << std::endl;
      thensor<T, 2> res({rows, depth});
      
  
      const T *src_data = img.data() ;
      T *d_ptr = res.data();
	  for (int32_t m = 0; m < i_rows - k_rows + 1; m += stride_vert)
	  {
		  const T *src_col = src_data + m * i_cols * channels;
		  for (int32_t n = 0; n < i_cols - k_cols + 1; n += stride_hor)
		  {
			  const T *src = src_col ;
			  for (int32_t i = 0; i < k_rows; i++)
			  {
				  std::copy(src, src + k_cols * channels, d_ptr);
				  d_ptr += k_cols * channels;
				  src += i_cols * channels;
			  }
			  src_col += stride_hor * channels;
		  }
	  }

      return res;
  }

  template<typename T>
  thensor<T, 3> backward_unroll_image(const thensor<T, 2>& unrolled_img, const std::vector<int32_t>& kernel_shape,
	  const std::vector<int32_t>& input_shape, int32_t stride_vert, int32_t stride_hor, int32_t pad_rows, int32_t pad_cols)
  {


	  const int32_t channels = kernel_shape[0];
	  assert(4 == kernel_shape.size());
	  const int32_t depth = kernel_shape[0] * kernel_shape[2] * kernel_shape[3];
	  const int32_t k_rows = kernel_shape[2], k_cols = kernel_shape[3];
	  const int32_t ch_depth = k_rows * k_cols;
	  const int32_t i_rows = input_shape[0], i_cols = input_shape[1];
	  const int32_t ch_size = i_rows * i_cols;
	  const int32_t r = (i_rows - k_rows + stride_vert) / stride_vert, c = (i_cols - k_cols + stride_hor) / stride_hor;
	  if  ( !((stride_vert * r) == (i_rows - k_rows + stride_vert))  ||  !((stride_hor * c) == (i_cols - k_cols + stride_hor)))
		  std::cout << "Warning : image size is not multiple of stride" << std::endl;
	  const int32_t rows = r * c;
	  assert(rows == unrolled_img.shape()[0]);
	  assert(depth == unrolled_img.shape()[1]);



	  
	   thensor<T, 3> res = linal::zero_thensor<T,3>(std::vector<int32_t>({i_rows - 2 * pad_rows, i_cols - 2 * pad_cols,channels}));
	   T* r_data = res.data();
	   const T* uim_data = unrolled_img.data();
	   const int32_t line_stride = channels * i_cols;
	   for (int32_t m = 0; m < r; m++)
	   {
		   for (int32_t n = 0; n < c; n++)
		   {
			   for (int32_t j = 0; j < k_rows; j++)
			   {
				   int32_t input_row = j + m * stride_vert - pad_rows;
				   if (input_row < 0 || input_row > res.shape()[0])
					   continue;
				   for (int32_t k = 0; k < k_cols; k++)
				   {
					   int32_t input_col = k + n * stride_hor - pad_cols;
					   if (input_col < 0 || input_col > res.shape()[1])
						   continue; 
					   for (int32_t c = 0; c < channels; c++, uim_data++)
					   {
						   *(r_data + c + channels * input_col + line_stride * input_row )=
							   *uim_data;
					   }
				   }

			   }
		   }
	   }
    
	   return res;
  }
  
  template<typename T>
  thensor<T, 3> conv2d(const thensor<T, 3> &src,
                                     const thensor<T, 4> &kernel,
                                     std::tuple<int32_t,int32_t> stride, // rows, cols
	                                 std::tuple<int32_t, int32_t> padding, // rows, cols (vertical, horizontal)
                                     T padding_val)
  {
	  int32_t pad_rows = std::get<0>(padding);
	  int32_t pad_cols = std::get<1>(padding);
	  int32_t stride_vert = std::get<0>(stride);
	  int32_t stride_hor = std::get<1>(stride);
      assert(pad_rows >= 0 && pad_cols >= 0);
      assert(stride_vert > 0 && stride_hor > 0);
      thensor<T, 3> mat;
	  if (pad_rows != 0 && pad_cols != 0)
          mat = padd_image(src, pad_rows, pad_cols, padding_val);  
      else
          mat = src;
      const int32_t  d_channels = kernel.shape()[1], k_rows = kernel.shape()[2], k_cols = kernel.shape()[3];
      const int32_t s_rows = mat.shape()[0], s_cols = mat.shape()[1];
     
      thensor<T, 2> filter = unroll_kernel(kernel);
      thensor<T, 2> img = unroll_image(mat, kernel.shape(), stride_vert, stride_hor);
      
      return reshape<T,2,3>( matmul(img, filter), 
	  { (s_rows - k_rows + stride_vert) / stride_vert, (s_cols - k_cols + stride_hor) / stride_hor , d_channels } );
  }

  std::tuple<int32_t, int32_t, int32_t> linal::getConv2dOutputShape(const std::vector<int32_t>& img_shape, const std::vector<int32_t>& kernel_shape, std::tuple<int32_t, int32_t> stride, std::tuple<int32_t, int32_t> padding)
  {
	  int32_t pad_rows = std::get<0>(padding);
	  int32_t pad_cols = std::get<1>(padding);
	  int32_t stride_vert = std::get<0>(stride);
	  int32_t stride_hor = std::get<1>(stride);
	  assert(pad_rows >= 0 && pad_cols >= 0);
	  assert(stride_vert > 0 && stride_hor > 0);
	  const int32_t  d_channels = kernel_shape[1], k_rows = kernel_shape[2], k_cols = kernel_shape[3];
	  const int32_t s_rows = img_shape[0] + 2 * pad_rows;
	  const int32_t s_cols = img_shape[1] + 2 * pad_cols;
	  return { (s_rows - k_rows + stride_vert) / stride_vert, (s_cols - k_cols + stride_hor) / stride_hor , d_channels };
  }

  
  
}
#endif //LIGHTY_ALGEBRA_HPP
