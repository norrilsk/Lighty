//
// Created by norrilsk on 19.09.19.
//

#ifndef LIGHTY_VECTOR_HPP
#define LIGHTY_VECTOR_HPP

#include<memory>
#include<cassert>
#include <ostream>
#include"thensor_data.hpp"
namespace linal
{
  class Container
  {
  public:
      Container() = default;
      virtual ~Container() = default;
      virtual const std::vector<int> &shape() const noexcept  = 0;
      virtual int size() const noexcept = 0;
	  Container(const Container& th) = default;
	  Container(Container&& th) = default;
  };
  
  
  
  template<typename T, int _dim>
  class thensor;
  template<typename T, int _dim>
  thensor<T, _dim> operator+(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  thensor<T, _dim> operator*(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  thensor<T, _dim> operator-(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  thensor<T, _dim> operator&(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  thensor<T, _dim> operator|(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  thensor<T, _dim> operator^(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  bool operator==(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  bool operator!=(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
  template<typename T, int _dim>
  std::ostream &operator<<(std::ostream &os, const thensor<T, _dim> &th);
  template<typename T>
  thensor<T, 1> operator*(const thensor<T, 2> &left, const thensor<T, 1> &right);
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 2> &left, const thensor<T, 2> &right);
  template<typename T>
  thensor<T, 2> matmul(const thensor<T, 1> &left, const thensor<T, 1> &right);
  template<typename T>
  thensor<T, 2> transpose(const thensor<T, 2> &left);
  template<typename T, int _dim>
  thensor<T, _dim>& zero_set(thensor<T, _dim>&);
  
  
  template<typename T, int _dim>
  class thensor : public Container
  {
  private:
      ThensorData<T, _dim> _data;
  
  public:
      typedef T d_type;
      
      friend thensor<T, _dim> operator+<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend thensor<T, _dim> operator-<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend thensor<T, _dim> operator&<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend thensor<T, _dim> operator|<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend thensor<T, _dim> operator^<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend thensor<T, _dim> operator*<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend std::ostream &operator<<<T>(std::ostream &os, const thensor<T, _dim> &th);
      
      template<typename U, typename S, int d>
      friend thensor<S, d> operator*(const U &left, const thensor<S, d> &right);
      template<typename U, typename S, int d>
      friend thensor<S, d> operator*(const thensor<S, d> &right, const U &left);
      friend bool operator==<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      friend bool operator!=<T>(const thensor<T, _dim> &left, const thensor<T, _dim> &right);
      
      thensor<T, _dim> operator-();
      thensor<T, _dim> operator+();
      thensor<T, _dim> &operator+=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator-=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator&=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator|=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator^=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator*=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator/=(const thensor<T, _dim> &right);
      thensor<T, _dim> &operator=(const thensor<T, _dim> &right) & { _data = right._data; return *this; };
      thensor<T, _dim> &operator=(thensor<T, _dim> &&right) & noexcept { _data = std::move(right._data); return *this; };
      thensor<T, _dim> &operator=(const thensor<T, _dim> &right) && { std::move(_data) = right._data; return *this; };
      thensor<T, _dim> &operator=(thensor<T, _dim> &&right) && { std::move(_data) = std::move(right._data); return *this; };
      T dot(const thensor<T, _dim> &right) const;
      
      thensor<T, _dim - 1> operator[](int idx) const;
      
      inline int size() const noexcept { return _data.size(); };
      inline const T *data() const noexcept { return _data.data(); }
      inline T *data() noexcept { return _data.data(); }
      inline const std::vector<int> &shape() const noexcept { return _data.shape(); };
      
      //this method initialized data from external source
      //it means that data will not be allocated or free-ed inside
      void wrap(T *data, const std::vector<int> &shapes) { _data.construct(data, shapes); };
      
      thensor<T, _dim> copy() const
      {
          thensor<T, _dim> tmp;
          tmp._data = _data.copy();
          return tmp;
      }
      
	  thensor<T, _dim>& copy(const thensor<T, _dim>& th) 
	  {
		  _data.copy(th._data);
		  return *this;
	  }
      thensor() = default;
	  thensor(const thensor<T, _dim>& th) = default;
	  thensor(thensor<T, _dim>&& th)  = default;
      explicit thensor(const std::vector<int> &shapes) : _data(shapes) {};
      ~thensor() = default;
  };
  
  template<typename T>
  class thensor<T,1> :  public Container
  {
  private:
      ThensorData<T, 1> _data;
  
  public:
      typedef T d_type;
      
      friend thensor<T, 1> operator+<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend thensor<T, 1> operator-<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend thensor<T, 1> operator&<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend thensor<T, 1> operator|<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend thensor<T, 1> operator^<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend thensor<T, 1> operator*<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend std::ostream &operator<<<T>(std::ostream &os, const thensor<T, 1> &th);
      
      template<typename U, typename S, int d>
      friend thensor<S, d> operator*(const U &left, const thensor<S, d> &right);
      template<typename U, typename S, int d>
      friend thensor<S, d> operator*(const thensor<S, d> &right, const U &left);
      friend bool operator==<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      friend bool operator!=<T>(const thensor<T, 1> &left, const thensor<T, 1> &right);
      
      thensor<T, 1> operator-();
      thensor<T, 1> operator+();
      thensor<T, 1> &operator+=(const thensor<T, 1> &right);
      thensor<T, 1> &operator-=(const thensor<T, 1> &right);
      thensor<T, 1> &operator&=(const thensor<T, 1> &right);
      thensor<T, 1> &operator|=(const thensor<T, 1> &right);
      thensor<T, 1> &operator^=(const thensor<T, 1> &right);
      thensor<T, 1> &operator*=(const thensor<T, 1> &right);
      thensor<T, 1> &operator/=(const thensor<T, 1> &right);
      thensor<T, 1> &operator=(const thensor<T, 1> &right) & { _data = right._data; return *this; };
      thensor<T, 1> &operator=(thensor<T, 1> &&right) & { _data = std::move(right._data); return *this; };
      thensor<T, 1> &operator=(const thensor<T, 1> &right) && { std::move(_data) = right._data; return *this; };
      thensor<T, 1> &operator=(thensor<T, 1> &&right) && { std::move(_data) = std::move(right._data); return *this; };
      T dot(const thensor<T, 1> &right) const;
      
      T& operator[](int idx) const;
      
      inline int size() const noexcept { return _data.size(); };
      inline const T *data() const noexcept { return _data.data(); }
      inline T *data() noexcept { return _data.data(); }
      inline const std::vector<int> &shape() const noexcept { return _data.shape(); };
      
      //this method initialized data from external source
      //it means that data will not be allocated or free-ed inside
      void wrap(T *data, const std::vector<int> &shapes) { _data.construct(data, shapes); };
      
      thensor<T, 1> copy()
      {
          thensor<T, 1> tmp;
          tmp._data = _data.copy();
          return tmp;
      }
	  thensor<T, 1>& copy(const thensor<T, 1>& th)
	  {
		  _data.copy(th._data);
		  return *this;
	  }
      
      thensor() = default;
      explicit thensor(const std::vector<int> &shapes) : _data(shapes) {};
      explicit thensor(int len) : _data({len}) {};
	  thensor(const thensor<T, 1>& th) = default;
	  thensor(thensor<T, 1> && th) = default;
      ~thensor() = default;
  };
  
  template<typename T, int _dim>
  bool operator==(const thensor<T, _dim> &left, const thensor<T, _dim> &right)
  {
      assert(left.size() == right.size());
      bool res = true;
      const T *lhs = left.data();
      const T *rhs = right.data();
      for (int i = 0; i < left.size(); i++)
      {
          res &= (lhs[i] == rhs[i]);
          if (!res)
              return false;
      }
      return res;
  }
  
  template<typename T, int _dim>
  bool operator!=(const thensor<T, _dim> &left, const thensor<T, _dim> &right)
  {
      return !(right == left);
  }
  
  template <typename T, int _dim>
  thensor<T,_dim> operator+(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp+=right;
  }
  template <typename T, int _dim>
  thensor<T,_dim> operator-(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp-=right;
  }
  template <typename T, int _dim>
  thensor<T,_dim> operator&(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp&=right;
  }
  template <typename T, int _dim>
  thensor<T,_dim> operator^(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp^=right;
  }
  template <typename T, int _dim>
  thensor<T,_dim> operator|(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp|=right;
  }
  template <typename T, int _dim>
  thensor<T,_dim> operator*(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp*=right;
  }
  
  template <typename T, int _dim>
  thensor<T,_dim> operator/(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
  {
      thensor<T,_dim> tmp(left);
      return tmp/=right;
  }
  
  template<typename U, typename S, int _dim>
  inline thensor<S,_dim> operator*(const U & left, const thensor<S,_dim>& right)
  {
      
      thensor<S,_dim> tmp(right);
      S *lhs = tmp.data();
      const S *rhs = right.data();
	  int sz = right.size();
      for (int i = 0; i < sz; i++)
      {
          lhs[i] = left*rhs[i];
      }
      return tmp;
  }
  
  template<typename U, typename S, int _dim>
  inline thensor<S,_dim> operator*(const thensor<S,_dim>& right, const U & left)
  {
      
      return left*right;
  }
  
  template<typename T>
  thensor<T, 2> transpose(const thensor<T, 2> &left)
  {
      auto& shp = left.shape();
      thensor<T,2> res({shp[1],shp[0]});
      const T* col = left.data();
      T* dst = res.data();
      for (int i = 0 ; i < shp[1]; i++,col++)
      {
          const T* src =col;
          for (int j = 0; j < shp[0]; j++,src+=shp[1],dst++)
          {
              *dst = *src;
          }
      }
      
      return res;
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
              const T * rhs_row = rhs  + k * N;
              T lhs_val = lhs_row[k];
              for (int j = 0; j < N; ++j)
                  dst_row[j] += lhs_val * rhs_row[j];
          }
      }
      return res;
  }
  
  template<typename T>
  thensor<T, 1> operator*(const thensor<T, 2> &left, const thensor<T, 1> &right)
  {
      assert(left.shape()[1] == right.shape()[0]);
      int outShape = left.shape()[0];
      int inShape = right.shape()[0];
      thensor<T,1> resVec(outShape);
      T*  res = resVec.data();
      const T* vec = right.data();
      const T* src = left.data();
      for (int i =0; i < outShape; i++)
      {
          T t= T();
          
          for (int j =0 ; j < inShape; j++,src++)
          {
              t+=vec[j]*(*src);
          }
          res[i] = t;
      }
      return resVec;
  }
  
  template<typename T, int _dim>
  std::ostream &operator<<(std::ostream &os, const thensor<T, _dim> &th)
  {
      std::vector<int> shape = th.shape();
      for(int i = 0 ; i <th.size(); i++)
      {
          int k = 1;
          if (i != 0)
          {
              for (int j = _dim - 1; j >= 0, i % (k * shape[j]) == 0; --j)
              {
                  os << '\n';
                  k*=shape[j];
              }
          }
          os << th.data()[i]<< ' ';
      }
      return os;
  }
  template<typename T, int _dim>
  thensor<T, _dim> &zero_set(thensor<T, _dim> & src)
  {
      T* data = src.data();
      for (int i = 0; i < src.size(); i++)
      {
          data[i] = T();
      }
      return src;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim - 1> thensor<T, _dim>::operator[](int idx) const
  {
      const ThensorData<T,_dim-1>& data = _data[idx];
      thensor<T, _dim - 1> tmp;
      
      tmp.wrap(data.data(),data.shape());
      return tmp;
  }
  
  template<typename T, int _dim>
  inline T thensor<T, _dim>::dot(const thensor<T, _dim> &right) const
  {
      assert(right.size() == size());
      assert(size() > 0);
      
      const T *lhs = _data.data();
      const T *rhs = right.data();
      T tmp = lhs[0] * rhs[0];
      for (int i = 1; i < size(); i++)
      {
          tmp += rhs[i] * lhs[i];
      }
      
      return tmp;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> thensor<T, _dim>::operator-()
  {
      thensor<T, _dim> tmp(shape());
      const T *lhs = tmp.data();
      const T *rhs = _data.data();
      for (int i = 0; i < size(); i++)
      {
          lhs[i] = -rhs[i];
      }
      return tmp;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> thensor<T, _dim>::operator+()
  {
      return this->copy();
  }

  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator+=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] += rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator-=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] -= rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator|=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] |= rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator&=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] &= rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator^=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] ^= rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator*=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] *= rhs[i];
      }
      return *this;
  }
  
  template<typename T, int _dim>
  inline thensor<T, _dim> &thensor<T, _dim>::operator/=(const thensor<T, _dim> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] /= rhs[i];
      }
      return *this;
  }
  

  template<typename T>
  inline T& thensor<T, 1>::operator[](int idx) const
  {
      return _data[idx];
  }
  
  template<typename T>
  inline T thensor<T, 1>::dot(const thensor<T, 1> &right) const
  {
      assert(right.size() == size());
      assert(size() > 0);
      
      const T *lhs = _data.data();
      const T *rhs = right.data();
      T tmp = lhs[0] * rhs[0];
      for (int i = 1; i < size(); i++)
      {
          tmp += rhs[i] * lhs[i];
      }
      
      return tmp;
  }
  
  template<typename T>
  inline thensor<T, 1> thensor<T, 1>::operator-()
  {
      thensor<T, 1> tmp(shape());
      const T *lhs = tmp.data();
      const T *rhs = _data.data();
      for (int i = 0; i < size(); i++)
      {
          lhs[i] = -rhs[i];
      }
      return tmp;
  }
  
  template<typename T>
  inline thensor<T, 1> thensor<T, 1>::operator+()
  {
      return this->copy();
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator+=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] += rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator-=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] -= rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator|=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] |= rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator&=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] &= rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator^=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] ^= rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator*=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] *= rhs[i];
      }
      return *this;
  }
  
  template<typename T>
  inline thensor<T, 1> &thensor<T, 1>::operator/=(const thensor<T, 1> &right)
  {
      assert(size() == right.size());
      
      T *lhs = data();
      const T *rhs = right.data();
      
      for (int i = 0; i < right.size(); i++)
      {
          lhs[i] /= rhs[i];
      }
      return *this;
  }
  
  
}

#endif //LIGHTY_VECTOR_HPP
