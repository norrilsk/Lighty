//
// Created by norrilsk on 19.09.19.
//

#ifndef LIGHTY_VECTOR_HPP
#define LIGHTY_VECTOR_HPP

#include<memory>
#include<cassert>
#include <vector>
#include <ostream>
namespace linal
{
template <typename T, int _dim> class thensor;
template <typename T, int _dim>  thensor<T,_dim> operator+(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  thensor<T,_dim> operator*(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  thensor<T,_dim> operator-(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  thensor<T,_dim> operator&(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  thensor<T,_dim> operator|(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  thensor<T,_dim> operator^(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  bool operator==(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  bool operator!=(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
template <typename T, int _dim>  std::ostream &operator<<(std::ostream& os, const thensor<T,_dim>& th);

template<typename T> thensor<T,2> matmul (const thensor<T,2>& left, const thensor<T,2>& right);
template<typename T> thensor<T,2> matmul (const thensor<T,1>& left, const thensor<T,1>& right);
template<typename T> thensor<T,2> transpose(const thensor<T,2>& left);

template <typename T> void default_deleter(T *ptr)
{
    delete[] ptr;
}
template <typename T> void  empty_deleter(T *ptr)
{
}


template <typename T, int _dim >
class thensor
{
private:
    int _size = 0;
    
    std::unique_ptr<int[]> _shape;
    std::unique_ptr<T[], void(*)(T*)> _data;
    
public:
    friend  thensor<T,_dim> operator+<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  thensor<T,_dim> operator-<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  thensor<T,_dim> operator&<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  thensor<T,_dim> operator|<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  thensor<T,_dim> operator^<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  thensor<T,_dim> operator*<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  std::ostream &operator<< <T>(std::ostream& os, const thensor<T,_dim>& th);
    
    template <typename U, typename S,int d> friend  thensor<S,d> operator*(const U& left, const thensor<S, d>& right);
    friend  bool operator==<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    friend  bool operator!=<T>(const thensor<T,_dim>& left, const thensor<T,_dim>& right);
    
    thensor<T,_dim> operator-();
    thensor<T,_dim> operator+();
    thensor<T,_dim>& operator+=(const thensor<T,_dim>& right);
    thensor<T,_dim>& operator-=(const thensor<T,_dim>& right);
    thensor<T,_dim>& operator=(const thensor<T,_dim>& right);
    thensor<T,_dim>& operator=(thensor<T,_dim>&& right);
    thensor(const int* shapes, T* data);
    thensor<T,_dim-1> operator[](int idx) const;
    int size() const { return _size; };
    int dim() const { return _dim;}
    const T* data() const{ return _data.get();}
    T* data() { return _data.get();}
    const std::unique_ptr<int[]> & shape() const {return _shape;}
    T dot(const thensor<T,_dim> &right);
    thensor() = default;
    thensor(const std::vector<int>& shapes );
    thensor(const std::unique_ptr<int[]>& shapes );
    
    thensor(const thensor<T,_dim>& vec);
    thensor(thensor<T,_dim>&& vec);
    ~thensor() = default;
};

template <typename T, int _dim> 
inline thensor<T,_dim>::thensor(const std::vector<int> &shapes): _data(nullptr, default_deleter)
{
    int size = 1;
    _shape = std::unique_ptr<int[]>(new int[_dim]);
    for (int i = 0 ; i < _dim; i++)
    {
        assert(shapes[i] >0);
        size*=shapes[i];
        _shape[i] = shapes[i];
    }
    _size = size;
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[size], default_deleter);
}
template <typename T, int _dim> 
inline thensor<T,_dim>::thensor(const std::unique_ptr<int[]> &shapes): _data(nullptr, default_deleter)
{
  int size = 1;
  _shape = std::unique_ptr<int[]>(new int[_dim]);
  for (int i = 0 ; i < _dim; i++)
  {
      assert(shapes[i] >0);
      size*=shapes[i];
      _shape[i] = shapes[i];
  }
  _size = size;
  _data = std::unique_ptr<T[], void(*)(T*)>(new T[size], default_deleter);
}
template <typename T, int _dim> 
inline thensor<T,_dim>::thensor(const thensor<T,_dim> & vec): _data(nullptr, default_deleter)
{
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[vec.size()], default_deleter);
    _shape = std::unique_ptr<int[]>(new int[_dim]);
    for (int i = 0; i < vec.size(); i++)
    {
        _data[i] = vec._data[i];
    }
    for (int i =0; i < _dim; i++)
    {
        _shape[i] = vec._shape[i];
    }
    _size=vec.size();
}
template <typename T, int _dim> 
inline thensor<T,_dim>::thensor(thensor<T,_dim> && vec): _data(nullptr, default_deleter)
{
    _data = std::move(vec._data);
    _size = vec.size();
    _shape = std::move(vec._shape);
}

template <typename T, int _dim> 
inline thensor<T,_dim>& thensor<T,_dim>::operator=(const thensor<T,_dim>& right)
{
    if (this == &right)
    {
        return *this;
    }
    _data = std::unique_ptr<T[]>(new T[right.size()]);
    _shape = std::unique_ptr<int[]>(new int[_dim]);
    for (int i = 0; i < right.size(); i++)
    {
        _data[i] = right._data[i];
    }
    for (int i = 0; i< right._dim; i++)
    {
        _shape[i] = right._shape[i];
    }
    _size = right.size();
    return *this;
}
template <typename T, int _dim> 
thensor<T,_dim>& thensor<T,_dim>::operator=(thensor<T,_dim>&& right)
{
    if (this == &right)
    {
        return *this;
    }
    _data = std::move(right._data);
    _size = right.size();
    _shape = std::move(right._shape);
    return *this;
}
template <typename T, int _dim> 
inline thensor<T,_dim-1> thensor<T,_dim>::operator[](int idx) const
{
    assert(idx<_size);
  
    return  thensor<T,_dim-1>(_shape.get()+1, _data.get()+_size/_shape[0]*idx);
    
}


template <typename T, int _dim>
inline T thensor<T,_dim>::dot(const thensor<T,_dim> &right)
{
    assert(right.size()==size());
    assert(size() > 0);
    
    T tmp = _data[0] * right._data[0];
    for (int i = 1; i < size(); i++)
    {
        tmp += _data[i]*right._data[i];
    }
    
    return tmp;
}

template <typename T, int _dim> 
inline thensor<T,_dim> thensor<T,_dim>::operator-()
{
    thensor<T,_dim> tmp(_dim,_shape);
    for (int i = 0; i < size(); i++)
    {
        tmp._data[i] = -_data[i];
    }
    return tmp;
}
template <typename T, int _dim> 
inline thensor<T,_dim> thensor<T,_dim>::operator+()
{
    thensor<T,_dim> tmp(_dim,_shape);
    for (int i = 0; i < size(); i++)
    {
        tmp._data[i] = _data[i];
    }
    return tmp;
}


template <typename T, int _dim> 
thensor<T,_dim> operator+(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());

    thensor<T,_dim> res(left._shape);
    for (int i = 0; i < left.size(); i++)
    {
        res._data[i] = left._data[i] + right._data[i];
    }
    return res;
}
template <typename T, int _dim>
thensor<T,_dim> operator-(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());

    thensor<T,_dim> res(left._shape);
    for (int i = 0; i < left.size(); i++)
    {
        res._data[i] = left._data[i] - right._data[i];
    }
    return res;
}
template <typename T, int _dim> 
thensor<T,_dim> operator&(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());
    
    thensor<T,_dim> res(left._shape);
    for (int i = 0; i < left.size(); i++)
    {
        res._data[i] = left._data[i] & right._data[i];
    }
    return res;
}
template <typename T, int _dim> 
thensor<T,_dim> operator|(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());
    

    thensor<T,_dim> res(left._shape);
    for (int i = 0; i < left.size(); i++)
    {
        res._data[i] = left._data[i] | right._data[i];
    }
    return res;
}
template <typename T, int _dim> 
thensor<T,_dim> operator^(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());
    

    thensor<T,_dim> res(left._shape);
    for (int i = 0; i < left.size(); i++)
    {
        res._data[i] = left._data[i] ^ right._data[i];
    }
    return res;
}

template <typename T, int _dim>
thensor<T,_dim> operator*(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
  assert(left.size() == right.size());
  
  thensor<T,_dim> res(left._shape);
  for (int i = 0; i < left.size(); i++)
  {
      res._data[i] = left._data[i] * right._data[i];
  }
  return res;
}

template<typename U, typename S, int _dim>
inline thensor<S,_dim> operator*(const U & left, const thensor<S,_dim>& right)
{
    thensor<S,_dim> tmp(right._shape);
    for (int i = 0; i < right.size(); i++)
    {
        tmp._data[i] = left*right._data[i];
    }
    return tmp;
}


template <typename T, int _dim> 
bool operator==(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());

    

    bool res = true;
    for (int i = 0; i < left.size(),res; i++)
    {
        res &= left._data[i] == right._data[i];
    }
    return res;
}
template <typename T, int _dim> 
bool operator!=(const thensor<T,_dim>& left, const thensor<T,_dim>& right)
{
    assert(left.size() == right.size());

    bool res = true;
    for (int i = 0; i < left.size(),res; i++)
    {
        res &= left._data[i] == right._data[i];
    }
    return !res;
}

  template <typename T, int _dim>
inline thensor<T,_dim>&  thensor<T,_dim>::operator+=(const thensor<T,_dim>& right)
{
    assert(size() == right.size());
    
    
    for (int i = 0; i < right.size(); i++)
    {
        _data[i] += right._data[i];
    }
    return *this;
}
template <typename T, int _dim> 
inline thensor<T,_dim>&  thensor<T,_dim>::operator-=( const thensor<T,_dim>& right)
{
    assert(right.size() == size());
    for (int i = 0; i < right.size(); i++)
    {
        _data[i] -= right._data[i];
    }
    return *this;
}
template <typename T, int _dim> 
thensor<T,_dim>::thensor(const int* shapes, T *data)
{
    int size = 1;
    _shape = std::unique_ptr<int[]>(new int[_dim]);
    for (int i = 0 ; i < _dim; i++)
    {
        assert(shapes[i] >0);
        size*=shapes[i];
        _shape[i] = shapes[i];
    }
    _size = size;
    _data = std::unique_ptr<T[], void(*)(T*)>(data, [](T* ptr){});
}


/*special case when dim ==1  */
  
  
  
  
template <typename T>  thensor<T,1> operator*(const thensor<T,1>& left, const thensor<T,2>& right);

template <typename T>
class thensor<T,1>
{
private:
  int _size = 0;
  std::unique_ptr<T[], void(*)(T*)> _data;
  std::unique_ptr<int[]> _shape;

public:
  friend  thensor<T,1> operator+<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  thensor<T,1> operator-<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  thensor<T,1> operator&<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  thensor<T,1> operator|<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  thensor<T,1> operator^<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  thensor<T,1> operator*<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  template <typename U, typename S,int d> friend  thensor<S,d> operator*(const U& left, const thensor<S, d>& right);
  friend  bool operator==<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  bool operator!=<T>(const thensor<T,1>& left, const thensor<T,1>& right);
  friend  std::ostream& operator<< <T>(std::ostream& os, const thensor<T,1>& th);
  
  thensor<T,1> operator-();
  thensor<T,1> operator+();
  thensor<T,1>& operator+=(const thensor<T,1>& right);
  thensor<T,1>& operator-=(const thensor<T,1>& right);
  thensor<T,1>& operator=(const thensor<T,1>& right);
  thensor<T,1>& operator=(thensor<T,1>&& right);
  thensor(const int* shapes, T* data);
  T& operator[](int idx) const;
  int size() const { return _size; };
  const T* data() const{ return _data.get();}
  T* data() { return _data.get();}
  T dot(const thensor<T,1> &right);
  const std::unique_ptr<int[]> & shape() const {return _shape;}
  thensor() = default;
  thensor(int size);
  thensor(const std::vector<int>& shapes );
  thensor(const std::unique_ptr<int[]>& shapes );
  
  thensor(const thensor<T,1>& vec);
  thensor(thensor<T,1>&& vec);
  ~thensor() = default;
};
template <typename T>
inline thensor<T,1>::thensor(int size) : _size(size),  _data(nullptr, default_deleter)
{
    //assert(size>0);
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[size], default_deleter);
    _shape = std::unique_ptr<int[]> (new int[1]);
    _shape[0]= size;
}
template <typename T>
inline thensor<T,1>::thensor(const std::vector<int> &shapes):_data(nullptr, default_deleter)
{
    int size = shapes[0];
    _size = size;
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[size], default_deleter);
    _shape = std::unique_ptr<int[]> (new int[1]);
    _shape[0]= size;
}
template <typename T>
inline thensor<T,1>::thensor(const std::unique_ptr<int[]> &shapes): _data(nullptr, default_deleter)
{
    int size = shapes[0];
    _size = size;
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[size], default_deleter);
    _shape = std::unique_ptr<int[]> (new int[1]);
    _shape[0]= size;
}
template <typename T>
inline thensor<T,1>::thensor(const thensor<T,1> & vec): _data(nullptr, default_deleter)
{
    _data = std::unique_ptr<T[], void(*)(T*)>(new T[vec.size()], default_deleter);
    for (int i = 0; i < vec.size(); i++)
    {
      _data[i] = vec._data[i];
    }
    _size=vec._size;
    _shape = std::unique_ptr<int[]> (new int[1]);
    _shape[0]= _size;
}
template <typename T>
inline thensor<T,1>::thensor(thensor<T,1> && vec): _data(nullptr, default_deleter)
{
  _data = std::move(vec._data);
  _size = vec.size();
  
}

template <typename T>
inline thensor<T,1>& thensor<T,1>::operator=(const thensor<T,1>& right)
{
  if (this == &right)
  {
      return *this;
  }
  _data = std::unique_ptr<T[], void(*)(T*)>(new T[right.size()], default_deleter);
  for (int i = 0; i < right.size(); i++)
  {
      _data[i] = right._data[i];
  }
  _size = right.size();
  return *this;
}
template <typename T>
thensor<T,1>& thensor<T,1>::operator=(thensor<T,1>&& right)
{
  if (this == &right)
  {
      return *this;
  }
  _data = std::move(right._data);
  _size = right.size();
  return *this;
}
template <typename T>
inline T& thensor<T,1>::operator[](int idx) const
{
  assert(idx<_size);
  return  _data[idx];
}


template <typename T >
inline T thensor<T,1>::dot(const thensor<T,1> &right)
{
  assert(right.size()==size());
  assert(size() > 0);
  
  T tmp = _data[0] * right._data[0];
  for (int i = 1; i < size(); i++)
  {
      tmp += _data[i]*right._data[i];
  }
  
  return tmp;
}

template <typename T>
inline thensor<T,1> thensor<T,1>::operator-()
{
  thensor<T,1> tmp(_size);
  for (int i = 0; i < _size; i++)
  {
      tmp._data[i] = -_data[i];
  }
  return tmp;
}
template <typename T>
inline thensor<T,1> thensor<T,1>::operator+()
{
  thensor<T,1> tmp(_size);
  for (int i = 0; i < _size; i++)
  {
      tmp._data[i] = _data[i];
  }
  return tmp;
}

template <typename T>
thensor<T,1>::thensor(const int* shapes, T *data): _data(nullptr, empty_deleter)
{
  _shape = std::unique_ptr<int[]>(new int[1]);
  _shape[0] = shapes[0];
  _size = shapes[0];
  _data = std::unique_ptr<T[], void(*)(T*)>(data, empty_deleter);
}

template <typename T>
inline thensor<T,1>&  thensor<T,1>::operator+=(const thensor<T,1>& right)
{
  assert(size() == right.size());
  
  
  for (int i = 0; i < right.size(); i++)
  {
      _data[i] += right._data[i];
  }
  return *this;
}
template <typename T>
inline thensor<T,1>&  thensor<T,1>::operator-=( const thensor<T,1>& right)
{
    assert(right.size() == size());
    for (int i = 0; i < right.size(); i++)
    {
        _data[i] -= right._data[i];
    }
    return *this;
}


//not well perfomanced at all ;)
template<typename T, int _dim>
std::ostream &operator<<(std::ostream &os, const thensor<T, _dim> &th)
{
  for(int i = 0 ; i <th.size(); i++)
  {
      int k = 1;
      if (i != 0)
      {
          for (int j = _dim - 1; j >= 0, i % (k * th._shape[j]) == 0; --j)
          {
              os << '\n';
              k*=th._shape[j];
          }
      }
      os << th._data[i]<< ' ';
  }
  return os;
}

  template<typename T>
  thensor<T, 1> operator*(const thensor<T, 1> &left, const thensor<T, 2> &right)
  {
      assert(left.shape()[0] == right.shape()[0]);
      int outShape = right.shape()[1];
      int inShape = left.shape()[0];
      thensor<T,1> resVec(outShape);
      T*  res = resVec.data();
      const T* vec = left.data();
      const T* src = right.data();
      for (int j =0 ; j < outShape; j++)
      {
            res[j] = T();
      }
      for (int i =0; i < inShape; i++)
      {
          const T& t =vec[i];
          for (int j =0 ; j < outShape; j++,src++)
          {
              res[j]+=t*(*src);
          }
      }
      return resVec;
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
}

#endif //LIGHTY_VECTOR_HPP
