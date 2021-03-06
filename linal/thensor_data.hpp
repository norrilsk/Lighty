#ifndef LIGHTY_THENSOR_DATA_HPP
#define LIGHTY_THENSOR_DATA_HPP
#include <vector>
#include <cassert>
namespace linal
{
  template<typename T, int32_t dim>
  class ThensorData
  {
  private:
      T *_data = nullptr;
      T *_allocated_data = nullptr;
      std::vector<int32_t> _shapes;
      int32_t _size = 0;
  public:
      inline int32_t size() const noexcept { return _size; } ;
      inline const T *data() const noexcept { return _data; }
      inline T *data() noexcept { return _data; }
      inline const std::vector<int32_t>& shape() const noexcept{ return _shapes;};
	  
	  ThensorData<T, dim> &copy(const ThensorData<T, dim>& thensor_data);
      ThensorData<T, dim> copy() const;
      ThensorData<T, dim> &operator=(const ThensorData<T, dim> &right) &;
	  ThensorData<T, dim> &operator=(const ThensorData<T, dim> &right) &&;
	  ThensorData<T, dim> &operator=(ThensorData<T, dim>&& right) & noexcept;
	  ThensorData<T, dim> &operator=(ThensorData<T, dim>&& right) &&;
      ThensorData<T, dim - 1> operator[](int32_t idx) const;
      //this method initialized data from external source
      //it means that data will not be allocated or free-ed inside
      void construct(T *data,const std::vector<int32_t> &shapes);
	  ThensorData<T, dim> &reshape(const std::vector<int32_t> &new_shape);
      ThensorData() = default;
      explicit ThensorData(const std::vector<int32_t> &shapes);
      ThensorData(const ThensorData<T, dim> &thensor_data);
      ThensorData(ThensorData<T, dim> &&thensor_data) noexcept;
      ~ThensorData();
  };
  template<typename T>
  class ThensorData<T, 1>
  {
  private:
      T *_data = nullptr;
      T *_allocated_data = nullptr;
      std::vector<int32_t> _shapes;
      int32_t _size = 0;
  public:
      int32_t size() const { return _size; };
      T *data() const { return _data; }
      inline const std::vector<int32_t>& shape() const noexcept{ return _shapes;};

	  ThensorData<T, 1> &copy(const ThensorData<T, 1>& thensor_data);
      ThensorData<T, 1> copy() const;
      ThensorData<T, 1> &operator=(const ThensorData<T, 1> &right) &&;
      ThensorData<T, 1> &operator=(const ThensorData<T, 1> &right) &;
	  ThensorData<T, 1> &operator=(ThensorData<T, 1> &&right) & noexcept;
	  ThensorData<T, 1> &operator=(ThensorData<T, 1> &&right) &&;
      T &operator[](int32_t idx) const;
      void construct(T *data, const std::vector<int32_t> &shapes);
      ThensorData() = default;
      explicit ThensorData(const std::vector<int32_t> &shapes);
      ThensorData(const ThensorData<T, 1> &thensor_data);
      ThensorData(ThensorData<T, 1> &&thensor_data) noexcept;
      ~ThensorData();
  };
  
  template<typename T, int32_t dim>
  ThensorData<T, dim>::~ThensorData()
  {
      delete[] _allocated_data;
  }
  
  template<typename T, int32_t dim>
  ThensorData<T, dim>::ThensorData(const std::vector<int32_t> &shapes)
  {
      assert(dim == shapes.size());
      std::vector<int32_t> tmp = shapes;
      int32_t size = 1;
      for (int32_t i = 0; i < dim; i++)
      {
          assert(shapes[i] > 0);
          size *= shapes[i];
      }
      T *tmp_data = new T[size];
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _size = size;
      _allocated_data = _data;
      _shapes = std::move(tmp);
  }
  
  template<typename T, int32_t dim>
  ThensorData<T, dim>::ThensorData(const ThensorData<T, dim> &thensor_data)
  {
      std::vector<int32_t> tmp_shapes = thensor_data._shapes;
	  delete[] _allocated_data;
      int32_t size = thensor_data._size;
      T *tmp_data = new T[size];
      std::copy(thensor_data._data, thensor_data._data + size, tmp_data);
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _allocated_data = _data;
      _size = size;
      _shapes = std::move(tmp_shapes);
  }
  template<typename T, int32_t dim>
  ThensorData<T, dim>::ThensorData(ThensorData<T, dim> &&thensor_data) noexcept
  {
	  delete[] _allocated_data;
      _data = thensor_data._data;
      _allocated_data = thensor_data._allocated_data;
      _size = thensor_data._size;
      _shapes = std::move(thensor_data._shapes);
      thensor_data._allocated_data = nullptr;
  }
  
  template<typename T, int32_t dim>
  ThensorData<T, dim> &ThensorData<T, dim>::operator=(const ThensorData<T, dim> &right) &
  {
      
      if (this == &right)
          return *this;
	  delete[] _allocated_data;
      std::vector<int32_t> tmp_shapes = right._shapes;
      int32_t size = right._size;
      T *tmp_data = new T[size];
      std::copy(right._data, right._data + size, tmp_data);
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _allocated_data = _data;
      _size = size;
      _shapes = std::move(tmp_shapes);
      return *this;
  }

  template<typename T, int32_t dim>
  ThensorData<T, dim>& ThensorData<T, dim>::operator=(const ThensorData<T, dim>& right)&&
  {
	  
	  return this->copy(right);
  }

  template<typename T, int32_t dim>
  ThensorData<T, dim> &ThensorData<T, dim>::operator=(ThensorData<T, dim> &&right) & noexcept 
  {
      if (this == &right)
          return *this;
	  delete[] _allocated_data;
      _data = right._data;
      _allocated_data = right._allocated_data;
      _size = right._size;
      _shapes = std::move(right._shapes);
      right._allocated_data = nullptr;
      return *this;
  }

  template<typename T, int32_t dim>
  ThensorData<T, dim> &ThensorData<T, dim>::operator=(ThensorData<T, dim> &&right) && 
  {
	  return this->copy(right);
  }
  template<typename T, int32_t dim>
  ThensorData<T, dim - 1> ThensorData<T, dim>::operator[](int32_t idx) const
  {
      assert(idx < _size);
      std::vector<int32_t> tmp_shapes(dim - 1ul);
      for (int32_t i = 0; i < dim - 1; i++)
      {
          tmp_shapes[i] = _shapes[i + 1];
      }
      ThensorData<T, dim - 1> tmp;
      tmp.construct(_data + _size / _shapes[0] * idx, tmp_shapes);
      return tmp;
  }
  
  template<typename T, int32_t dim>
  void ThensorData<T, dim>::construct( T *data, const std::vector<int32_t> &shapes)
  {
      int32_t size = 1;
      for (auto s: shapes)
          size*=s;
      _data = data;
      _shapes = shapes;
      _size = size;
  }
  template<typename T, int32_t dim>
  ThensorData<T, dim> &ThensorData<T, dim>::reshape(const std::vector<int32_t> &new_shape)
  {
	  size_t new_size = 1;
	  for (auto& sh : new_shape)
	  {
		  new_size *= sh;
	  }
	  assert(new_size == this->size());
	  _shapes = new_shape;
	  return *this;
  }
  template<typename T, int32_t dim>
  ThensorData<T, dim> & ThensorData<T, dim>::copy(const ThensorData<T, dim> &thensor_data)
  {
	  assert(thensor_data.size() == this->size());
	  assert(thensor_data.shape() == this->shape());

	  std::copy(thensor_data.data(), thensor_data.data() + thensor_data.size(), this->data());
	  return *this;
  }

  template<typename T, int32_t dim>
  ThensorData<T, dim> ThensorData<T, dim>::copy() const
  {
      ThensorData<T, dim> tmp(*this);
      return tmp;
  }
  
  template<typename T>
  T &ThensorData<T, 1>::operator[](int32_t idx) const
  {
      assert(idx < _size);
      return _data[idx];
  }
  
  template<typename T>
  ThensorData<T, 1>::~ThensorData()
  {
      delete[] _allocated_data;
  }
  
  template<typename T>
  ThensorData<T, 1>::ThensorData(const std::vector<int32_t> &shapes)
  {
      assert(1 == shapes.size());
      std::vector<int32_t> tmp = shapes;
	  delete[] _allocated_data;
      int32_t size = 1;
      for (int32_t i = 0; i < 1; i++)
      {
          assert(shapes[i] > 0);
          size *= shapes[i];
      }
      T *tmp_data = new T[size];
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _size = size;
      _allocated_data = _data;
      _shapes = std::move(tmp);
  }
  
  template<typename T>
  ThensorData<T, 1>::ThensorData(const ThensorData<T, 1> &thensor_data)
  {
      std::vector<int32_t> tmp_shapes = thensor_data._shapes;
      int32_t size = thensor_data._size;
      T *tmp_data = new T[size];
      std::copy(thensor_data._data, thensor_data._data + size, tmp_data);
	  delete[] _allocated_data;
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _allocated_data = _data;
      _size = size;
      _shapes = std::move(tmp_shapes);
  }
  template<typename T>
  ThensorData<T, 1>::ThensorData(ThensorData<T, 1> &&thensor_data) noexcept
  {
	  delete[] _allocated_data;
      _data = thensor_data._data;
      _allocated_data = thensor_data._allocated_data;
      _size = thensor_data._size;
      _shapes = std::move(thensor_data._shapes);
      thensor_data._allocated_data = nullptr;
  }

 
  template<typename T>
  ThensorData<T, 1> &ThensorData<T, 1>::operator=(const ThensorData<T, 1> &right) &
  {
      
      if (this == &right)
          return *this;
	  delete[] _allocated_data;
      std::vector<int32_t> tmp_shapes = right._shapes;
      int32_t size = right._size;
      T *tmp_data = new T[size];
      std::copy(right._data, right._data + size, tmp_data);
      //Kalb line
      //---------------------------------------------------------------
      _data = tmp_data;
      _allocated_data = _data;
      _size = size;
      _shapes = std::move(tmp_shapes);
      return *this;
  }

  template<typename T>
  ThensorData<T, 1>& ThensorData<T, 1>::operator=(const ThensorData<T, 1>& right) &&
  {
	  return this->copy(right);
  }

  template<typename T>
  ThensorData<T, 1> &ThensorData<T, 1>::operator=(ThensorData<T, 1> &&right) & noexcept
  {
      if (this == &right)
          return *this;
	  delete[] _allocated_data;
      _data = right._data;
      _allocated_data = right._allocated_data;
      _size = right._size;
      _shapes = std::move(right._shapes);
      right._allocated_data = nullptr;
      return *this;
  }

  template<typename T>
  ThensorData<T, 1>& ThensorData<T, 1>::operator=(ThensorData<T, 1>&& right) &&
  {
	  return this->copy(right);
  }
  template<typename T>
  void ThensorData<T, 1>::construct(T *data, const std::vector<int32_t> &shapes)
  {
      int32_t size = 1;
      for (auto s: shapes)
          size*=s;
      _data = data;
      _shapes = shapes;
      _size = size;
  }

  template<typename T>
  ThensorData<T,1> &ThensorData<T, 1>::copy(const ThensorData<T,1>& thensor_data)
  {
	  assert(thensor_data.size() == this->size());
	  assert(thensor_data.shape() == this->shape());

	  std::copy(thensor_data.data(), thensor_data.data() + thensor_data.size(), this->data());
	  return *this;
  }
  template<typename T>
  ThensorData<T, 1> ThensorData<T, 1>::copy() const
  {
      ThensorData<T, 1> tmp(*this);
      return tmp;
  }
  
  
  
}
#endif //LIGHTY_THENSOR_DATA_HPP
