//
// Created by norrilsk on 18.01.20.
//
//https://en.wikipedia.org/wiki/Stochastic_gradient_descent
//https://habr.com/ru/post/318970/
#ifndef LIGHTY_OPTIMIZERS_HPP
#define LIGHTY_OPTIMIZERS_HPP
#include "linal/thensor.hpp"
#include <cmath>
//Be patient optimizers change src data
namespace optim
{
  
  enum optimizer_t
  {
      OPTIMIZER_SGD,
      OPTIMIZER_MOMENTUM, 
	  OPTIMIZER_ADAGRAD,
	  OPTIMIZER_ADADELTA,
	  OPTIMIZER_RMSPROP,
	  OPTIMIZER_ADAM,
      OPTIMIZER_NONE
  };
  class Optimizer;
  
  template <typename T>
  std::unique_ptr<Optimizer> get_optimizer(optimizer_t type, float lr);
  
  class Optimizer
  {
  protected:
      float lr;
  public:
      explicit Optimizer(float lr) : lr(lr) {};
      virtual linal::Container &operator()(linal::Container &src, const linal::Container &grad_src) = 0;
  };
  
  template<typename T>
  class SGD : public Optimizer
  {
  private:
  
  public:
      explicit SGD(float lr = 1e-3) : Optimizer(lr) {};
      linal::Container &operator()(linal::Container &src, const linal::Container &grad_src) final;
  };
  
  template<typename T>
  class Momentum : public Optimizer
  {
  private:
      float _alpha = 0.9;
      T _speed;
  public:
      explicit Momentum(float lr = 1e-3 ,float alpha = 0.9) : Optimizer(lr), _alpha(alpha) {};
      linal::Container &operator()(linal::Container &src, const linal::Container &grad_src) final;
  };
  
  template<typename T>
  class Adagrad : public Optimizer
  {
  private:
	  //Smoothing parameter required to avoid division by 0.
      float _e = 0.9;
	  typename T::d_type _G = typename T::d_type();
  public:
      explicit Adagrad(float lr = 1e-3 ,float e = 1e-6) : Optimizer(lr), _e(e) {};
      linal::Container &operator()(linal::Container &src, const linal::Container &grad_src) final;
  };

  template<typename T>
  class Adadelta : public Optimizer
  {
  private:
	  float _alpha = 0.9;
	  float _e = 0.9;
	  typename T::d_type _mean_grad_square = typename T::d_type();
	  typename T::d_type  _prev_increment_mean_square = typename T::d_type();
  public:
	  explicit Adadelta(float lr = 1e-3, float alpha = 0.9, float e = 1e-6) : Optimizer(lr), _alpha(alpha), _e(e) {};
	  linal::Container& operator()(linal::Container& src, const linal::Container& grad_src) final;
  };

  template<typename T>
  class RMSProp : public Optimizer
  {
  private:
	  float _alpha = 0.9;
	  float _e = 0.9;
	  typename T::d_type _mean_grad_square = typename T::d_type();
  public:
	  explicit RMSProp(float lr = 1e-3, float alpha = 0.9, float e = 1e-6) : Optimizer(lr), _alpha(alpha), _e(e) {};
	  linal::Container& operator()(linal::Container& src, const linal::Container& grad_src) final;
  };

  template<typename T>
  linal::Container &SGD<T>::operator()(linal::Container &src, const linal::Container &grad_src)
  {
      T &x = dynamic_cast<T &>(src);
      const T &grad_x = dynamic_cast<const T &>(grad_src);
      x = x - lr * grad_x;
      return x;
  }

/*
* gradient depends on alpha
* square gradient depends on beta 
* _acum_(alpha/beta) contain (_alpha/_beta)^(iteration)	
*/
  template<typename T>
  class Adam : public Optimizer
  {
  private:
	  float _alpha = 0.9;
	  float _beta = 0.9;
	  float _acum_alpha, _acum_beta;
	  float _e = 1e-6;
	  int _iteration = 1;
	  T _mean_grad;
	  typename T::d_type _mean_square_grad = typename T::d_type();
  public:
	  explicit Adam(float lr = 1e-3, float alpha = 0.9, float beta = 0.9, float e = 1e-6) : Optimizer(lr), _alpha(alpha) , _beta(beta), _e(e)
	  {
		  _acum_alpha = _alpha; 
		  _acum_beta = _beta;
	  };
	  linal::Container& operator()(linal::Container& src, const linal::Container& grad_src) final;
  };

 
  template<typename T>
  linal::Container &Momentum<T>::operator()(linal::Container &src, const linal::Container &grad_src)
  {
      T &x = dynamic_cast<T &>(src);
      const T &grad_x = dynamic_cast<const T &>(grad_src);
      T cur_speed;
      if (x.size() != _speed.size())
      {
          cur_speed = T(x.shape());
          linal::zero_set<typename T:: d_type>(cur_speed);
      }
      else
          cur_speed = _speed;
      T next_speed = _alpha * cur_speed  +  lr * grad_x;
      x = x - next_speed;
      // ---------------------Calb line-----------------------------------------
      _speed = std::move(next_speed);
	  return src;
  }
  
  template<typename T>
  linal::Container& Adagrad<T>::operator()(linal::Container& src, const linal::Container& grad_src)
  {
	  T& x = dynamic_cast<T&>(src);
	  const T& grad_x = dynamic_cast<const T&>(grad_src);
	  typename T::d_type next_G = _G + grad_x.dot(grad_x);
	  x = x - lr * 1.f/std::sqrt(next_G + _e ) * grad_x;
	  // ---------------------Calb line-----------------------------------------
	  _G = std::move(next_G);
	  return src;
  }

  template<typename T>
  linal::Container& Adadelta<T>::operator()(linal::Container& src, const linal::Container& grad_src)
  {
	  T& x = dynamic_cast<T&>(src);
	  const T& grad_x = dynamic_cast<const T&>(grad_src);
	  typename T::d_type next_mean_grad = _mean_grad_square * _alpha + (1 - _alpha) * grad_x.dot(grad_x);
	  typename T::d_type RMS_grad = std::sqrt(next_mean_grad + _e);
	  typename T::d_type increment_mean_square;
	  typename T::d_type RMS_prev_incr = std::sqrt(_prev_increment_mean_square + _e);
	  T delta = 1.f * RMS_prev_incr / RMS_grad * grad_x;
	  increment_mean_square = _prev_increment_mean_square * _alpha + (1 - _alpha) * delta.dot(delta);
	  
	  x = x - delta;
	  // ---------------------Calb line-----------------------------------------
	  _mean_grad_square = std::move(next_mean_grad);
	  _prev_increment_mean_square = std::move(increment_mean_square);
	  return src;
  }

  template<typename T>
  linal::Container& RMSProp<T>::operator()(linal::Container& src, const linal::Container& grad_src)
  {
	  T& x = dynamic_cast<T&>(src);
	  const T& grad_x = dynamic_cast<const T&>(grad_src);
	  typename T::d_type next_mean = _mean_grad_square * _alpha + (1 - _alpha) * grad_x.dot(grad_x);
	  x = x - lr * 1.f / std::sqrt(next_mean + _e) * grad_x;
	  // ---------------------Calb line-----------------------------------------
	  _mean_grad_square = std::move(next_mean);
	  return src;
  }

  template<typename T>
  linal::Container& Adam<T>::operator()(linal::Container& src, const linal::Container& grad_src)
  {
	  T& x = dynamic_cast<T&>(src);
	  const T& grad_x = dynamic_cast<const T&>(grad_src);
	  T cur_mean_grad;
	  if (grad_x.size() != _mean_grad.size())
	  {
		  cur_mean_grad = T(grad_x.shape());
		  linal::zero_set<typename T::d_type>(cur_mean_grad);
	  }
	  else
		  cur_mean_grad = _mean_grad;

	  T next_mean_grad = _alpha * cur_mean_grad + (1 - _alpha) * grad_x;
	  typename T::d_type next_mean_square_grad = _beta * _mean_square_grad + (1 - _beta) * grad_x.dot(grad_x);
	  if (_iteration < 10)
	  {
		  next_mean_grad = 1.f / (1 - _acum_alpha) * next_mean_grad;
	  }
	  if (_iteration < 200)
	  {
		  next_mean_square_grad = next_mean_square_grad * 1.f / (1 - _acum_beta);
	  }
	  x -= lr * 1.f / std::sqrt(next_mean_square_grad +_e) * next_mean_grad;
	  // ---------------------Calb line-----------------------------------------
	  _iteration++;
	  _mean_grad = std::move(next_mean_grad);
	  _mean_square_grad = std::move(next_mean_square_grad);
	  _acum_alpha *= _alpha;
	  _acum_beta *= _beta;
	  return src;
  }


  template<typename T>
  std::unique_ptr<Optimizer> get_optimizer(optimizer_t type, float lr)
  {
      switch(type)
      {
          case OPTIMIZER_SGD:
              return std::make_unique<SGD<T> >(lr);
          case OPTIMIZER_MOMENTUM:
              return std::make_unique<Momentum<T> >(lr);
		  case OPTIMIZER_ADAGRAD:
			  return std::make_unique<Adagrad<T> >(lr);
		  case OPTIMIZER_ADADELTA:
			  return std::make_unique<Adadelta<T> >(lr);
		  case OPTIMIZER_RMSPROP:
			  return std::make_unique<RMSProp<T> >(lr);
		  case OPTIMIZER_ADAM:
			  return std::make_unique<Adam<T> >(lr);
          case OPTIMIZER_NONE:
              return nullptr;
          default:
              assert(false && "wrong optimizer type ");
      }
      return std::unique_ptr<Optimizer>();
  }
  
}
#endif //LIGHTY_OPTIMIZERS_HPP
