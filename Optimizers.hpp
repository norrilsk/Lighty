//
// Created by norrilsk on 18.01.20.
//
//https://en.wikipedia.org/wiki/Stochastic_gradient_descent
#ifndef LIGHTY_OPTIMIZERS_HPP
#define LIGHTY_OPTIMIZERS_HPP
#include "linal/thensor.hpp"
namespace optim
{
  
  enum optimizer_t
  {
      OPTIMIZER_SGD,
      OPTIMIZER_MOMENTUM, // Nesterov momentum
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
      //https://habr.com/ru/post/318970/
      float _alpha = 0.9;
      T speed;
  public:
      explicit Momentum(float lr = 1e-3 ,float alpha = 0.9) : Optimizer(lr), _alpha(alpha) {};
      linal::Container &operator()(linal::Container &src, const linal::Container &grad_src) final;
  };
  
  template<typename T>
  linal::Container &SGD<T>::operator()(linal::Container &src, const linal::Container &grad_src)
  {
      T &x = dynamic_cast<T &>(src);
      const T &grad_x = dynamic_cast<const T &>(grad_src);
      x = x - lr * grad_x;
      return x;
  }
 
  template<typename T>
  linal::Container &Momentum<T>::operator()(linal::Container &src, const linal::Container &grad_src)
  {
      T &x = dynamic_cast<T &>(src);
      const T &grad_x = dynamic_cast<const T &>(grad_src);
      T cur_speed;
      if (x.size() != speed.size())
      {
          cur_speed = T(x.shape());
          linal::zero_set<typename T:: d_type>(cur_speed);
      }
      else
          cur_speed = speed;
      T next_speed = _alpha * cur_speed  +  lr * grad_x;
      x = x - next_speed;
      // ---------------------Calb line-----------------------------------------
      speed = std::move(next_speed);
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
          case OPTIMIZER_NONE:
              return nullptr;
          default:
              assert(false && "wrong optimizer type ");
      }
      return std::unique_ptr<Optimizer>();
  }
  
}
#endif //LIGHTY_OPTIMIZERS_HPP
