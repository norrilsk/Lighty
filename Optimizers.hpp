//
// Created by norrilsk on 18.01.20.
//

#ifndef LIGHTY_OPTIMIZERS_HPP
#define LIGHTY_OPTIMIZERS_HPP
#include "linal/thensor.hpp"
namespace optim
{
  
  enum optimizer_t
  {
      OPTIMIZER_SGD,
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
      SGD(float lr = 1e-3) : Optimizer(lr) {};
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
  std::unique_ptr<Optimizer> get_optimizer(optimizer_t type, float lr)
  {
      switch(type)
      {
          case OPTIMIZER_SGD:
              return std::make_unique<SGD<T> >(lr);
          case OPTIMIZER_NONE:
              return nullptr;
          default:
              assert(false && "wrong optimizer type ");
      }
      return std::unique_ptr<Optimizer>();
  }
  
}
#endif //LIGHTY_OPTIMIZERS_HPP
