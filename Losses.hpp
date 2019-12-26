//
// Created by norrilsk on 28.10.19.
//

#ifndef LIGHTY_LOSSES_HPP
#define LIGHTY_LOSSES_HPP
#include "linal/thensor.hpp"
template <typename T, typename R, typename S>
class Loss
{
public:
    typedef T returnType;
    Loss() = default;
    virtual ~Loss() = default;
    virtual T  operator()(const linal::thensor<R,2>& prediction,const linal::thensor<S,2>& label) = 0;
    virtual linal::thensor<R,2> grad(const linal::thensor<R,2>& prediction,const linal::thensor<S,2>& label) = 0;
};

template <typename T, typename S>
class MSE :public Loss<T,S,S>
{
public:
    ~MSE() = default;
    MSE() = default;
    T  operator()(const linal::thensor<S,2>& prediction,const linal::thensor<S,2>& label) final;
    linal::thensor<S,2> grad(const linal::thensor<S,2>& prediction, const linal::thensor<S,2>& label) final;
};
template<typename T, typename S>
T MSE<T, S>::operator()(const linal::thensor<S, 2>& prediction, const linal::thensor<S, 2>& label)
{
    linal::thensor<S,2> delta = prediction - label;
    int batch_size = delta.shape()[0];
    S res = 0;
    for (int i = 0 ; i < batch_size ; i++)
    {
        res+= delta[i].dot(delta[i]);
    }
    
    return static_cast<T>(res)/batch_size;
}
template<typename T, typename S>
linal::thensor<S, 2> MSE<T, S>::grad(const linal::thensor<S, 2>& prediction, const linal::thensor<S, 2>& label)
{
    return 2*(label - prediction);
}

#endif //LIGHTY_LOSSES_HPP
