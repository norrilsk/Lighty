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
    Loss() = default;
    virtual ~Loss() = default;
    virtual T  operator()(linal::thensor<R,1>prediction,linal::thensor<S,1> label) = 0;
    virtual linal::thensor<R,1> grad(linal::thensor<R,1>prediction,linal::thensor<S,1> label) = 0;
};

template <typename T, typename S>
class MSE :public Loss<T,S,S>
{
public:
    ~MSE() = default;
    MSE() = default;
    T  operator()(linal::thensor<S,1>prediction,linal::thensor<S,1> label) final;
    linal::thensor<S,1> grad(linal::thensor<S,1>prediction,linal::thensor<S,1> label) final;
};
template<typename T, typename S>
T MSE<T, S>::operator()(linal::thensor<S, 1> prediction, linal::thensor<S, 1> label)
{
    linal::thensor<S,1> delta = prediction - label;
    return static_cast<T>(delta.dot(delta));
}
template<typename T, typename S>
linal::thensor<S, 1> MSE<T, S>::grad(linal::thensor<S, 1> prediction, linal::thensor<S, 1> label)
{
    return 2*(label - prediction);
}

#endif //LIGHTY_LOSSES_HPP
