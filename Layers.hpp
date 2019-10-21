//
// Created by norrilsk on 12.10.19.
//

#ifndef LIGHTY_LAYERS_HPP
#define LIGHTY_LAYERS_HPP
#include <vector>
#include <random>
#include "linal/thensor.hpp"
template <typename T, typename S>
class Layers
{
public:
    virtual void forward(const std::vector<T> &src, std::vector<S> &dst) = 0;
    virtual void backward(const std::vector<T> &src, std::vector<S> &dst) = 0;
    Layers() =default;
    virtual ~Layers() = default;
};
template <typename T, typename S>
class Dense1D final : Layers<T,S>
{
public:
    void forward(const std::vector<T> &src, std::vector<S> &dst){};
    void backward(const std::vector<T> &src, std::vector<S> &dst){};
    Dense1D(unsigned inputSize, unsigned outputSize) { throw std::logic_error{"Invalid template type. Use float instead."};};
    ~Dense1D() final  = default;
};

template <>
class Dense1D<float, float> final : Layers<float,float>
{
    unsigned inputSize = 0;
    unsigned outputSize = 0;
    fmat weights;
public:
    void forward(const fvec &src, fvec &dst) final;
    void backward(const fvec &src, fvec &dst) final;
    Dense1D(unsigned inputSize, unsigned outputSize);
    ~Dense1D() final  = default;
};


Dense1D<float, float> ::Dense1D(unsigned inputSize, unsigned outputSize) :
inputSize(inputSize), outputSize(outputSize)
{
    weights.resize(outputSize,fvec(inputSize));
    std::default_random_engine generator{ };
    std::normal_distribution<float> distribution{0,0.2};
    for( auto &it : weights)
    {
        for (auto &v : it)
        {
            v = distribution(generator);
        }
    }
}
// we do not want to allocate dst each time forward is called
void Dense1D<float, float> ::forward(const fvec &src, fvec &dst)
{

}
#endif //LIGHTY_LAYERS_HPP
