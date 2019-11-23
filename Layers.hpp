//
// Created by norrilsk on 12.10.19.
//

#ifndef LIGHTY_LAYERS_HPP
#define LIGHTY_LAYERS_HPP
#include <vector>
#include <random>
#include "linal/thensor.hpp"
typedef linal::thensor<float,1> fvec;
typedef linal::thensor<float,2> fmat;

enum TrainigMode
{
    Production = 0,
    Trainig = 1
};
template <typename T, typename S>
class Layers
{
protected:
    bool training = true;
    linal::thensor<T,1> input;
public:
    virtual void forward(const linal::thensor<T,1> &src, linal::thensor<S,1> &dst) = 0;
    //http://neuralnetworksanddeeplearning.com/chap2.html
    virtual void backward(const linal::thensor<S,1> &deltaLower, linal::thensor<T,1> &deltaUpper) = 0;
    Layers(): input(){};
    virtual ~Layers() = default;
    void set_mode(TrainigMode mode){training = mode;}
};
template <typename T, typename S>
class Dense1D final : Layers<T,S>
{
public:
    void forward(const linal::thensor<T,1> &src, linal::thensor<S,1> &dst){};
    void backward(const linal::thensor<T,1> &src, linal::thensor<S,1> &dst){};
    Dense1D(int inputSize, int outputSize) { throw std::logic_error{"Invalid template type. Use float instead."};};
    ~Dense1D() final  = default;
};

template <>
class Dense1D<float, float> final : public Layers<float,float>
{
    int inputSize = 0;
    int outputSize = 0;
    fmat _weights;
    fvec _bias;
public:
    void forward(const fvec &src, fvec &dst) final;
    void backward(const fvec &deltaLower, fvec &deltaUpper) final;
    fmat weights(){return _weights;}
    fvec bias(){return _bias;}
    
    Dense1D(int inputSize, int outputSize);
    ~Dense1D() final  = default;
};


Dense1D<float, float> ::Dense1D(int inputSize, int outputSize) :
inputSize(inputSize), outputSize(outputSize), _weights({outputSize,inputSize}),
_bias({outputSize})
{
    //dense_layer_number_that_garantee_non_repeateble_random_seed_for_random_engine
    static unsigned denseID =0;
    std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 42 + denseID);
    std::normal_distribution<float> distribution{0.0,0.01};
    std::normal_distribution<float> distribution_b{0.005,0.005};
    int size = outputSize*inputSize;
    for(int i = 0 ; i <outputSize; i++)
    {
        _bias[i] = distribution_b(generator);
        fvec&& row = _weights[i];
        for (int j = 0 ; j <inputSize;j++)
        {
            row[j] = distribution(generator);
        }
    }
    denseID++;
}
// we do not want to allocate dst each time forward is called
void Dense1D<float, float> ::forward(const fvec &src, fvec &dst)
{
    if (training)
        input = src;
    dst = _weights*src + _bias;
}

void Dense1D<float, float>::backward(const fvec &deltaLower, fvec &deltaUpper)
{
    deltaUpper = linal::transpose(_weights)*deltaLower;
    //TODO: run optimizer here
    _bias += 1e-2 * deltaLower;
    _weights += 1e-2 * linal::matmul(input, deltaLower);
}


template <typename T, typename S>
class Relu1D final : Layers<T,S>
{
public:
    void forward(const linal::thensor<T,1> &src, linal::thensor<S,1> &dst){};
    void backward(const linal::thensor<T,1> &src, linal::thensor<S,1> &dst){};
    Relu1D() { throw std::logic_error{"Invalid template type. Use float instead."};};
    ~Relu1D() final  = default;
};

template <typename T>
class Relu1D<T, T> final : public Layers<T,T>
{
public:
    void forward(const linal::thensor<T,1>  &src, linal::thensor<T,1> &dst) final;
    void backward(const linal::thensor<T,1> &deltaLower, linal::thensor<T,1> &deltaUpper);
    Relu1D() = default;
    ~Relu1D() final  = default;
};

template <typename T>
void Relu1D<T, T> ::forward(const linal::thensor<T,1> &src, linal::thensor<T,1> &dst)
{
    if (this->training)
        this->input = src;
    for(int i = 0; i < src.size(); i++)
    {
        dst[i] = std::max(T(), src[i]);
    }
}

template <typename T>
void Relu1D<T, T>::backward(const linal::thensor<T,1> &deltaLower, linal::thensor<T,1> &deltaUpper)
{
    for(int i = 0; i < deltaLower.size(); i++)
    {
        if(this->input[i] > 0)
        {
            deltaUpper[i] = deltaLower[i];
        }
        else
        {
            deltaUpper[i] = T();
        }
    }
}
#endif //LIGHTY_LAYERS_HPP
