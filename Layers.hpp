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

class Layers
{
protected:
    bool training = true;

public:
    virtual const linal::Container& forward(const linal::Container &src ) = 0;
    //http://neuralnetworksanddeeplearning.com/chap2.html
    virtual const linal::Container& backward(const linal::Container &delta ) = 0;
    Layers() = default;
    virtual ~Layers() = default;
    void set_mode(TrainigMode mode){training = mode;}
};
template <typename T, typename S>
class Dense1D final : public Layers
{
public:
    const linal::Container& forward(const linal::Container &src ) override {};
    const linal::Container& backward(const linal::Container &delta ) override {};
    Dense1D(int inputSize, int outputSize) { throw std::logic_error{"Invalid template type. Use float instead."};};
    ~Dense1D() final  = default;
};

template <>
class Dense1D<float, float> final : public Layers
{
    const fmat* _input_batch = nullptr; // need for backprop
    fmat _output_batch;
    fmat _delta_batch; // need for backprop
    int _inputSize = 0;
    int _outputSize = 0;
    fmat _weights;
    fvec _bias;
public:
    const linal::Container& forward(const linal::Container &src ) final;
    const linal::Container& backward(const linal::Container &delta ) final;
    fmat weights(){return _weights;}
    fvec bias(){return _bias;}
    
    Dense1D(int inputSize, int outputSize);
    ~Dense1D() final  = default;
};


Dense1D<float, float> ::Dense1D(int inputSize, int outputSize) :
_inputSize(inputSize), _outputSize(outputSize), _weights({inputSize,outputSize}),
_bias(outputSize),_output_batch(), _delta_batch()
{
    // TODO:
    //dense_layer_number_that_garantee_non_repeateble_random_seed_for_random_engine

    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 42 );
    std::normal_distribution<float> distribution{0.0,0.01};
    std::normal_distribution<float> distribution_b{0.005,0.005};
    for(int i = 0 ; i <outputSize; i++)
    {
        _bias[i] = distribution_b(generator);
    }
    for(int i = 0 ; i <inputSize; i++)
    {
        fvec &&row = _weights[i];
        for (int j = 0; j < outputSize; j++)
        {
            row[j] = distribution(generator);
        }
    }
}

const linal::Container& Dense1D<float, float>::forward(const linal::Container &src )
{
    const fmat* input_batch_tmp = dynamic_cast<const fmat *>(&src);
    fmat output_batch_tmp = linal::matmul((*input_batch_tmp), _weights);
    int batch_size = input_batch_tmp->shape()[0];
    for ( int i = 0 ; i < batch_size; i++)
    {
        output_batch_tmp[i] += _bias;
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = input_batch_tmp;
    _output_batch = output_batch_tmp;
    return dynamic_cast<const linal::Container&>(_output_batch);
}
const linal::Container& Dense1D<float, float>::backward(const linal::Container &delta )
{
    const fmat& deltaLower = dynamic_cast<const fmat&>(delta);
    int batch_size = deltaLower.shape()[0];
    assert(batch_size == _input_batch->shape()[0]);
    fmat deltaUpper = linal::matmul(deltaLower, linal::transpose(_weights));
    // ---------------------Calb line-----------------------------------------
    
    //TODO: run optimizer here
    double r_batch = 1. /batch_size;
    for (int i = 0; i < batch_size; i++)
    {
        _bias += 1e-2 * r_batch * deltaLower[i];
        _weights += 1e-2 * r_batch * linal::matmul(deltaLower[i], (*_input_batch)[i]);
    }
    _input_batch = nullptr;
    _delta_batch = deltaUpper;
    return dynamic_cast<const linal::Container&>(_delta_batch);
}


template <typename T, typename S>
class Relu1D final : Layers
{
private:
public:
    const linal::Container& forward(const linal::Container &src) override {};
    const linal::Container& backward(const linal::Container &delta ) override {};
    Relu1D() { throw std::logic_error{"Invalid template type. Use same input and output instead."};};
    ~Relu1D() final  = default;
};

template <typename T>
class Relu1D<T, T> final : public Layers
{
private:
    const fmat* _input_batch = nullptr; // need for backprop
    fmat _output_batch;
    fmat _delta_batch; // need for backprop
public:
    const linal::Container& forward(const linal::Container &src) final;
    const linal::Container& backward(const linal::Container &delta) final;
    Relu1D():_output_batch(), _delta_batch() {};
    ~Relu1D() final  = default;
};

template <typename T>
const linal::Container& Relu1D<T, T> :: forward(const linal::Container &src)
{
    const fmat* input_batch_tmp = dynamic_cast<const fmat *>(&src);
    fmat output_batch_tmp(input_batch_tmp->shape());
    const T* p_src = input_batch_tmp->data();
    T* p_dst = output_batch_tmp.data();
    for(int i = 0; i < input_batch_tmp->size(); i++)
    {
        p_dst[i] = std::max(T(), p_src[i]);
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = input_batch_tmp;
    // FIXME:
    // there is copy constructor called
    // it is not i expected
    _output_batch = std::move(output_batch_tmp);
    return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T>
const linal::Container& Relu1D<T, T>::backward(const linal::Container &delta)
{
    const fmat& deltaLower = dynamic_cast<const fmat&>(delta);
    assert(deltaLower.size() == _input_batch->size());
    fmat deltaUpper(deltaLower.shape());
    const T* p_src = deltaLower.data();
    const T* p_img = _input_batch->data();
    T* p_dst = deltaUpper.data();
    for(int i = 0; i < deltaLower.size(); i++)
    {
        p_dst[i] = (p_img[i] > T()) ? p_src[i] : T();
        
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = nullptr;
    _delta_batch = deltaUpper;
    return dynamic_cast<const linal::Container&>(_delta_batch);
}
#endif //LIGHTY_LAYERS_HPP
