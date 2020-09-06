//
// Created by norrilsk on 12.10.19.
//

#ifndef LIGHTY_LAYERS_HPP
#define LIGHTY_LAYERS_HPP
#include <vector>
#include <random>
#include "linal/thensor.hpp"
#include "linal/algebra.hpp"
#include "Optimizers.hpp"
#include <chrono>
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
    virtual void set_optimizer(optim::optimizer_t type, float lr) {};
};

template <typename T, typename S>
class Conv2D final : public Layers
{
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	Conv2D(int inputChannels, std::tuple<int,int> kernelSize, int outputSize) { throw std::logic_error{ "Invalid template type. Use float instead." }; };
	~Conv2D() final = default;
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
    std::unique_ptr<optim::Optimizer> _weights_optimizer = nullptr;
    std::unique_ptr<optim::Optimizer> _bias_optimizer = nullptr;
public:
    const linal::Container& forward(const linal::Container &src ) final;
    const linal::Container& backward(const linal::Container &delta ) final;
    fmat weights(){return _weights;}
    fvec bias(){return _bias;}
    //sometimes we need to change optimizers on fly
    //set optimizers to nullptr to freeze layer
    void set_optimizers(std::unique_ptr<optim::Optimizer> weights_optimizer,
        std::unique_ptr<optim::Optimizer>  bias_optimizer)
        {
            _weights_optimizer = std::move(weights_optimizer);
            _bias_optimizer = std::move(bias_optimizer);
        };
    void set_optimizer(optim::optimizer_t type, float lr) final;
    Dense1D(int inputSize, int outputSize,
        std::unique_ptr<optim::Optimizer> weights_optimizer = std::make_unique<optim::SGD<fmat>>(),
        std::unique_ptr<optim::Optimizer>  bias_optimizer = std::make_unique<optim::SGD<fvec>>());
    ~Dense1D() final  = default;
};


Dense1D<float, float> ::Dense1D(int inputSize, int outputSize,
    std::unique_ptr<optim::Optimizer> weights_optimizer,
    std::unique_ptr<optim::Optimizer>  bias_optimizer):
_inputSize(inputSize), _outputSize(outputSize), _weights({inputSize,outputSize}),
_bias(outputSize),_output_batch(), _delta_batch() , _weights_optimizer(std::move(weights_optimizer)),
_bias_optimizer(std::move(bias_optimizer))
{
    // TODO: think about correct initialization (may be some articles)

    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 42 );
    std::normal_distribution<float> distribution{0.0f,0.01f};
    std::normal_distribution<float> distribution_b{0.005f,0.005f};
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
void Dense1D<float, float>::set_optimizer(optim::optimizer_t type, float lr)
{
    _weights_optimizer = std::move(optim::get_optimizer<fmat>(type, lr));
    _bias_optimizer = std::move(optim::get_optimizer<fvec>(type, lr));
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

    
    double r_batch = 1. /batch_size;
    
    fvec grad_bias;
    fmat grad_weights;
    if (_bias_optimizer)
    {
        grad_bias = fvec(_bias.size());
        linal::zero_set(grad_bias);
        // computing average gradient on batch
        for (int i = 0; i < batch_size; i++)
        {
            grad_bias += deltaLower[i];
        }
        grad_bias  = -r_batch *grad_bias;
        
    }
    if ( _weights_optimizer)
    {
        grad_weights = fmat(_weights.shape());
        linal::zero_set(grad_weights);
        // computing average gradient on batch
        grad_weights -= linal::matmul(linal::transpose(*_input_batch), deltaLower);
    }
    
    // ---------------------Calb line-----------------------------------------
    //there has to be but...
    if (_bias_optimizer)
        (*_bias_optimizer)(_bias, grad_bias);
    if (_weights_optimizer)
        (*_weights_optimizer)(_weights, grad_weights);
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
    const linal::thensor<T,2>* _input_batch = nullptr; // need for backprop
    linal::thensor<T,2> _output_batch;
    linal::thensor<T,2> _delta_batch; // need for backprop
public:
    const linal::Container& forward(const linal::Container &src) final;
    const linal::Container& backward(const linal::Container &delta) final;
    Relu1D():_output_batch(), _delta_batch() {};
    ~Relu1D() final  = default;
};

template <typename T>
const linal::Container& Relu1D<T, T> :: forward(const linal::Container &src)
{
    auto* input_batch_tmp = dynamic_cast<const linal::thensor<T,2> *>(&src);
    linal::thensor<T,2> output_batch_tmp(input_batch_tmp->shape());
    const T* p_src = input_batch_tmp->data();
    T* p_dst = output_batch_tmp.data();
    for(int i = 0; i < input_batch_tmp->size(); i++)
    {
        p_dst[i] = std::max(T(), p_src[i]);
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = input_batch_tmp;
    _output_batch = std::move(output_batch_tmp);
    return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T>
const linal::Container& Relu1D<T, T>::backward(const linal::Container &delta)
{
    const auto& deltaLower = dynamic_cast<const linal::thensor<T,2>&>(delta);
    assert(deltaLower.size() == _input_batch->size());
    linal::thensor<T,2> deltaUpper(deltaLower.shape());
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

template <typename T, typename S>
class Sigmoid1D final : Layers
{
private:
public:
    const linal::Container& forward(const linal::Container &src) override {};
    const linal::Container& backward(const linal::Container &delta ) override {};
    Sigmoid1D() { throw std::logic_error{"Invalid template type. Use same input and output instead."};};
    ~Sigmoid1D() final  = default;
};

template <typename T>
class Sigmoid1D<T, T> final : public Layers
{
private:
    const linal::thensor<T,2>* _input_batch = nullptr; // need for backprop
    linal::thensor<T,2> _output_batch;
    linal::thensor<T,2> _delta_batch; // need for backprop
public:
    const linal::Container& forward(const linal::Container &src) final;
    const linal::Container& backward(const linal::Container &delta) final;
    Sigmoid1D():_output_batch(), _delta_batch() {};
    ~Sigmoid1D() final  = default;
};

template <typename T>
const linal::Container& Sigmoid1D<T, T> :: forward(const linal::Container &src)
{
    const auto* input_batch_tmp = dynamic_cast<const linal::thensor<T,2> *>(&src);
    linal::thensor<T,2> output_batch_tmp(input_batch_tmp->shape());
    const T* p_src = input_batch_tmp->data();
    T* p_dst = output_batch_tmp.data();
    const T one = T() +1;
    for(int i = 0; i < input_batch_tmp->size(); i++)
    {
        p_dst[i] = one/(std::exp(-p_src[i]) + one);
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = input_batch_tmp;
    _output_batch = std::move(output_batch_tmp);
    return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T>
const linal::Container& Sigmoid1D<T, T>::backward(const linal::Container &delta)
{
    const auto& deltaLower = dynamic_cast<const linal::thensor<T,2>&>(delta);
    assert(deltaLower.size() == _input_batch->size());
    linal::thensor<T,2> deltaUpper(deltaLower.shape());
    const T* p_src = deltaLower.data();
    const T* p_img = _output_batch.data();
    T* p_dst = deltaUpper.data();
    T one = T() + 1;
    for(int i = 0; i < deltaLower.size(); i++)
    {
        p_dst[i] = p_src[i] * (p_img[i] * (one - p_img[i]));
        
    }
    // ---------------------Calb line-----------------------------------------
    _input_batch = nullptr;
    _delta_batch = deltaUpper;
    return dynamic_cast<const linal::Container&>(_delta_batch);
}
#endif //LIGHTY_LAYERS_HPP
