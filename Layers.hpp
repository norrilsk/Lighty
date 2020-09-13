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
#include <tuple>
typedef linal::thensor<float,1> fvec;
typedef linal::thensor<float,2> fmat;
typedef linal::thensor<float, 1> fthensor1;
typedef linal::thensor<float, 2> fthensor2;
typedef linal::thensor<float, 3> fthensor3;
typedef linal::thensor<float, 4> fthensor4;

typedef linal::thensor<int, 1> ivec;
typedef linal::thensor<int, 2> imat;
typedef linal::thensor<int, 1> ithensor1;
typedef linal::thensor<int, 2> ithensor2;
typedef linal::thensor<int, 3> ithensor3;
typedef linal::thensor<int, 4> ithensor4;

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
	Conv2D(int inputChannels, std::tuple<int, int> kernelSize, int outputSize, std::tuple<int, int> stride = { 1,1 },
		std::tuple<int, int> padding = { 0,0 }) { throw std::logic_error{ "Invalid template type. Use float instead." }; };
	~Conv2D() final = default;
};

template <>
class Conv2D<float, float> final : public Layers
{
	//const fmat* _input_batch = nullptr; 
	linal::thensor<float,4> _output_batch;
	fmat _unrolled_batch; // need for backprop
	fthensor4 _delta_batch;
	int _inputChannels = 0;
	int _outputChannels = 0;
	int _depth;
	fmat _weights;
	fvec _bias;
	std::vector<int> _input_shape;
	std::tuple<int, int> _stride;
	std::tuple<int, int> _padding;
	std::tuple<int, int> _kernelSize;

	std::unique_ptr<optim::Optimizer> _weights_optimizer = nullptr;
	std::unique_ptr<optim::Optimizer> _bias_optimizer = nullptr;
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	const fmat& weights() { return _weights; }
	fvec bias() { return _bias; }
	//sometimes we need to change optimizers on fly
	//set optimizers to nullptr to freeze layer
	void set_optimizers(std::unique_ptr<optim::Optimizer> weights_optimizer,
		std::unique_ptr<optim::Optimizer>  bias_optimizer)
	{
		_weights_optimizer = std::move(weights_optimizer);
		_bias_optimizer = std::move(bias_optimizer);
	};
	void set_optimizer(optim::optimizer_t type, float lr) final;
	Conv2D(int inputChannels, std::tuple<int, int> kernelSize, int outputChannels,
		std::tuple<int, int> stride = { 1,1 }, std::tuple<int, int> padding = { 0,0 },
		std::unique_ptr<optim::Optimizer> weights_optimizer = std::make_unique<optim::SGD<fmat>>(),
		std::unique_ptr<optim::Optimizer>  bias_optimizer = std::make_unique<optim::SGD<fvec>>());
	~Conv2D() final = default;
};


Conv2D<float, float> ::Conv2D(int inputChannels, std::tuple<int, int> kernelSize, int outputChannels,
	std::tuple<int, int> stride , std::tuple<int, int> padding,
	std::unique_ptr<optim::Optimizer> weights_optimizer,
	std::unique_ptr<optim::Optimizer>  bias_optimizer ) :
	_inputChannels(inputChannels), _kernelSize(kernelSize), _outputChannels(outputChannels),
	_stride(stride), _padding(padding),
	_weights_optimizer(std::move(weights_optimizer)),
	_bias_optimizer(std::move(bias_optimizer)),
	_depth(inputChannels * std::get<0>(kernelSize) * std::get<1>(kernelSize)),
	_weights({ _depth ,outputChannels }),
	_bias(outputChannels)
{
	// TODO: think about correct initialization (may be some articles)

	static std::default_random_engine generator(static_cast<unsigned>(time(nullptr)) + 42);
	std::normal_distribution<float> distribution{ 0.0f,0.01f };
	std::normal_distribution<float> distribution_b{ 0.005f,0.005f };
	for (int i = 0; i <outputChannels; i++)
	{
		_bias[i] = distribution_b(generator);
	}
	for (int i = 0; i < _depth; i++)
	{
		fvec &&row = _weights[i];
		for (int j = 0; j < outputChannels; j++)
		{
			row[j] = distribution(generator);
		}
	}
}
void Conv2D<float, float>::set_optimizer(optim::optimizer_t type, float lr)
{
	_weights_optimizer = std::move(optim::get_optimizer<fmat>(type, lr));
	_bias_optimizer = std::move(optim::get_optimizer<fvec>(type, lr));
}


const linal::Container& Conv2D<float, float>::forward(const linal::Container &src)
{
	const fthensor4* input_batch_tmp = dynamic_cast<const fthensor4 *>(&src);
	std::vector<int> kernel_shape = { _inputChannels, _outputChannels, std::get<0>(_kernelSize),
		std::get<1>(_kernelSize) };
	const int batch_size = input_batch_tmp->shape()[0];
	const int height = input_batch_tmp->shape()[1];
	const int width = input_batch_tmp->shape()[2];
	const int channels = input_batch_tmp->shape()[3];
	std::tuple<int, int, int> output_shape = linal::getConv2dOutputShape({ height, width,channels }, kernel_shape, _stride, _padding);
	const int output_channels = std::get<2>(output_shape);
	const int output_height = std::get<0>(output_shape);
	const int output_width = std::get<1>(output_shape);
	const int unrolled_image_width = channels * kernel_shape[2] * kernel_shape[3];
	const int unrolled_image_height = output_width * output_height;

	fthensor3 padded_image;
	fthensor3 unrolled_image({batch_size, unrolled_image_height ,unrolled_image_width });
	for (int i = 0; i < batch_size; i++)
	{
		padded_image = linal::padd_image((*input_batch_tmp)[i], std::get<0>(_padding), std::get<1>(_padding), 0.f);
		unrolled_image[i] = linal::unroll_image(padded_image, kernel_shape, std::get<0>(_stride), std::get<1>(_stride));

	}
	fmat unrolled_batch_tmp = linal::reshape<float, 3, 2>(unrolled_image, {unrolled_image_height * batch_size, unrolled_image_width});
	fmat result = linal::matmul(unrolled_batch_tmp, _weights);
	for (int i = 0; i < result.shape()[0]; i++)
	{
		result[i] += _bias;
	}
	fthensor4 output_batch = linal::reshape<float, 2, 4>(result, { batch_size, output_height, output_width, output_channels });
	// ---------------------Calb line-----------------------------------------
	 _unrolled_batch = unrolled_batch_tmp;
	 _input_shape = input_batch_tmp->shape();
	 return output_batch;

}
const linal::Container& Conv2D<float, float>::backward(const linal::Container &delta)
{
	const fthensor4& deltaLower = dynamic_cast<const fthensor4&>(delta);
	const int batch_size = deltaLower.shape()[0];
	assert(batch_size == _input_shape[0]);
	const int output_height = deltaLower.shape()[1];
	const int output_width = deltaLower.shape()[2];
	const int output_channels = deltaLower.shape()[3];
	std::vector<int> kernel_shape = { _inputChannels, _outputChannels, std::get<0>(_kernelSize),
		std::get<1>(_kernelSize) };
	const fmat delta2d = linal::reshape<float, 4, 2>(deltaLower, { batch_size * output_height * output_width ,output_channels });
	fvec grad_bias;
	fmat grad_weights;
	if (_bias_optimizer)
	{
		grad_bias = fvec(_bias.size());
		linal::zero_set(grad_bias);
		// computing average gradient on batch
		for (int i = 0; i <  batch_size * output_height * output_width; i++)
		{
			grad_bias += delta2d[i];
		}
		grad_bias = -grad_bias;
	}

	if (_weights_optimizer)
	{
		grad_weights = fmat(_weights.shape());
		linal::zero_set(grad_weights);
		// computing average gradient on batch
		grad_weights -= linal::matmul(linal::transpose(_unrolled_batch), delta2d);
	}
	fmat unrolled_delta = linal::matmul(linal::transpose(_weights), delta2d); // need formal prove
	fthensor4 result(_input_shape);
	std::vector<int> input_shape(_input_shape.begin() + 1, _input_shape.end());
	for (int i = 0; i < batch_size; i++)
	{
		result[i] = linal::bacward_unroll_image(unrolled_delta, kernel_shape,
			input_shape, std::get<0>(_stride), std::get<1>(_stride),
			std::get<0>(_padding), std::get<1>(_padding));
	}
	// ---------------------Calb line-----------------------------------------
	//there has to be but... optimizers..
	if (_bias_optimizer)
		(*_bias_optimizer)(_bias, grad_bias);
	if (_weights_optimizer)
		(*_weights_optimizer)(_weights, grad_weights);
	_delta_batch = result;

	return dynamic_cast<const linal::Container&>(_delta_batch);
}


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
