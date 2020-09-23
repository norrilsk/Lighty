#ifndef LIGHTY_DENSE_HPP
#define LIGHTY_DENSE_HPP

#include<Layers/Base.hpp>

//http://neuralnetworksanddeeplearning.com/chap2.html

template <typename T, typename S>
class Dense1D final : public Layers
{
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	Dense1D(int32_t inputSize, int32_t outputSize) { throw std::logic_error{ "Invalid template type. Use float instead." }; };
	~Dense1D() final = default;
};

template <>
class Dense1D<float, float> final : public Layers
{
	const fmat* _input_batch = nullptr; // need for backprop
	fmat _output_batch;
	fmat _delta_batch; // need for backprop
	int32_t _inputSize = 0;
	int32_t _outputSize = 0;
	fmat _weights;
	fvec _bias;
	std::unique_ptr<optim::Optimizer> _weights_optimizer = nullptr;
	std::unique_ptr<optim::Optimizer> _bias_optimizer = nullptr;
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	fmat weights() { return _weights; }
	fvec bias() { return _bias; }
	void set_weights(fmat& w) { _weights = w; };
	void set_bias(fvec& b) { _bias = b; };
	//sometimes we need to change optimizers on fly
	//set optimizers to nullptr to freeze layer
	void set_optimizers(std::unique_ptr<optim::Optimizer> weights_optimizer,
		std::unique_ptr<optim::Optimizer>  bias_optimizer)
	{
		_weights_optimizer = std::move(weights_optimizer);
		_bias_optimizer = std::move(bias_optimizer);
	};
	void set_optimizer(optim::optimizer_t type, float lr) final;
	void dump(std::ostream & output) final;
	Dense1D(int32_t inputSize, int32_t outputSize,
		std::unique_ptr<optim::Optimizer> weights_optimizer = std::make_unique<optim::SGD<fmat>>(),
		std::unique_ptr<optim::Optimizer>  bias_optimizer = std::make_unique<optim::SGD<fvec>>());
	~Dense1D() final = default;
};

void Dense1D<float, float>::dump(std::ostream & output)
{
	uint32_t lr_nm = static_cast<uint32_t>(LayersType::LAYER_DENSE1D);
	uint32_t type1 = LayerIOType2Int<real32_t>();
	uint32_t type2 = LayerIOType2Int<real32_t>();
	output.write(reinterpret_cast<char*>(&lr_nm), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&type1), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&type2), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&_inputSize), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&_outputSize), sizeof(uint32_t));

	const real32_t* data_w = _weights.data();
	output.write(reinterpret_cast<const char*>(data_w), _weights.size() * sizeof(real32_t));
	const real32_t* data_b = _bias.data();
	output.write(reinterpret_cast<const char*>(data_b), _bias.size() * sizeof(real32_t));
}
Dense1D<float, float> ::Dense1D(int32_t inputSize, int32_t outputSize,
	std::unique_ptr<optim::Optimizer> weights_optimizer,
	std::unique_ptr<optim::Optimizer>  bias_optimizer) :
	_inputSize(inputSize), _outputSize(outputSize), _weights({ inputSize,outputSize }),
	_bias(outputSize), _output_batch(), _delta_batch(), _weights_optimizer(std::move(weights_optimizer)),
	_bias_optimizer(std::move(bias_optimizer))
{
	// TODO: think about correct initialization (may be some articles)

	static std::default_random_engine generator(static_cast<unsigned>(time(nullptr)) + 42);
	std::normal_distribution<float> distribution{ 0.0f,0.01f };
	std::normal_distribution<float> distribution_b{ 0.005f,0.005f };
	for (int32_t i = 0; i <outputSize; i++)
	{
		_bias[i] = distribution_b(generator);
	}
	for (int32_t i = 0; i <inputSize; i++)
	{
		fvec &&row = _weights[i];
		for (int32_t j = 0; j < outputSize; j++)
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

const linal::Container& Dense1D<float, float>::forward(const linal::Container &src)
{
	const fmat* input_batch_tmp = dynamic_cast<const fmat *>(&src);
	fmat output_batch_tmp = linal::matmul((*input_batch_tmp), _weights);
	int32_t batch_size = input_batch_tmp->shape()[0];
	for (int32_t i = 0; i < batch_size; i++)
	{
		output_batch_tmp[i] += _bias;
	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = input_batch_tmp;
	_output_batch = output_batch_tmp;
	return dynamic_cast<const linal::Container&>(_output_batch);
}
const linal::Container& Dense1D<float, float>::backward(const linal::Container &delta)
{
	const fmat& deltaLower = dynamic_cast<const fmat&>(delta);
	int32_t batch_size = deltaLower.shape()[0];
	assert(batch_size == _input_batch->shape()[0]);
	fmat deltaUpper = linal::matmul(deltaLower, linal::transpose(_weights));


	double r_batch = 1. / batch_size;

	fvec grad_bias;
	fmat grad_weights;
	if (_bias_optimizer)
	{
		grad_bias = fvec(_bias.size());
		linal::zero_set(grad_bias);
		// computing average gradient on batch
		for (int32_t i = 0; i < batch_size; i++)
		{
			grad_bias += deltaLower[i];
		}
		grad_bias = -r_batch *grad_bias;

	}
	if (_weights_optimizer)
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

#endif