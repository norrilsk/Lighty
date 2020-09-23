#ifndef LIGHTY_CONV2D_HPP
#define LIGHTY_CONV2D_HPP

#include<Layers/Base.hpp>

template <typename T, typename S>
class Conv2D final : public Layers
{
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	virtual void dump(std::ostream & output) = 0;
	Conv2D(int32_t inputChannels, std::tuple<int32_t, int32_t> kernelSize, int32_t outputSize, std::tuple<int32_t, int32_t> stride = { 1,1 },
		std::tuple<int32_t, int32_t> padding = { 0,0 }) {
		throw std::logic_error{ "Invalid template type. Use float instead." };
	};
	~Conv2D() final = default;
};

template <typename T, typename S>
class Flattern4to2 final : public Layers
{
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	virtual void dump(std::ostream & output) final;
	Flattern4to2() { throw std::logic_error{ "Invalid template type. Use float instead." }; };
	~Flattern4to2() final = default;
};


template <>
class Conv2D<float, float> final : public Layers
{
	//const fmat* _input_batch = nullptr; 
	fthensor4 _output_batch;
	fmat _unrolled_batch; // need for backprop
	fthensor4 _delta_batch;
	int32_t _inputChannels = 0;
	int32_t _outputChannels = 0;
	int32_t _depth;
	fmat _weights;
	fvec _bias;
	std::vector<int32_t> _input_shape;
	std::tuple<int32_t, int32_t> _stride;
	std::tuple<int32_t, int32_t> _padding;
	std::tuple<int32_t, int32_t> _kernelSize;
	std::unique_ptr<optim::Optimizer> _weights_optimizer = nullptr;
	std::unique_ptr<optim::Optimizer> _bias_optimizer = nullptr;
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	const fmat& weights() { return _weights; }
	fvec bias() { return _bias; }
	void set_bias(fvec& b) { _bias = b; }
	void set_weights(fmat& w) { _weights = w; }
	//sometimes we need to change optimizers on fly
	//set optimizers to nullptr to freeze layer
	void set_optimizers(std::unique_ptr<optim::Optimizer> weights_optimizer,
		std::unique_ptr<optim::Optimizer>  bias_optimizer)
	{
		_weights_optimizer = std::move(weights_optimizer);
		_bias_optimizer = std::move(bias_optimizer);
	};
	void set_optimizer(optim::optimizer_t type, float lr) final;
	virtual void dump(std::ostream & output) final;
	Conv2D(int32_t inputChannels, std::tuple<int32_t, int32_t> kernelSize, int32_t outputChannels,
		std::tuple<int32_t, int32_t> stride = { 1,1 }, std::tuple<int32_t, int32_t> padding = { 0,0 },
		std::unique_ptr<optim::Optimizer> weights_optimizer = std::make_unique<optim::SGD<fmat>>(),
		std::unique_ptr<optim::Optimizer>  bias_optimizer = std::make_unique<optim::SGD<fvec>>());
	~Conv2D() final = default;
};



template <>
class Flattern4to2<float, float> final : public Layers
{
	const fthensor4* _input_batch = nullptr; // need for backprop
	fmat _output_batch;
	fthensor4 _delta_batch;
	std::vector<int32_t> _saved_shape;
public:
	virtual void dump(std::ostream & output) final
	{
		uint32_t lr_nm = static_cast<uint32_t>(LayersType::LAYER_FLATTERN4TO2);
		uint32_t type1 = LayerIOType2Int<float>();
		uint32_t type2 = LayerIOType2Int<float>();
		output.write(reinterpret_cast<char*>(&lr_nm), sizeof(uint32_t));
		output.write(reinterpret_cast<char*>(&type1), sizeof(uint32_t));
		output.write(reinterpret_cast<char*>(&type2), sizeof(uint32_t));
	}
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;

	Flattern4to2() {};
	~Flattern4to2() final = default;
};

Conv2D<float, float> ::Conv2D(int32_t inputChannels, std::tuple<int32_t, int32_t> kernelSize, int32_t outputChannels,
	std::tuple<int32_t, int32_t> stride, std::tuple<int32_t, int32_t> padding,
	std::unique_ptr<optim::Optimizer> weights_optimizer,
	std::unique_ptr<optim::Optimizer>  bias_optimizer) :
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
	for (int32_t i = 0; i <outputChannels; i++)
	{
		_bias[i] = distribution_b(generator);
	}
	for (int32_t i = 0; i < _depth; i++)
	{
		fvec &&row = _weights[i];
		for (int32_t j = 0; j < outputChannels; j++)
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

void Conv2D<float, float>::dump(std::ostream & output)
{
	uint32_t lr_nm = static_cast<uint32_t>(LayersType::LAYER_CONV2D);
	uint32_t type1 = LayerIOType2Int<real32_t>();
	uint32_t type2 = LayerIOType2Int<real32_t>();
	uint32_t ker_x = std::get<0>(_kernelSize), ker_y = std::get<1>(_kernelSize);
	uint32_t stride_x = std::get<0>(_stride), stride_y = std::get<1>(_stride);
	uint32_t pad_x = std::get<0>(_padding), pad_y = std::get<1>(_padding);
	output.write(reinterpret_cast<char*>(&lr_nm), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&type1), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&type2), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&_inputChannels), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&ker_x), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&ker_y), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&_outputChannels), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&stride_x), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&stride_y), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&pad_x), sizeof(uint32_t));
	output.write(reinterpret_cast<char*>(&pad_y), sizeof(uint32_t));
	const real32_t* data_w = _weights.data();
	output.write(reinterpret_cast<const char*>(data_w), _weights.size() * sizeof(real32_t));
	const real32_t* data_b = _bias.data();
	output.write(reinterpret_cast<const char*>(data_b), _bias.size() * sizeof(real32_t));
}

const linal::Container& Conv2D<float, float>::forward(const linal::Container &src)
{
	const fthensor4* input_batch_tmp = dynamic_cast<const fthensor4 *>(&src);
	std::vector<int32_t> kernel_shape = { _inputChannels, _outputChannels, std::get<0>(_kernelSize),
		std::get<1>(_kernelSize) };
	const int32_t batch_size = input_batch_tmp->shape()[0];
	const int32_t height = input_batch_tmp->shape()[1];
	const int32_t width = input_batch_tmp->shape()[2];
	const int32_t channels = input_batch_tmp->shape()[3];
	std::tuple<int32_t, int32_t, int32_t> output_shape = linal::getConv2dOutputShape({ height, width,channels }, kernel_shape, _stride, _padding);
	const int32_t output_channels = std::get<2>(output_shape);
	const int32_t output_height = std::get<0>(output_shape);
	const int32_t output_width = std::get<1>(output_shape);
	const int32_t unrolled_image_width = channels * kernel_shape[2] * kernel_shape[3];
	const int32_t unrolled_image_height = output_width * output_height;

	fthensor3 padded_image;
	fthensor3 unrolled_image({ batch_size, unrolled_image_height ,unrolled_image_width });
	for (int32_t i = 0; i < batch_size; i++)
	{
		padded_image = linal::padd_image((*input_batch_tmp)[i], std::get<0>(_padding), std::get<1>(_padding), 0.f);
		unrolled_image[i] = linal::unroll_image(padded_image, kernel_shape, std::get<0>(_stride), std::get<1>(_stride));

	}
	fmat unrolled_batch_tmp = linal::reshape<float, 3, 2>(unrolled_image, { unrolled_image_height * batch_size, unrolled_image_width });
	fmat result = linal::matmul(unrolled_batch_tmp, _weights);
	for (int32_t i = 0; i < result.shape()[0]; i++)
	{
		result[i] += _bias;
	}
	fthensor4 output_batch = linal::reshape<float, 2, 4>(result, { batch_size, output_height, output_width, output_channels });
	// ---------------------Calb line-----------------------------------------
	_unrolled_batch = std::move(unrolled_batch_tmp);
	_input_shape = input_batch_tmp->shape();
	_output_batch = std::move(output_batch);
	return _output_batch;

}


const linal::Container& Conv2D<float, float>::backward(const linal::Container &delta)
{
	const fthensor4& deltaLower = dynamic_cast<const fthensor4&>(delta);
	const int32_t batch_size = deltaLower.shape()[0];
	assert(batch_size == _input_shape[0]);
	const int32_t output_height = deltaLower.shape()[1];
	const int32_t output_width = deltaLower.shape()[2];
	const int32_t output_channels = deltaLower.shape()[3];
	std::vector<int32_t> kernel_shape = { _inputChannels, _outputChannels, std::get<0>(_kernelSize),
		std::get<1>(_kernelSize) };
	const fmat delta2d = linal::reshape<float, 4, 2>(deltaLower, { batch_size * output_height * output_width ,output_channels });
	fvec grad_bias;
	fmat grad_weights;
	if (_bias_optimizer)
	{
		grad_bias = fvec(_bias.size());
		linal::zero_set(grad_bias);
		// computing average gradient on batch
		for (int32_t i = 0; i < batch_size * output_height * output_width; i++)
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
	fmat unrolled_delta = linal::matmul(delta2d, linal::transpose(_weights)); // need formal prove
	float*  unrolled_delta_d = unrolled_delta.data();
	fthensor4 result(_input_shape);
	const int32_t step = unrolled_delta.shape()[1] * unrolled_delta.shape()[0] / batch_size;
	std::vector<int32_t> input_shape(_input_shape.begin() + 1, _input_shape.end());
	for (int32_t i = 0; i < batch_size; i++, unrolled_delta_d += step)
	{
		fthensor2 unrolled_delta_image;
		unrolled_delta_image.wrap(unrolled_delta_d, { unrolled_delta.shape()[0] / batch_size , unrolled_delta.shape()[1] });
		result[i] = linal::backward_unroll_image(unrolled_delta_image, kernel_shape,
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

const linal::Container& Flattern4to2<float, float>::forward(const linal::Container &src)
{
	const fthensor4* input_batch_tmp = dynamic_cast<const fthensor4 *>(&src);

	fmat _reshaped_tmp = linal::reshape<float, 4, 2>(*input_batch_tmp, { input_batch_tmp->shape()[0], input_batch_tmp->shape()[1] * input_batch_tmp->shape()[2] * input_batch_tmp->shape()[3] });
	// ---------------------Calb line-----------------------------------------
	_output_batch = std::move(_reshaped_tmp);
	_saved_shape = input_batch_tmp->shape();
	return dynamic_cast<const linal::Container&>(_output_batch);

}
const linal::Container& Flattern4to2<float, float>::backward(const linal::Container &delta)
{
	const fmat* input_batch_tmp = dynamic_cast<const fmat *>(&delta);
	fthensor4 delta_batch_tmp = linal::reshape<float, 2, 4>(*input_batch_tmp, _saved_shape);
	// ---------------------Calb line-----------------------------------------
	_delta_batch = std::move(delta_batch_tmp);
	return dynamic_cast<const linal::Container&>(_delta_batch);
}

template<typename T, typename S>
void Flattern4to2<T, S>::dump(std::ostream & output)
{
	output << static_cast<uint32_t>(LayersType::LAYER_FLATTERN4TO2)
		<< LayerIOType2Int<T>() << LayerIOType2Int<S>();
}


#endif // !LIGHTY_CONV2D_HPP


