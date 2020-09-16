#ifndef LIGHTY_ACTIVATIONS_HPP
#define LIGHTY_ACTIVATIONS_HPP

#include<Layers/Base.hpp>

template <typename T, typename S, int _dim>
class Relu final : Layers
{
private:
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	Relu() { throw std::logic_error{ "Invalid template type. Use same input and output instead." }; };
	~Relu() final = default;
};

template <typename T, typename S, int _dim>
class Sigmoid final : Layers
{
private:
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	Sigmoid() { throw std::logic_error{ "Invalid template type. Use same input and output instead." }; };
	~Sigmoid() final = default;
};

template <typename T, typename S, int _dim>
class Tanh final : Layers
{
private:
public:
	const linal::Container& forward(const linal::Container &src) override {};
	const linal::Container& backward(const linal::Container &delta) override {};
	Tanh() { throw std::logic_error{ "Invalid template type. Use same input and output instead." }; };
	~Tanh() final = default;
};

template <typename T, typename S>
using Relu2D = Relu<T, S, 4>;
template <typename T, typename S>
using Relu1D = Relu<T, S, 2>;
template <typename T, typename S>
using Sigmoid2D = Sigmoid<T, S, 4>;
template <typename T, typename S>
using Sigmoid1D = Sigmoid<T, S, 2>;
template <typename T, typename S>
using Tanh2D = Tanh<T, S, 4>;
template <typename T, typename S>
using Tanh1D = Tanh<T, S, 2>;

template <typename T, int _dim>
class Relu<T, T, _dim> final : public Layers
{
private:
	const linal::thensor<T, _dim>* _input_batch = nullptr; // need for backprop
	linal::thensor<T, _dim> _output_batch;
	linal::thensor<T, _dim> _delta_batch; // need for backprop
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	Relu() :_output_batch(), _delta_batch() {};
	~Relu() final = default;
};




template <typename T, int _dim>
class Sigmoid<T, T, _dim> final : public Layers
{
private:
	const linal::thensor<T, _dim>* _input_batch = nullptr; // need for backprop
	linal::thensor<T, _dim> _output_batch;
	linal::thensor<T, _dim> _delta_batch; // need for backprop
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	Sigmoid() :_output_batch(), _delta_batch() {};
	~Sigmoid() final = default;
};

template <typename T, int _dim>
class Tanh<T, T, _dim> final : public Layers
{
private:
	const linal::thensor<T, _dim>* _input_batch = nullptr; // need for backprop
	linal::thensor<T, _dim> _output_batch;
	linal::thensor<T, _dim> _delta_batch; // need for backprop
public:
	const linal::Container& forward(const linal::Container &src) final;
	const linal::Container& backward(const linal::Container &delta) final;
	Tanh() :_output_batch(), _delta_batch() {};
	~Tanh() final = default;
};



template <typename T, int _dim>
const linal::Container& Relu<T, T, _dim> ::forward(const linal::Container &src)
{
	auto* input_batch_tmp = dynamic_cast<const linal::thensor<T, _dim> *>(&src);
	linal::thensor<T, _dim> output_batch_tmp(input_batch_tmp->shape());
	const T* p_src = input_batch_tmp->data();
	T* p_dst = output_batch_tmp.data();
	for (int i = 0; i < input_batch_tmp->size(); i++)
	{
		p_dst[i] = std::max(T(), p_src[i]);
	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = input_batch_tmp;
	_output_batch = std::move(output_batch_tmp);
	return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T, int _dim>
const linal::Container& Relu<T, T, _dim>::backward(const linal::Container &delta)
{
	const auto& deltaLower = dynamic_cast<const linal::thensor<T, _dim>&>(delta);
	assert(deltaLower.size() == _input_batch->size());
	linal::thensor<T, _dim> deltaUpper(deltaLower.shape());
	const T* p_src = deltaLower.data();
	const T* p_img = _input_batch->data();
	T* p_dst = deltaUpper.data();
	for (int i = 0; i < deltaLower.size(); i++)
	{
		p_dst[i] = (p_img[i] > T()) ? p_src[i] : T();

	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = nullptr;
	_delta_batch = deltaUpper;
	return dynamic_cast<const linal::Container&>(_delta_batch);
}


template <typename T, int _dim>
const linal::Container& Sigmoid<T, T, _dim> ::forward(const linal::Container &src)
{
	const auto* input_batch_tmp = dynamic_cast<const linal::thensor<T, _dim> *>(&src);
	linal::thensor<T, _dim> output_batch_tmp(input_batch_tmp->shape());
	const T* p_src = input_batch_tmp->data();
	T* p_dst = output_batch_tmp.data();
	const T one = T() + 1;
	for (int i = 0; i < input_batch_tmp->size(); i++)
	{
		p_dst[i] = one / (std::exp(-p_src[i]) + one);
	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = input_batch_tmp;
	_output_batch = std::move(output_batch_tmp);
	return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T, int _dim>
const linal::Container& Sigmoid<T, T, _dim>::backward(const linal::Container &delta)
{
	const auto& deltaLower = dynamic_cast<const linal::thensor<T, _dim>&>(delta);
	assert(deltaLower.size() == _input_batch->size());
	linal::thensor<T, _dim> deltaUpper(deltaLower.shape());
	const T* p_src = deltaLower.data();
	const T* p_img = _output_batch.data();
	T* p_dst = deltaUpper.data();
	T one = T() + 1;
	for (int i = 0; i < deltaLower.size(); i++)
	{
		p_dst[i] = p_src[i] * (p_img[i] * (one - p_img[i]));

	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = nullptr;
	_delta_batch = deltaUpper;
	return dynamic_cast<const linal::Container&>(_delta_batch);
}


template <typename T, int _dim>
const linal::Container& Tanh<T, T, _dim> ::forward(const linal::Container &src)
{
	const auto* input_batch_tmp = dynamic_cast<const linal::thensor<T, _dim> *>(&src);
	linal::thensor<T, _dim> output_batch_tmp(input_batch_tmp->shape());
	const T* p_src = input_batch_tmp->data();
	T* p_dst = output_batch_tmp.data();
	const T one = T() + 1;
	for (int i = 0; i < input_batch_tmp->size(); i++)
	{
		T e_mx = std::exp(-p_src[i]);
		T e_x = one / e_mx;
		p_dst[i] = (e_x - e_mx) / (e_x + e_mx);
	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = input_batch_tmp;
	_output_batch = std::move(output_batch_tmp);
	return dynamic_cast<const linal::Container&>(_output_batch);
}

template <typename T, int _dim>
const linal::Container& Tanh<T, T, _dim>::backward(const linal::Container &delta)
{
	const auto& deltaLower = dynamic_cast<const linal::thensor<T, _dim>&>(delta);
	assert(deltaLower.size() == _input_batch->size());
	linal::thensor<T, _dim> deltaUpper(deltaLower.shape());
	const T* p_src = deltaLower.data();
	const T* p_img = _output_batch.data();
	T* p_dst = deltaUpper.data();
	T one = T() + 1;
	for (int i = 0; i < deltaLower.size(); i++)
	{
		p_dst[i] = p_src[i] * (one - p_img[i] * p_img[i]) ;

	}
	// ---------------------Calb line-----------------------------------------
	_input_batch = nullptr;
	_delta_batch = deltaUpper;
	return dynamic_cast<const linal::Container&>(_delta_batch);
}

#endif // !LIGHTY_ACTIVATIONS_HPP
