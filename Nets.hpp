//
// Created by norrilsk on 07.12.19.
//

#ifndef LIGHTY_NETS_HPP
#define LIGHTY_NETS_HPP
#include<string>
#include<vector>
#include<memory>
#include<random>
#include<algorithm>
#include<functional>
#include"Layers.hpp"
#include <iostream>
#include <chrono>
#include "Optimizers.hpp"

enum NetType
{
	NET_NONE = 0,
	NET_SEQUENTIONAL =0x1
};
enum CompressionType
{
	COMPRESSION_NONE = 0 //:)
};

class Sequential
{
private:
    
    std::vector<std::unique_ptr<Layers>> layers;
public:
    template <typename S, typename T, typename...  Args>
    void addRelu1D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addRelu2D(Args... args);
    template <typename S, typename T, typename...  Args>
    void addSigmoid1D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addSigmoid2D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addTanh1D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addTanh2D(Args... args);
    template <typename S, typename T, typename...  Args>
    void addDense1D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addConv2D(Args... args);
	template <typename S, typename T, typename...  Args>
	void addFlattern4to2(Args... args);
	void addLayer(std::unique_ptr<Layers> layer) { layers.emplace_back(std::move(layer)); };
	void dump(std::ostream& stream);
	void load(std::istream& stream);
    template<typename T,  typename T2, typename S>
    void train(const T & data, T2& labels, int32_t batch_size, int32_t epochs = 100, bool verbose = false,
        std::function<void(T &)> aug = [](T& _data){}, const S& loss_function  = S());
    template<typename T, typename Tret>
    Tret predict(const T & data);
    template<typename T, typename Tret>
    Tret predict_batch(const T & data);
    //set common optimizer for whole net
    void set_optimizers(optim::optimizer_t type, float lr);
    Sequential() = default;
    ~Sequential() = default;
};

template<typename S, typename T, typename... Args>
void Sequential::addRelu1D(Args... args)
{
    layers.emplace_back( new Relu1D<S,T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addRelu2D(Args... args)
{
	layers.emplace_back(new Relu2D<S, T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addSigmoid1D(Args... args)
{
    layers.emplace_back( new Sigmoid1D<S,T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addSigmoid2D(Args... args)
{
	layers.emplace_back(new Sigmoid2D<S, T>(args...));
}


template<typename S, typename T, typename... Args>
void Sequential::addTanh1D(Args... args)
{
	layers.emplace_back(new Tanh1D<S, T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addTanh2D(Args... args)
{
	layers.emplace_back(new Tanh2D<S, T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addDense1D(Args... args)
{
    layers.emplace_back( new Dense1D<S,T>(args...));
}


template<typename S, typename T, typename... Args>
void Sequential::addConv2D(Args... args)
{
	layers.emplace_back(new Conv2D<S, T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addFlattern4to2(Args... args)
{
	layers.emplace_back(new Flattern4to2<S, T>(args...));
}


template<typename T,  typename T2, typename S>
void Sequential::train(const T & data, T2& labels, int32_t batch_size,
    int32_t epochs, bool verbose,std::function<void(T &)> aug, const S& loss_function)
{
    std::vector<int32_t> batch_shape = data.shape();
    std::vector<int32_t> label_shape = labels.shape();
    assert(batch_shape[0] == label_shape[0]);
    assert(batch_shape.size() > 1);
    int32_t N = batch_shape[0];
    batch_shape[0] = batch_size;
    label_shape[0] = batch_size;
    T batch(batch_shape);
    T2 label_batch(label_shape);
    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 1189);
    std::vector<int32_t> indexes(N);
    for (int32_t i = 0 ; i < N; i++)
        indexes[i] = i;
    
    for (int32_t i = 0 ; i < epochs; i++)
    {
        auto loss  = decltype(loss_function(label_batch,label_batch))();
        auto start = std::chrono::high_resolution_clock::now();
        std::shuffle(indexes.begin(), indexes.end(), generator);
        int32_t j = 0;
        for (j = 0 ; j <= N - batch_size; j+= batch_size)
        {
            for (int32_t k = 0; k < batch_size; k++)
            {
                label_batch[k] = labels[indexes[j + k]];
                batch[k] = data[indexes[j + k]].copy();
            }
            const linal::Container* x = &(dynamic_cast<const linal::Container&>(batch));
            
            for (auto& layer : layers )
            {
                x = &(layer->forward(*x));
            }
            const T2& answer = dynamic_cast<const T2&>(*x);
            
            if (verbose)
                loss+= loss_function(answer, label_batch);
            
            auto&& grad = loss_function.grad(answer, label_batch);
            
            const linal::Container* delta = &(dynamic_cast<const linal::Container&>(grad));
            for (auto it  = layers.rbegin(); it < layers.rend(); it++)
            {
                auto& layer = *it;
                delta = &(layer->backward(*delta));
            }
        }
        if (verbose)
        {
            auto end = std::chrono::high_resolution_clock::now();
            long dif = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "epoch # " << i << "\t loss = " << loss / (N / batch_size) << " (" << dif <<" ms)"<< std::endl;
        }
    }
}
template<typename T, typename Tret>
Tret Sequential::predict_batch(const T &data)
{
    std::vector<int32_t> batch_shape = data.shape();
    assert(batch_shape.size() > 1);
    int32_t N = batch_shape[0];
    int32_t batch_size = batch_shape[0];
    T batch(batch_shape);

    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 1189);
    std::vector<int32_t> indexes(N);
    for (int32_t i = 0 ; i < N; i++)
        indexes[i] = i;
    
    int32_t j = 0;
    
    for (int32_t k = 0; k < batch_size; k++)
    {
        batch[k] = data[indexes[j + k]].copy();
    }
    const linal::Container* x = &(dynamic_cast<const linal::Container&>(batch));
    
    for (auto& layer : layers )
    {
        x = &(layer->forward(*x));
    }
    const Tret& answer = dynamic_cast<const Tret&>(*x);
    return answer;
}
void Sequential::dump(std::ostream & stream)
{
	uint32_t ns = static_cast<uint32_t>(NET_SEQUENTIONAL);
	uint32_t comp = static_cast<uint32_t>(COMPRESSION_NONE);
	uint32_t lsz = static_cast<uint32_t>(layers.size());
	stream.write(reinterpret_cast<char*>(&ns), sizeof(uint32_t));
	stream.write(reinterpret_cast<char*>(&comp), sizeof(uint32_t));
	stream.write(reinterpret_cast<char*>(&lsz), sizeof(uint32_t));

	for (auto& layer : layers)
	{
		layer->dump(stream);
	}
}

void Sequential::set_optimizers(optim::optimizer_t type, float lr)
{
    for (auto& layer : layers )
    {
        layer->set_optimizer(type,lr);
    }
}

#define CommonActivation_m(network,stream,layer_num,activation1D,activation2D) \
{\
	uint32_t input_type;\
	uint32_t output_type;\
	uint32_t dim;\
	stream.read(reinterpret_cast<char*>(&input_type), sizeof(uint32_t));\
	stream.read(reinterpret_cast<char*>(&output_type), sizeof(uint32_t));\
	stream.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));\
	if (input_type == LayerIOType::LTYPE_FLOAT && output_type == LayerIOType::LTYPE_FLOAT)\
	{\
		switch (dim)\
		{\
		case 2:\
			network.activation1D<real32_t, real32_t>();\
			break;\
		case 4:\
			network.activation2D<real32_t, real32_t>();\
			break;\
		default:\
			throw std::runtime_error("Unsupported dim  in layer #" + std::to_string(layer_num));\
		}\
	}\
	else\
	{\
		throw std::runtime_error("Unsupported IOtype in layer #" + std::to_string(layer_num));\
	}\
}


void Sequential::load(std::istream & stream)
{
	uint32_t nt;
	uint32_t compression;
	uint32_t layers_num;
	uint32_t layer_type;
	Sequential tmp_net;
	stream.read(reinterpret_cast<char*>(&nt), sizeof(uint32_t));
	if (static_cast<NetType>(nt) != NetType::NET_SEQUENTIONAL)
		throw std::runtime_error("Wrong Network Type");
	stream.read(reinterpret_cast<char*>(&compression), sizeof(uint32_t));
	if (static_cast<CompressionType>(compression) != CompressionType::COMPRESSION_NONE)
		throw std::runtime_error("Unsupported compression type");
	stream.read(reinterpret_cast<char*>(&layers_num), sizeof(uint32_t));
	for (int i = 0; i < layers_num; i++)
	{
		stream.read(reinterpret_cast<char*>(&layer_type), sizeof(uint32_t));
		switch (layer_type)
		{
		case LayersType::LAYER_RELU:
			CommonActivation_m(tmp_net, stream, i, addRelu1D, addRelu2D);
		    break;
		case LayersType::LAYER_SIGMOID:
			CommonActivation_m(tmp_net, stream, i, addSigmoid1D, addSigmoid2D);
			break;
		case LayersType::LAYER_TANH:
			CommonActivation_m(tmp_net, stream, i, addTanh1D, addTanh2D);
			break;
		case LayersType::LAYER_DENSE1D:
			{
				uint32_t input_type, output_type;
				int32_t input_size, output_size;
				stream.read(reinterpret_cast<char*>(&input_type), sizeof(uint32_t)); 
				stream.read(reinterpret_cast<char*>(&output_type), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&input_size), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&output_size), sizeof(int32_t));

				if (input_type == LayerIOType::LTYPE_FLOAT && output_type == LayerIOType::LTYPE_FLOAT)
				{
					std::unique_ptr<Dense1D<real32_t, real32_t>> dense(new Dense1D<real32_t, real32_t> (input_size, output_size));
					fmat weights({ input_size,output_size });
					real32_t* data_w = weights.data();
					stream.read(reinterpret_cast<char*>(data_w), weights.size() * sizeof(real32_t));
					dense->set_weights(weights);
					fvec bias(output_size);
					real32_t* data_b = bias.data();
					stream.read(reinterpret_cast<char*>(data_b), bias.size() * sizeof(real32_t));
					dense->set_bias(bias);
					tmp_net.addLayer(std::move(dense));
				}
				else
				{
					throw std::runtime_error("Unsupported IOtype in layer #" + std::to_string(i));
				}

		}
		break;
		case LayersType::LAYER_CONV2D:
			{
				uint32_t input_type, output_type;
				int32_t input_channels, output_channels;
				int32_t kernel_x, kernel_y, stride_x, stride_y, padding_x, padding_y;

				stream.read(reinterpret_cast<char*>(&input_type), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&output_type), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&input_channels), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&kernel_x), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&kernel_y), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&output_channels), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&stride_x), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&stride_y), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&padding_x), sizeof(int32_t));
				stream.read(reinterpret_cast<char*>(&padding_y), sizeof(int32_t));
				if (input_type == LayerIOType::LTYPE_FLOAT && output_type == LayerIOType::LTYPE_FLOAT)
				{
					int32_t depth = input_channels * kernel_x * kernel_y;
					std::unique_ptr<Conv2D<real32_t, real32_t>> conv2d(new Conv2D<real32_t, real32_t>(input_channels,
					{ kernel_x,kernel_y }, output_channels, { stride_x , stride_y }, { padding_x , padding_y }));
					fmat weights({ depth,output_channels });
					real32_t* data_w = weights.data();
					stream.read(reinterpret_cast<char*>(data_w), weights.size() * sizeof(real32_t));
					conv2d->set_weights(weights);
					fvec bias(output_channels);
					real32_t* data_b = bias.data();
					stream.read(reinterpret_cast<char*>(data_b), bias.size() * sizeof(real32_t));
					conv2d->set_bias(bias);
					tmp_net.addLayer(std::move(conv2d));
				}
				else
				{
					throw std::runtime_error("Unsupported IOtype in layer #" + std::to_string(i));
				}

			}
		break;
		case LayersType::LAYER_FLATTERN4TO2:
			{
				uint32_t input_type;
				uint32_t output_type;
				stream.read(reinterpret_cast<char*>(&input_type), sizeof(uint32_t));
				stream.read(reinterpret_cast<char*>(&output_type), sizeof(uint32_t));
				if (input_type == LayerIOType::LTYPE_FLOAT && output_type == LayerIOType::LTYPE_FLOAT)
				{

					tmp_net.addFlattern4to2<real32_t, real32_t>();

				}
				else
				{
					throw std::runtime_error("Unsupported IOtype in layer #" + std::to_string(i));
				}
			}
			break;
		default:
			throw std::runtime_error("Unsupported Layer Type in layer #" + std::to_string(i));
			break;
		}
	}

	// ---------------------Calb line-----------------------------------------
	this->layers = std::move(tmp_net.layers);
}


#endif //LIGHTY_NETS_HPP
