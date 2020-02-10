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
#include "Optimizers.hpp"
class Sequential
{
private:
    
    std::vector<std::unique_ptr<Layers>> layers;
public:
    template <typename S, typename T, typename...  Args>
    void addRelu1D(Args... args);
    template <typename S, typename T, typename...  Args>
    void addSigmoid1D(Args... args);
    template <typename S, typename T, typename...  Args>
    void addDense1D(Args... args);
    template<typename T,  typename T2, typename S>
    void train(const T & data, T2& labels, int batch_size, int epochs = 100, bool verbose = false,
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
void Sequential::addSigmoid1D(Args... args)
{
    layers.emplace_back( new Sigmoid1D<S,T>(args...));
}

template<typename S, typename T, typename... Args>
void Sequential::addDense1D(Args... args)
{
    layers.emplace_back( new Dense1D<S,T>(args...));
}

template<typename T,  typename T2, typename S>
void Sequential::train(const T & data, T2& labels, int batch_size,
    int epochs, bool verbose,std::function<void(T &)> aug, const S& loss_function)
{
    std::vector<int> batch_shape = data.shape();
    std::vector<int> label_shape = labels.shape();
    assert(batch_shape[0] == label_shape[0]);
    assert(batch_shape.size() > 1);
    int N = batch_shape[0];
    batch_shape[0] = batch_size;
    label_shape[0] = batch_size;
    T batch(batch_shape);
    T2 label_batch(label_shape);
    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 1189);
    std::vector<int> indexes(N);
    for (int i = 0 ; i < N; i++)
        indexes[i] = i;
    
    for (int i = 0 ; i < epochs; i++)
    {
        auto loss  = decltype(loss_function(label_batch,label_batch))();
        std::shuffle(indexes.begin(), indexes.end(), generator);
        int j = 0;
        for (j = 0 ; j <= N - batch_size; j+= batch_size)
        {
            for (int k = 0; k < batch_size; k++)
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
            std::cout<<"epoch # " <<  i << "\t loss = " << loss/(N/batch_size) <<std::endl;
    }
}
template<typename T, typename Tret>
Tret Sequential::predict_batch(const T &data)
{
    std::vector<int> batch_shape = data.shape();
    assert(batch_shape.size() > 1);
    int N = batch_shape[0];
    int batch_size = batch_shape[0];
    T batch(batch_shape);

    static std::default_random_engine generator(static_cast<unsigned>(time(nullptr))+ 1189);
    std::vector<int> indexes(N);
    for (int i = 0 ; i < N; i++)
        indexes[i] = i;
    
    int j = 0;
    
    for (int k = 0; k < batch_size; k++)
    {
        batch[k] = data[indexes[j + k]].copy();
    }
    const linal::Container* x = &(dynamic_cast<const linal::Container&>(batch));
    
    for (auto& layer : layers )
    {
        x = &(layer->forward(*x));
    }
    const T& answer = dynamic_cast<const T&>(*x);
    return answer;
}
void Sequential::set_optimizers(optim::optimizer_t type, float lr)
{
    for (auto& layer : layers )
    {
        layer->set_optimizer(type,lr);
    }
}
#endif //LIGHTY_NETS_HPP
