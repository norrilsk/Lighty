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
#include"Layers.hpp"
#include <iostream>
class Sequential
{
private:
    
    std::vector<std::unique_ptr<Layers>> layers;
public:
    template <typename S, typename T, typename...  Args>
    void addRelu1D(Args... args);
    template <typename S, typename T, typename...  Args>
    void addDense1D(Args... args);
    template<typename T,  typename T2, typename S, bool verbose = false>
    void train(const T & data, T2& labels, S & loss_function, int batch_size, int epochs = 100,
        std::function<void(T &)> aug = [](T& _data){});
    template<typename T, typename Tret>
    Tret predict(const T & data);
    template<typename T, typename Tret>
    Tret predict_batch(const T & data);
    Sequential() = default;
    ~Sequential() = default;
};
template<typename S, typename T, typename... Args>
void Sequential::addRelu1D(Args... args)
{
    layers.emplace_back( new Relu1D<float,float>(args...));
}
template<typename S, typename T, typename... Args>
void Sequential::addDense1D(Args... args)
{
    layers.emplace_back( new Dense1D<float,float>(args...));
}

template<typename T,  typename T2, typename S, bool verbose>
void Sequential::train(const T & data, T2& labels, S & loss_function, int batch_size,
    int epochs , std::function<void(T &)> aug)
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
    
    for (int i = 0 ; i < epochs; i ++)
    {
        decltype(loss_function(label_batch,label_batch)) loss  = decltype(loss_function(label_batch,label_batch))();
        std::shuffle(indexes.begin(), indexes.end(), generator);
        int j = 0;
        for (j = 0 ; j < N - batch_size; j+= batch_size)
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
            std::cout<<"epoch # " <<  i << "\t loss = " << loss <<std::endl;
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

#endif //LIGHTY_NETS_HPP