#ifndef LIGHTY_LAYERS_BASE_HPP
#define LIGHTY_LAYERS_BASE_HPP


#include <vector>
#include <random>
#include "linal/thensor.hpp"
#include "linal/algebra.hpp"
#include "Optimizers.hpp"
#include <chrono>
#include <tuple>

typedef linal::thensor<float, 1> fvec;
typedef linal::thensor<float, 2> fmat;
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
	virtual const linal::Container& forward(const linal::Container &src) = 0;
	//http://neuralnetworksanddeeplearning.com/chap2.html
	virtual const linal::Container& backward(const linal::Container &delta) = 0;
	Layers() = default;
	virtual ~Layers() = default;
	void set_mode(TrainigMode mode) { training = mode; }
	virtual void set_optimizer(optim::optimizer_t type, float lr) {};
};
#endif // !LIGHTY_LAYERS_BASE_HPP