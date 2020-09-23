#ifndef LIGHTY_LAYERS_BASE_HPP
#define LIGHTY_LAYERS_BASE_HPP


#include <vector>
#include <random>
#include <chrono>
#include <tuple>
#include <ostream>

#include "linal/thensor.hpp"
#include "linal/algebra.hpp"
#include "Optimizers.hpp"
#include "Catalog.hpp"

typedef linal::thensor<float, 1> fvec;
typedef linal::thensor<float, 2> fmat;
typedef linal::thensor<float, 1> fthensor1;
typedef linal::thensor<float, 2> fthensor2;
typedef linal::thensor<float, 3> fthensor3;
typedef linal::thensor<float, 4> fthensor4;

typedef linal::thensor<int32_t, 1> ivec;
typedef linal::thensor<int32_t, 2> imat;
typedef linal::thensor<int32_t, 1> ithensor1;
typedef linal::thensor<int32_t, 2> ithensor2;
typedef linal::thensor<int32_t, 3> ithensor3;
typedef linal::thensor<int32_t, 4> ithensor4;

typedef float real32_t;
typedef double real64_t;

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
	virtual const linal::Container& backward(const linal::Container &delta) = 0;
	Layers() = default;
	virtual ~Layers() = default;
	void set_mode(TrainigMode mode) { training = mode; }
	virtual void set_optimizer(optim::optimizer_t type, float lr) {};
	virtual void dump(std::ostream & output) = 0;
};
#endif // !LIGHTY_LAYERS_BASE_HPP