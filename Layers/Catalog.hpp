#ifndef LIGHTY_LAYERS_CATALOG
#define LIGHTY_LAYERS_CATALOG

enum LayersType
{
	LAYER_NONE = 0,
	LAYER_FLATTERN4TO2 = 0x01,
	LAYER_DENSE1D = 0x10,
	LAYER_CONV2D = 0x11,

	LAYER_RELU = 0x30,
	LAYER_SIGMOID = 0x31,
	LAYER_TANH = 0x32
	
};

enum LayerIOType
{
	LTYPE_NONE = 0,
	LTYPE_FLOAT = 0x1
};

template <typename T>
uint32_t LayerIOType2Int() { return LTYPE_NONE; };
template <>
uint32_t LayerIOType2Int<float>() { return LTYPE_FLOAT; };
#endif // !LIGHTY_LAYERS_CATALOG

