#include "lenet.h"
#include <memory.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>

static void convolute_valid(double *src, double *conv, double *des, const long dh, const long dw, const long ch, const long cw)
{
	const long sw = dw + cw - 1;
	for (long d0 = 0; d0 < dh; ++d0)
		for (long d1 = 0; d1 < dw; ++d1)
		{
			for (long c0 = 0; c0 < ch; ++c0)
				for (long c1 = 0; c1 < cw; ++c1)
				{
					des[d0 * dw + d1] += src[(d0 + c0)*sw + d1 + c1] * conv[c0*cw + c1];
				}
		}
}

static void convolute_full(double *src, double *conv, double *des, const long sh, const long sw, const long ch, const long cw)
{
	const long dw = sw + cw - 1;
	for (long s0 = 0; s0 < sh; ++s0)
		for (long s1 = 0; s1 < sw; ++s1)
		{
			for (long c0 = 0; c0 < ch; ++c0)
				for (long c1 = 0; c1 < cw; ++c1)
				{
					des[(s0 + c0)*dw + s1 + c1] += src[s0*sw + s1] * conv[c0*cw + c1];
				}
		}
}



static void vector_x_matrix(double *src, double *mat, double *des, const long height, const long width)
{
	for (long y = 0; y < width; ++y)
	{
		for (long x = 0; x < height; ++x)
		{
			des[y] += src[x] * mat[x*width + y];
		}
	}
}

static void matrix_x_vector(double *mat, double *src, double *des, const long height, const long width)
{
	for (long x = 0; x < height; ++x)
	{
		for (long y = 0; y < width; ++y)
		{
			des[x] += src[y] * mat[x*width + y];
		}
	}
}

static void convolution_forward(double *src, double *conv, double *des,double *bias,double(*active)(double), const long dh, const long dw, const long ch, const long cw, const long sn, const long dn)
{
	const long srcSize = (dh + ch - 1) * (dw + cw - 1), desSize = dh * dw, convSize = ch * cw;
	for (int y = 0; y < dn; ++y)
		for (int x = 0; x < sn; ++x)
			convolute_valid(src + x * srcSize, conv + (x * dn + y)*convSize, des + y*desSize, dh, dw, ch, cw);
	for (int i = 0; i < dn; ++i)
	{
		double *desMat = des + i * desSize;
		for (int j = 0; j < desSize; ++j)
		{
			desMat[j] = active(desMat[j] + bias[i]);
		}
	}
}

static void convolution_backward(double *src, double *conv, double *des, double *desl, double *wd, double *bd, double(*activegrad)(double), const long sh, const long sw, const long ch, const long cw, const long sn, const long dn)
{
	const long srcSize = sh * sw, desSize = (sh + ch - 1) * (sw + cw - 1), convSize = ch * cw;
	for (int x = 0; x < dn; ++x)
		for (int y = 0; y < sn; ++y)
		{
			convolute_full(src + y*srcSize, conv + (x*sn + y)*convSize, des + x*desSize, sh, sw, ch, cw);
		}
	for (int i = 0; i < desSize * dn; ++i)
		des[i] *= activegrad(desl[i]);
	for (int i = 0; i < sn; ++i)
		for (int j = 0; j < srcSize; ++j)
			bd[i] += src[i * srcSize + j];
	for (int x = 0; x < dn; ++x)
		for (int y = 0; y < sn; ++y)
		{
			convolute_valid(desl + x *desSize, src + y *srcSize, wd + (x*sn + y)*convSize, ch, cw, sh, sw);
		}
}

static void subsamp_max_forward(double *src, double *des, const long sh, const long sw, const long dh, const long dw, const long n)
{
	const long srcSize = sh * sw, desSize = dh * dw;
	const long lh = sh / dh, lw = sw / dw;
	for (long i = 0; i < n; ++i)
	{
		for (long d0 = 0; d0 < dh; ++d0)
			for (long d1 = 0; d1 < dw; ++d1)
			{
				long x = d0 * lh * sw + d1 * lw;
				for (long l = 1; l < lh * lw; ++l)
				{
					long index = (d0 * lh + l / lw) * sw + d1 * lw + l % lw;
					x += (src[index] > src[x]) * (index - x);
				}
				des[d0 * dw + d1] = src[x];
			}
		src += srcSize;
		des += desSize;
	}
}

static void subsamp_max_backward(double *desl, double *src, double *des, const long sh, const long sw, const long dh, const long dw, const long n)
{
	const long srcSize = sh * sw, desSize = dh * dw;
	const long lh = dh / sh, lw = dw / sw;
	for (long i = 0; i < n; ++i)
	{
		for (long s0 = 0; s0 < sh; ++s0)
			for (long s1 = 0; s1 < sw; ++s1)
			{
				long x = s0 * lh * dw + s1 * lw;
				for (long l = 1; l < lh * lw; ++l)
				{
					long index = (s0 * lh + l / lw) * dw + s1 * lw + l % lw;
					x += (desl[index] > desl[x]) * (index - x);
				}
				des[x] = src[s0 * sw + s1];
			}
		src += srcSize;
		des += desSize;
		desl += desSize;
	}
}

static void dot_product_forward(double *src, double *mat, double *des,double *bias, double(*active)(double), const long height, const long width)
{
	vector_x_matrix(src, mat, des, height, width);
	for (int i = 0; i < width; ++i)
		des[i] = active(des[i] + bias[i]);
}

static void dot_product_backward(double *src, double *mat, double *des, double *desl, double *wd, double *bd, double(*activegrad)(double), const long height, const long width)
{
	matrix_x_vector(mat, src, des, height, width);
	for (int i = 0; i < height; ++i)
		des[i] *= activegrad(desl[i]);
	for (int i = 0; i < width; ++i)
		bd[i] += src[i];
	for (int x = 0; x < height; ++x)
		for (int y = 0; y < width; ++y)
			wd[x * width + y] += desl[x] * src[y];
}

#define GETLENGTH(array) (sizeof(array)/sizeof(*(array)))

#define GETCOUNT(array)  (sizeof(array)/sizeof(double))

#define SUBSAMP_MAX_FORWARD(input,output)								\
{																		\
	subsamp_max_forward((double *)input,(double *)output,				\
							GETLENGTH(*input),GETLENGTH(**input),		\
							GETLENGTH(*output),GETLENGTH(**output),GETLENGTH(output));\
}

#define SUBSAMP_MAX_BACKWARD(input,inerror,outerror)										\
{																							\
	subsamp_max_backward((double *)input,(double *)outerror,(double *)inerror,				\
							GETLENGTH(*outerror),GETLENGTH(**outerror),						\
							GETLENGTH(*inerror),GETLENGTH(**inerror), GETLENGTH(outerror));	\
}

#define DOT_PRODUCT_FORWARD(input,output,weight,bias,action)				\
{																			\
	dot_product_forward((double *)input,(double *)weight,(double *)output,	\
				(double *)bias,action,GETLENGTH(weight),GETLENGTH(*weight));\
}



#define DOT_PRODUCT_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)	\
{																				\
	dot_product_backward((double *)outerror,(double *)weight,(double *)inerror,	\
						(double *)input,(double *)wd,(double *)bd,actiongrad,	\
							GETLENGTH(weight),GETLENGTH(*weight));				\
}

#define CONVOLUTION_FORWARD(input,output,weight,bias,action)								\
{																							\
	convolution_forward((double *)input,(double *)weight,(double *)output,(double *)bias,	\
			action,GETLENGTH(*output),GETLENGTH(**output),GETLENGTH(**weight),				\
				GETLENGTH(***weight),GETLENGTH(weight),GETLENGTH(*weight));					\
}

#define CONVOLUTION_BACKWARD(input,inerror,outerror,weight,wd,bd,actiongrad)			\
{																						\
	convolution_backward((double *)outerror,(double *)weight,(double *)inerror,			\
						(double *)input,(double *)wd,(double *)bd,actiongrad,			\
						GETLENGTH(*outerror),GETLENGTH(**outerror),GETLENGTH(**weight),	\
						GETLENGTH(***weight),GETLENGTH(*weight),GETLENGTH(weight));		\
}


double relu(double x)
{
	return x*(x > 0);
}

double relugrad(double y)
{
	return y > 0;
}

static void forward(LeNet5 *lenet, Feature *features, double(*action)(double))
{
	CONVOLUTION_FORWARD(features->input, features->layer1, lenet->weight0_1, lenet->bias0_1, action);
	SUBSAMP_MAX_FORWARD(features->layer1, features->layer2);
	CONVOLUTION_FORWARD(features->layer2, features->layer3, lenet->weight2_3, lenet->bias2_3, action);
	SUBSAMP_MAX_FORWARD(features->layer3, features->layer4);
	CONVOLUTION_FORWARD(features->layer4, features->layer5, lenet->weight4_5, lenet->bias4_5, action);
	DOT_PRODUCT_FORWARD(features->layer5, features->output, lenet->weight5_6, lenet->bias5_6, action);
}

static void backward(LeNet5 *lenet, LeNet5 *deltas, Feature *errors, Feature *features, double(*actiongrad)(double))
{
	DOT_PRODUCT_BACKWARD(features->layer5, errors->layer5, errors->output, lenet->weight5_6, deltas->weight5_6, deltas->bias5_6, actiongrad);
	CONVOLUTION_BACKWARD(features->layer4, errors->layer4, errors->layer5, lenet->weight4_5, deltas->weight4_5, deltas->bias4_5, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer3, errors->layer3, errors->layer4);
	CONVOLUTION_BACKWARD(features->layer2, errors->layer2, errors->layer3, lenet->weight2_3, deltas->weight2_3, deltas->bias2_3, actiongrad);
	SUBSAMP_MAX_BACKWARD(features->layer1, errors->layer1, errors->layer2);
	CONVOLUTION_BACKWARD(features->input, errors->input, errors->layer1, lenet->weight0_1, deltas->weight0_1, deltas->bias0_1, actiongrad);
}

static inline void load_input(Feature *features, image input)
{
	double (*layer0)[LENGTH_FEATURE0][LENGTH_FEATURE0] = features->input;
	const long sz = sizeof(image) / sizeof(**input);
	double mean = 0, std = 0;
	for(int j = 0; j < sizeof(image) / sizeof(*input); ++j)
		for(int k = 0; k < sizeof(*input) / sizeof(**input); ++k)
	{
		mean += input[j][k];
		std += input[j][k] * input[j][k];
	}
	mean /= sz;
	std = sqrt(std / sz - mean*mean);
	for(int j = 0; j < sizeof(image) / sizeof(*input); ++j)
		for(int k = 0; k < sizeof(*input) / sizeof(**input); ++k)
	{
		layer0[0][j + PADDING][k + PADDING] = (input[j][k] - mean) / std;
	}
}

static uint8 get_result(double *output, uint8 count)
{
	uint8 result = 0;
	for (uint8 i = 1; i < count; ++i)
		result += (i - result) * (output[i] > output[result]);
	return result;
}

static inline void softmax(double input[OUTPUT], double loss[OUTPUT], uint8 label, int count)
{
	double max = input[get_result(input, count)];
	double k = 0, inner = 0;
	for (uint8 i = 0; i < count; ++i)
	{
		loss[i] = exp(input[i] - max);
		k += loss[i];
	}
	k = 1. / k;
	for (uint8 i = 0; i < count; ++i)
	{
		loss[i] *= k;
		inner -= loss[i] * loss[i];
	}
	inner += loss[label];
	for (uint8 i = 0; i < count; ++i)
	{
		loss[i] *= (i == label) - loss[i] - inner;
	}
}

void TrainBatch(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize)
{
	double buffer[GETCOUNT(LeNet5)] = { 0 };
	int i = 0;
#pragma omp parallel for
	for (i = 0; i < batchSize; ++i)
	{
		Feature features = { 0 };
		Feature errors = { 0 };
		LeNet5	deltas = { 0 };
		load_input(&features, inputs[i]);
		forward(lenet, &features, relu);
		softmax(features.output, errors.output, labels[i], GETCOUNT(features.output));
		backward(lenet, &deltas, &errors, &features, relugrad);
		#pragma omp critical
		{
			for(int j = 0;j < GETCOUNT(LeNet5); ++j)
				buffer[j] += ((double *)&deltas)[j];
		}
	}
	double k = ALPHA / batchSize;
	for(int i = 0; i < GETCOUNT(LeNet5); ++i)
		((double *)lenet)[i] += k * buffer[i];
}

void Train(LeNet5 *lenet, image input, uint8 label)
{
	Feature features = { 0 };
	Feature errors = { 0 };
	LeNet5 deltas = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	softmax(features.output, errors.output, label, GETCOUNT(features.output));
	backward(lenet, &deltas, &errors, &features, relugrad);
	for(int i = 0; i < GETCOUNT(LeNet5); ++i)
		((double *)lenet)[i] += ALPHA * ((double *)&deltas)[i];
}

uint8 Predict(LeNet5 *lenet, image input,uint8 count)
{
	Feature features = { 0 };
	load_input(&features, input);
	forward(lenet, &features, relu);
	return get_result(features.output, count);
}

void Initial(LeNet5 *lenet)
{
	//srand((unsigned)time(0));
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->bias0_1; *pos++ = rand()*(2. / RAND_MAX) - 1);
	for (double *pos = (double *)lenet->weight0_1; pos < (double *)lenet->weight2_3; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (INPUT + LAYER1))));
	for (double *pos = (double *)lenet->weight2_3; pos < (double *)lenet->weight4_5; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER2 + LAYER3))));
	for (double *pos = (double *)lenet->weight4_5; pos < (double *)lenet->weight5_6; *pos++ *= sqrt(6.0 / (LENGTH_KERNEL * LENGTH_KERNEL * (LAYER4 + LAYER5))));
	for (double *pos = (double *)lenet->weight5_6; pos < (double *)lenet->bias0_1; *pos++ *= sqrt(6.0 / (LAYER5 + OUTPUT)));
	for (int *pos = (int *)lenet->bias0_1; pos < (int *)(lenet + 1); *pos++ = 0);
}