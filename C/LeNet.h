#ifndef LENET
#define LENET
#include <stdio.h>
#include "Quant.h"

// 量化参数和权重bias所在位置
#define QUANT_PATH "../python/export_model/lenet_quant.txt"
#define PARAM_PATH "../python/export_model/lenet_weight.txt"
#define BIN_WEIGHT_PATH "../python/export_model/lenet_weight.bin"
#define BIN_BIAS_PATH "../python/export_model/lenet_bias.bin"


#define TRUE  1
#define FALSE 0

typedef struct{
    QuantConv2d     c1;
    QuantAvePool2d  s2;
    QuantConv2d     c3;
    QuantAvePool2d  s4;
    QuantConv2d     c5;
    QuantLinear     f6;
    QuantLinear     output;
}LeNet;

void LeNet_init(LeNet* net);          // 调用其他的初始化函数

void LeNet_init_base(LeNet* net);           // 网络参数的搭建
void LeNet_txt_init_quant(LeNet* net, const char* quant_path);    // 网络量化参数的设置
void LeNet_txt_init_param(LeNet* net, const char* param_path);    // 网络具体参数的读入
void LeNet_bin_init_param(LeNet* net, const char* weight_path, const char* bias_path);

TYPE* ReLU1(int* x, int len);
void ReLU(int* x, int len);
int* Argmax(int* x, int num, int class);

int* LeNet_forward(LeNet net, TYPE* x, Shape* shape, int class);

#endif