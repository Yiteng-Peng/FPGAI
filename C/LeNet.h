#ifndef LENET
#define LENET
#include <stdio.h>
#include "Quant.h"

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

void LeNet_init(LeNet* net, const char* quant_path, const char* param_path);          // 调用其他的初始化函数

void LeNet_init_base(LeNet* net);           // 网络参数的搭建
void LeNet_init_quant(LeNet* net, const char* quant_path);    // 网络量化参数的设置
void LeNet_init_param(LeNet* net, const char* param_path);    // 网络具体参数的读入

void LeNet_load_tag_check(FILE* fp, char check);
void LeNet_load_int(int** list, FILE* fp, int num);
void LeNet_load_uint8(unsigned char** list, FILE* fp, int num);

void LeNet_layer_load_quant(LeNet* net, unsigned char quant_list[][3]);   // 为指定的层设置量化参数

void ReLU(int* x, int len);
int* Argmax(int* x, int num, int class);

int* LeNet_forward(LeNet net, int** x, Shape* shape, int class);

#endif