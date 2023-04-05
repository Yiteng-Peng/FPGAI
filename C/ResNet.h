#ifndef RESNET
#define RESNET
#include "Quant.h"
#include <stdio.h>

#define ResNet18 0  // BasicBlock
#define ResNet34 1  // BasicBlock
#define ResNet50 2  // Bottleneck

#define BLOCKTYPE Bottleneck
#define BLOCK_INIT Bottleneck_init
#define BLOCK_TXT_INIT_QUANT Bottleneck_txt_init_quant
#define BLOCK_BIN_INIT_PARAM Bottleneck_bin_init_param
#define BLOCK_FORWARD Bottleneck_forward
#define BLOCK_EXPANSION 4 // BasicBlock:1 Bottleneck:4

// 量化参数和权重bias所在位置
#define QUANT_PATH "../python/export_model/ResNet50_quant.txt"
#define PARAM_PATH "../python/export_model/ResNet50_weight.txt"
#define BIN_WEIGHT_PATH "../python/export_model/ResNet50_weight.bin"
#define BIN_BIAS_PATH "../python/export_model/ResNet50_bias.bin"

typedef struct{
    QuantConv2d     conv1;
    QuantConv2d     conv2;
    QuantConv2d*    downsample;
}BasicBlock;

typedef struct{
    QuantConv2d     conv1;
    QuantConv2d     conv2;
    QuantConv2d     conv3;
    QuantConv2d*    downsample;
}Bottleneck;

void Bottleneck_init(Bottleneck* block, int inplanes, int outplanes, int stride);
void Bottleneck_txt_init_quant(Bottleneck* block, FILE* fp);
void Bottleneck_bin_init_param(Bottleneck* block, FILE* fp_weight, FILE* fp_bias);
TYPE* Bottleneck_forward(Bottleneck block, TYPE* x, Shape* shape);

typedef struct{
    QuantConv2d     conv1;
    QuantMaxPool2d  maxpool;

    BLOCKTYPE*     layers[4];
    
    QuantAvePool2d  avgpool;
    QuantLinear     fc;
}ResNet;

void ResNet_init(ResNet* net);          // 调用其他的初始化函数
void ResNet_init_base(ResNet* net);
void ResNet_txt_init_quant(ResNet* net, const char* quant_path);    // 网络量化参数的设置
void ResNet_bin_init_param(ResNet* net, const char* weight_path, const char* bias_path);

void ResNet_param_count(ResNet net);

void Conv3x3_init(QuantConv2d* layer, int inplanes, int outplanes, int stride);
void Conv1x1_init(QuantConv2d* layer, int inplanes, int outplanes, int stride);

int* ResNet_forward(ResNet net, TYPE* x, Shape* shape, int class);

#endif