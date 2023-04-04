#define _CRT_SECURE_NO_WARNINGS
#include "ResNet.h"
#include "Load.h"

#include <stdio.h>
#include <stdlib.h>

const int cfgs[][4] = {
    {2, 2, 2, 2},       // 18 BasicBlock
    {3, 4, 6, 3},       // 34 BasicBlock
    {3, 4, 6, 3}        // 50 Bottleneck
};

void Conv3x3_init(QuantConv2d* layer, int inplanes, int outplanes, int stride){
    layer->in_channels = inplanes; layer->out_channels = outplanes;
    layer->kernel_size = 3; layer->stride = stride; layer->padding = 1;
}

void Conv1x1_init(QuantConv2d* layer, int inplanes, int outplanes, int stride){
    layer->in_channels = inplanes; layer->out_channels = outplanes;
    layer->kernel_size = 1; layer->stride = stride; layer->padding = 0;
}

void ResNet_init(ResNet* net){
    ResNet_init_base(net);
    printf("init base successful\n");
    // ResNet_param_count(*net);
    ResNet_txt_init_quant(net, QUANT_PATH);
    ResNet_bin_init_param(net, BIN_WEIGHT_PATH, BIN_BIAS_PATH);
}

void Bottleneck_init(Bottleneck* block, int inplanes, int outplanes, int stride){
    Conv1x1_init(&block->conv1, inplanes, outplanes, 1);
    Conv3x3_init(&block->conv2, outplanes, outplanes, stride);
    Conv1x1_init(&block->conv3, outplanes, outplanes * BLOCK_EXPANSION, 1);
    block->downsample = NULL;
}

void ResNet_init_base(ResNet* net){
    // 初始层
    Conv3x3_init(&net->conv1, 3, 64, 1);
    // padding = 1
    net->maxpool.kernel_size = 3; net->maxpool.stride = 2;

    int inplanes = 64;  // 和上面的out_channels一致
    const int outplanes_list[] = {64, 128, 256, 512};
    const int stride_list[] = {1, 2, 2, 2};

    // 主干层
    for(int i = 0; i < 4; i++){
        int block_num = cfgs[ResNet50][i];
        net->layers[i] = (BLOCKTYPE*)malloc(sizeof(BLOCKTYPE)*block_num);
        
        // _make_layer
        BLOCK_INIT(&net->layers[i][0], inplanes, outplanes_list[i], stride_list[i]);
        if(stride_list[i] != 1 || inplanes != outplanes_list[i] * BLOCK_EXPANSION){
            QuantConv2d* downsample = (QuantConv2d*)malloc(sizeof(QuantConv2d));
            Conv1x1_init(downsample, inplanes, outplanes_list[i] * BLOCK_EXPANSION, stride_list[i]);
            net->layers[i][0].downsample = downsample;
        }

        inplanes = outplanes_list[i] * BLOCK_EXPANSION;
        int j = 1;
        for(j = 1; j < block_num; j++){
            BLOCK_INIT(&net->layers[i][j], inplanes, outplanes_list[i], 1);
        }
    }

    // 全连接层
    net->fc.in_features = 512 * BLOCK_EXPANSION;
    net->fc.out_features= 10;
}

void ResNet_param_count(ResNet net) {
    int sum = 0;

    sum += 3 * 64 * 3 * 3;

    for (int i = 0; i < 4; i++) {
        int block_num = cfgs[ResNet50][i];
        for (int j = 0; j < block_num; j++) {
            sum += net.layers[i][j].conv1.in_channels * net.layers[i][j].conv1.out_channels * net.layers[i][j].conv1.kernel_size * net.layers[i][j].conv1.kernel_size;
            sum += net.layers[i][j].conv2.in_channels * net.layers[i][j].conv2.out_channels * net.layers[i][j].conv2.kernel_size * net.layers[i][j].conv2.kernel_size;
            sum += net.layers[i][j].conv3.in_channels * net.layers[i][j].conv3.out_channels * net.layers[i][j].conv3.kernel_size * net.layers[i][j].conv3.kernel_size;
            if (net.layers[i][j].downsample) {
                sum += net.layers[i][j].downsample->in_channels * net.layers[i][j].downsample->out_channels * net.layers[i][j].downsample->kernel_size * net.layers[i][j].downsample->kernel_size;
                printf("ds: %d %d\n", i, j);
            }
        }
    }

    sum += 512 * BLOCK_EXPANSION * 10;

    printf("%d", sum);
    exit(0);
}

void Bottleneck_txt_init_quant(Bottleneck* block, FILE* fp){
    txt_load_quant(fp, &block->conv1.scale, &block->conv1.shift, &block->conv1.zero_point);
    txt_load_quant(fp, &block->conv2.scale, &block->conv2.shift, &block->conv2.zero_point);
    txt_load_quant(fp, &block->conv3.scale, &block->conv3.shift, &block->conv3.zero_point);
    if(block->downsample != NULL)
        txt_load_quant(fp, &block->downsample->scale, &block->downsample->shift, &block->downsample->zero_point);
}

void Bottleneck_bin_init_param(Bottleneck* block, FILE* fp_weight, FILE* fp_bias){
    bin_load_conv(&block->conv1, fp_weight, fp_bias);
    bin_load_conv(&block->conv2, fp_weight, fp_bias);
    bin_load_conv(&block->conv3, fp_weight, fp_bias);
    if(block->downsample != NULL)
        bin_load_conv(block->downsample, fp_weight, fp_bias);
}

void ResNet_txt_init_quant(ResNet* net, const char* quant_path){
    FILE *fp = fopen(quant_path, "r");
    if(!fp){printf("Failed to open file: %s\n", quant_path);exit(1);}

    // 初始层
    txt_load_quant(fp, &net->conv1.scale, &net->conv1.shift, &net->conv1.zero_point);

    // 主干层
    for(int i = 0; i < 4; i++){
        int block_num = cfgs[ResNet50][i];
        for(int j = 0; j < block_num; j++){
            BLOCK_TXT_INIT_QUANT(&net->layers[i][j], fp);
        }
    }

    // 全连接层
    txt_load_quant(fp, &net->fc.scale, &net->fc.shift, &net->fc.zero_point);
    
    printf("quant load successful\n");
}

void ResNet_bin_init_param(ResNet* net, const char* weight_path, const char* bias_path){
    FILE *fp_weight = fopen(weight_path, "rb");
    if(!fp_weight){printf("Failed to open file: %s\n", weight_path);exit(1);}
    FILE *fp_bias = fopen(bias_path, "rb");
    if(!fp_bias){printf("Failed to open file: %s\n", bias_path);exit(1);}

    int num = 0;

    // 初始层
    bin_load_conv(&net->conv1, fp_weight, fp_bias);

    // 主干层
    for(int i = 0; i < 4; i++){
        int block_num = cfgs[ResNet50][i];
        for(int j = 0; j < block_num; j++){
            BLOCK_BIN_INIT_PARAM(&net->layers[i][j], fp_weight, fp_bias);
        }
    }
    
    bin_load_fc(&net->fc, fp_weight, fp_bias);

    fclose(fp_weight);
    fclose(fp_bias);

    printf("param load successful\n");
}

TYPE* Bottleneck_forward(Bottleneck block, TYPE* x, Shape* shape){
    // 暂存x的初始值
    int len_x = shape->N*shape->C*shape->H*shape->W;
    TYPE* copy_x = (TYPE*)malloc(sizeof(TYPE)*len_x);
    memcpy(copy_x, x, sizeof(TYPE)*len_x);
    Shape* copy_shape = (Shape*)malloc(sizeof(Shape));
    memcpy(copy_shape, shape, sizeof(Shape));

    int* mid_x;     // x的中间值，在每次乘加运算后会变成32位，之后用ReLU1激活到8位
    mid_x = QuantConv2d_forward(block.conv1, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    mid_x = QuantConv2d_forward(block.conv2, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    mid_x = QuantConv2d_forward(block.conv3, x, shape);
    if(block.downsample != NULL){
        int* downsample_x;
        downsample_x = QuantConv2d_forward(*block.downsample, copy_x, copy_shape);
        IntAddInt(mid_x, downsample_x, shape->N*shape->C*shape->H*shape->W);
        free(downsample_x);
        free(copy_shape);
    } else {
        IntAddType(mid_x, copy_x, shape->N*shape->C*shape->H*shape->W);
        free(copy_x);
        free(copy_shape);
    }
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);

    return x;
}

int* ResNet_forward(ResNet net, TYPE* x, Shape* shape, int class){
    int* mid_x;     // x的中间值，在每次乘加运算后会变成32位，之后用ReLU1激活到8位
    mid_x = QuantConv2d_forward(net.conv1, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    Pad(&x, 1, shape);
    x = QuantMaxPool2d_forward(net.maxpool, x, shape);

    for(int i = 0; i < 4; i++){
        int block_num = cfgs[ResNet50][i];
        for(int j = 0; j < block_num; j++){
            x = BLOCK_FORWARD(net.layers[i][j], x, shape);
        }
    }

    x = QuantAvePool2d_forward(net.avgpool, x, shape);
    mid_x = QuantLinear_forward(net.fc, x, shape);

    return Argmax(mid_x, shape->N, class);
}