#define _CRT_SECURE_NO_WARNINGS
#include "LeNet.h"
#include "Load.h"

#include <stdio.h>
#include <stdlib.h>

void LeNet_init(LeNet* net){
    LeNet_init_base(net);
    LeNet_txt_init_quant(net, QUANT_PATH);
    // LeNet_txt_init_param(net, PARAM_PATH);
    LeNet_bin_init_param(net, BIN_WEIGHT_PATH, BIN_BIAS_PATH);
}

void LeNet_init_base(LeNet* net){
    net->c1.in_channels     = 1;
    net->c1.out_channels    = 6;
    net->c1.kernel_size     = 5;
    net->c1.stride          = 1;
    net->c1.padding         = 2;

    net->s2.kernel_size     = 2;
    net->s2.stride          = 2;

    net->c3.in_channels     = 6;
    net->c3.out_channels    = 16;
    net->c3.kernel_size     = 5;
    net->c3.stride          = 1;
    net->c3.padding         = 0;

    net->s4.kernel_size     = 2;
    net->s4.stride          = 2;

    net->c5.in_channels     = 16;
    net->c5.out_channels    = 120;
    net->c5.kernel_size     = 5;
    net->c5.stride          = 1;
    net->c5.padding         = 0;

    net->f6.in_features     = 120;
    net->f6.out_features    = 84;

    net->output.in_features = 84;
    net->output.out_features= 10;
}

void LeNet_txt_init_quant(LeNet* net, const char* quant_path){
    FILE *fp = fopen(quant_path, "r");
    if(!fp){printf("Failed to open file: %s\n", quant_path);exit(1);}

    txt_load_quant(fp, &net->c1.scale, &net->c1.shift, &net->c1.zero_point);
    txt_load_quant(fp, &net->c3.scale, &net->c3.shift, &net->c3.zero_point);
    txt_load_quant(fp, &net->c5.scale, &net->c5.shift, &net->c5.zero_point);
    txt_load_quant(fp, &net->f6.scale, &net->f6.shift, &net->f6.zero_point);
    txt_load_quant(fp, &net->output.scale, &net->output.shift, &net->output.zero_point);
    
    printf("quant load successful\n");
}

void LeNet_txt_init_param(LeNet* net, const char* param_path){
    FILE *fp = fopen(param_path, "r");
    if(!fp){printf("Failed to open file: %s\n", param_path);exit(1);}

    int num = 0;

    txt_load_tag_check(fp, '#');
    num = net->c1.in_channels * net->c1.out_channels * net->c1.kernel_size * net->c1.kernel_size;
    txt_load_uint8(&net->c1.quant_weight, fp, num);
    txt_load_tag_check(fp, '$');
    num = net->c1.out_channels;
    txt_load_int(&net->c1.quant_bias, fp, num);

    txt_load_tag_check(fp, '#');
    num = net->c3.in_channels * net->c3.out_channels * net->c3.kernel_size * net->c3.kernel_size;
    txt_load_uint8(&net->c3.quant_weight, fp, num);
    txt_load_tag_check(fp, '$');
    num = net->c3.out_channels;
    txt_load_int(&net->c3.quant_bias, fp, num);

    txt_load_tag_check(fp, '#');
    num = net->c5.in_channels * net->c5.out_channels * net->c5.kernel_size * net->c5.kernel_size;
    txt_load_uint8(&net->c5.quant_weight, fp, num);
    txt_load_tag_check(fp, '$');
    num = net->c5.out_channels;
    txt_load_int(&net->c5.quant_bias, fp, num);

    txt_load_tag_check(fp, '#');
    num = net->f6.in_features * net->f6.out_features;
    txt_load_uint8(&net->f6.quant_weight, fp, num);
    txt_load_tag_check(fp, '$');
    num = net->f6.out_features;
    txt_load_int(&net->f6.quant_bias, fp, num);

    txt_load_tag_check(fp, '#');
    num = net->output.in_features * net->output.out_features;
    txt_load_uint8(&net->output.quant_weight, fp, num);
    txt_load_tag_check(fp, '$');
    num = net->output.out_features;
    txt_load_int(&net->output.quant_bias, fp, num);
    
    fclose(fp);

    printf("param load successful\n");
}

void LeNet_bin_init_param(LeNet* net, const char* weight_path, const char* bias_path){
    FILE *fp_weight = fopen(weight_path, "rb");
    if(!fp_weight){printf("Failed to open file: %s\n", weight_path);exit(1);}
    FILE *fp_bias = fopen(bias_path, "rb");
    if(!fp_bias){printf("Failed to open file: %s\n", bias_path);exit(1);}

    int num = 0;

    num = net->c1.in_channels * net->c1.out_channels * net->c1.kernel_size * net->c1.kernel_size;
    bin_load_uint8(&net->c1.quant_weight, fp_weight, num);
    num = net->c1.out_channels;
    bin_load_int(&net->c1.quant_bias, fp_bias, num);

    num = net->c3.in_channels * net->c3.out_channels * net->c3.kernel_size * net->c3.kernel_size;
    bin_load_uint8(&net->c3.quant_weight, fp_weight, num);
    num = net->c3.out_channels;
    bin_load_int(&net->c3.quant_bias, fp_bias, num);

    num = net->c5.in_channels * net->c5.out_channels * net->c5.kernel_size * net->c5.kernel_size;
    bin_load_uint8(&net->c5.quant_weight, fp_weight, num);
    num = net->c5.out_channels;
    bin_load_int(&net->c5.quant_bias, fp_bias, num);

    num = net->f6.in_features * net->f6.out_features;
    bin_load_uint8(&net->f6.quant_weight, fp_weight, num);
    num = net->f6.out_features;
    bin_load_int(&net->f6.quant_bias, fp_bias, num);

    num = net->output.in_features * net->output.out_features;
    bin_load_uint8(&net->output.quant_weight, fp_weight, num);
    num = net->output.out_features;
    bin_load_int(&net->output.quant_bias, fp_bias, num);
    
    fclose(fp_weight);
    fclose(fp_bias);

    printf("param load successful\n");
}

TYPE* ReLU1(int* x, int len){
    TYPE* out_x = (TYPE*)malloc(sizeof(TYPE)*len);

    int i = 0;
    for(i = 0; i < len; i++) {
        if(x[i] > 255){
            out_x[i] = 255;
        } else if (x[i] < 0){
            out_x[i] = 0;
        } else {
            out_x[i] = x[i];
        }
    }

    free(x);
    return out_x;
}

void ReLU(int* x, int len){
    int i = 0;
    for(i = 0; i < len; i++) {
        if(x[i] > 0){
            x[i] = x[i];
        } else {
            x[i] = 0;
        }
    }
}

int* Argmax(int* x, int num, int class){
    int* result = (int*)malloc(sizeof(int)*num);
    int i, j;
    for(i = 0; i < num; i++){
        result[i] = 0;
        int max = INT_MIN;
        for(j = 0; j < class; j++){
            if(x[i*num+j] > max){
                max = x[i*num+j];
                result[i] = j;
            }
        }
    }
    free(x);
    return result;
}

int* LeNet_forward(LeNet net, TYPE* x, Shape* shape, int class){
    int* mid_x;     // x的中间值，在每次乘加运算后会变成32位，之后用ReLU1激活到8位
    mid_x = QuantConv2d_forward(net.c1, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    x = QuantAvePool2d_forward(net.s2, x, shape);

    mid_x = QuantConv2d_forward(net.c3, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    x = QuantAvePool2d_forward(net.s4, x, shape);
    
    mid_x = QuantConv2d_forward(net.c5, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C*shape->H*shape->W);
    
    mid_x = QuantLinear_forward(net.f6, x, shape);
    x = ReLU1(mid_x, shape->N*shape->C);
    
    mid_x = QuantLinear_forward(net.output, x, shape);
    return Argmax(mid_x, shape->N, class);
}
