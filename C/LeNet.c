#define _CRT_SECURE_NO_WARNINGS
#include "LeNet.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void LeNet_init(LeNet* net, const char* quant_path, const char* param_path){
    LeNet_init_base(net);
    LeNet_init_quant(net, quant_path);
    LeNet_init_param(net, param_path);
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

void LeNet_layer_load_quant(LeNet* net, unsigned char quant_list[][3]){
    net->c1.scale = quant_list[0][0];net->c1.shift = quant_list[0][1];net->c1.zero_point = quant_list[0][2];
    net->c3.scale = quant_list[1][0];net->c3.shift = quant_list[1][1];net->c3.zero_point = quant_list[1][2];
    net->c5.scale = quant_list[2][0];net->c5.shift = quant_list[2][1];net->c5.zero_point = quant_list[2][2];
    net->f6.scale = quant_list[3][0];net->f6.shift = quant_list[3][1];net->f6.zero_point = quant_list[3][2];
    net->output.scale = quant_list[4][0];net->output.shift = quant_list[4][1];net->output.zero_point = quant_list[4][2];
}

void LeNet_init_quant(LeNet* net, const char* quant_path){
    FILE *fp = fopen(quant_path, "r");
    if(!fp){printf("Failed to open file: %s\n", quant_path);exit(1);}

    // 参数一般不超过1024字符，因此设置1024字符的缓存区
    char buffer[1024];
    // 每行读取参数
    unsigned char quant_list[5][3];
    int layer_num = 0;
    while(fgets(buffer, 1024, fp) != NULL){
        // 处理为一个完整的字符串
        if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}
        
        // 拆分三个参数，不等于三的时候报错
        int num_quant = 0;
        char* quant_param = strtok(buffer, ",");
        while (quant_param != NULL) {
            if(num_quant > 3){printf("Too many quant param\n");exit(0);}

            quant_list[layer_num][num_quant] = atoi(quant_param);
            num_quant++;
            quant_param = strtok(NULL, ",");
        }
        if(num_quant < 3){printf("Miss some quant param\n");exit(0);}
        layer_num++;
    }
    LeNet_layer_load_quant(net, quant_list);
    printf("quant load successful\n");
}

void LeNet_load_uint8(unsigned char** list, FILE* fp, int num){
    char buffer[1024];
    *list = (unsigned char*)malloc(sizeof(unsigned char)*num);

    int count = 0;
    while(fgets(buffer, 1024, fp) != NULL){
        // 处理为一个完整的字符串
        if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}
        
        char* param = strtok(buffer, ",");
        while (param != NULL) {
            if (count == num) {printf("Too many param, want %d\n", num);exit(0);}
            (*list)[count] = atoi(param);
            count++;
            param = strtok(NULL, ",");
        }
        if (count == num) {
            break;
        }
    }
}

void LeNet_load_int(int** list, FILE* fp, int num){
    char buffer[1024];
    *list = (int*)malloc(sizeof(int)*num);

    int count = 0;
    while(fgets(buffer, 1024, fp) != NULL){
        // 处理为一个完整的字符串
        if(buffer[strlen(buffer)-1] == '\n'){buffer[strlen(buffer)-1] = '\0';}
        
        char* param = strtok(buffer, ",");
        while (param != NULL) {
            if (count == num) {printf("Too many param, want %d\n", num);exit(0);}
            (*list)[count] = atoi(param);
            count++;
            param = strtok(NULL, ",");
        }
        if (count == num) {
            break;
        }
    }
}

void LeNet_load_tag_check(FILE* fp, char check){
    char buffer[10];
    if(fgets(buffer, 10, fp) == NULL){
        printf("miss param error!\n");
        exit(1);
    } else {
        if(check == buffer[0]){
            return;
        } else {
            printf("wrong check error!\n");
            exit(1);
        }
    } 
}

void LeNet_init_param(LeNet* net, const char* param_path){
    FILE *fp = fopen(param_path, "r");
    if(!fp){printf("Failed to open file: %s\n", param_path);exit(1);}

    int num = 0;

    LeNet_load_tag_check(fp, '#');
    num = net->c1.in_channels * net->c1.out_channels * net->c1.kernel_size * net->c1.kernel_size;
    LeNet_load_uint8(&net->c1.quant_weight, fp, num);
    LeNet_load_tag_check(fp, '$');
    num = net->c1.out_channels;
    LeNet_load_int(&net->c1.quant_bias, fp, num);

    LeNet_load_tag_check(fp, '#');
    num = net->c3.in_channels * net->c3.out_channels * net->c3.kernel_size * net->c3.kernel_size;
    LeNet_load_uint8(&net->c3.quant_weight, fp, num);
    LeNet_load_tag_check(fp, '$');
    num = net->c3.out_channels;
    LeNet_load_int(&net->c3.quant_bias, fp, num);

    LeNet_load_tag_check(fp, '#');
    num = net->c5.in_channels * net->c5.out_channels * net->c5.kernel_size * net->c5.kernel_size;
    LeNet_load_uint8(&net->c5.quant_weight, fp, num);
    LeNet_load_tag_check(fp, '$');
    num = net->c5.out_channels;
    LeNet_load_int(&net->c5.quant_bias, fp, num);

    LeNet_load_tag_check(fp, '#');
    num = net->f6.in_features * net->f6.out_features;
    LeNet_load_uint8(&net->f6.quant_weight, fp, num);
    LeNet_load_tag_check(fp, '$');
    num = net->f6.out_features;
    LeNet_load_int(&net->f6.quant_bias, fp, num);

    LeNet_load_tag_check(fp, '#');
    num = net->output.in_features * net->output.out_features;
    LeNet_load_uint8(&net->output.quant_weight, fp, num);
    LeNet_load_tag_check(fp, '$');
    num = net->output.out_features;
    LeNet_load_int(&net->output.quant_bias, fp, num);
    
    fclose(fp);

    printf("param load successful\n");
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

int* LeNet_forward(LeNet net, int** x, Shape* shape, int class){
    QuantConv2d_forward(net.c1, x, shape);
    ReLU(*x, shape->N*shape->C*shape->H*shape->W);
    QuantAvePool2d_forward(net.s2, x, shape);
    QuantConv2d_forward(net.c3, x, shape);
    ReLU(*x, shape->N*shape->C*shape->H*shape->W);
    QuantAvePool2d_forward(net.s4, x, shape);
    QuantConv2d_forward(net.c5, x, shape);
    QuantLinear_forward(net.f6, x, shape->N);
    QuantLinear_forward(net.output, x, shape->N);
    return Argmax(*x, shape->N, class);
}
