#define _CRT_SECURE_NO_WARNINGS
// 模型推理
#include "LeNet.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 数据集所在位置，一般来说先运行python后会自动下载该数据集
const char* DATASET_DATA = "../python/data/MNIST/raw/t10k-images-idx3-ubyte";
const char* DATASET_LABEL = "../python/data/MNIST/raw/t10k-labels-idx1-ubyte";

// 量化参数和权重bias所在位置
const char* QUANT_PATH = "../python/txt_model/lenet_quant.txt";
const char* PARAM_PATH = "../python/txt_model/lenet_weight.txt";

#define MODEL LeNet
#define MODEL_INIT LeNet_init
#define MODEL_FORWARD LeNet_forward

#define SCALE 1
#define BATCH_SIZE 10
#define CLASS 10

int count_TP(int* x, unsigned char* y, int len) {
    int i; int count = 0;
    for (i = 0; i < len; i++) {
        if (x[i] == y[i])
            count++;
    }
    return count;
}

int main(){
    MODEL* net = (MODEL*)malloc(sizeof(MODEL));
    int num_images, num_rows, num_cols;

    MODEL_INIT(net, QUANT_PATH, PARAM_PATH);

    IMG_TYPE* data_list = read_mnist_images(DATASET_DATA, &num_images, &num_rows, &num_cols);
    unsigned char* label_list = read_mnist_labels(DATASET_LABEL);

    int i = 0; int count = 0;
    for(i = 0; i < num_images / SCALE; i+=BATCH_SIZE){
        Shape* shape = (Shape*)malloc(sizeof(Shape));
        shape->N = BATCH_SIZE;shape->C = 1;shape->H = num_cols;shape->W = num_rows;

        IMG_TYPE* batch_list = (IMG_TYPE*)malloc(sizeof(IMG_TYPE) * BATCH_SIZE * num_rows * num_cols);
        memcpy(batch_list, &data_list[i * num_rows * num_cols], sizeof(IMG_TYPE) * BATCH_SIZE * num_rows * num_cols);

        int* result = MODEL_FORWARD(*net, batch_list, shape, CLASS);
        count += count_TP(result, &label_list[i], BATCH_SIZE);
        free(shape);
    }

    printf("%f\n", count * 1.0 / (num_images / SCALE));
    printf("successful\n");
    return 0;
}
