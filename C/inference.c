#define _CRT_SECURE_NO_WARNINGS
// 模型推理
// #include "LeNet.h"
#include "ResNet.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 数据集所在位置，一般来说先运行python后会自动下载该数据集
const char* MNIST_DATA_PATH = "E:/PengYiteng/FPGAI/python/data/MNIST/raw/t10k-images-idx3-ubyte";
const char* MNIST_LABEL_PATH = "E:/PengYiteng/FPGAI/python/data/MNIST/raw/t10k-labels-idx1-ubyte";
const char* CIFAR10_PATH = "E:/PengYiteng/FPGAI/C/data/cifar-10/test_batch.bin";

#define MODEL ResNet
#define MODEL_INIT ResNet_init
#define MODEL_FORWARD ResNet_forward

// #define MODEL LeNet
// #define MODEL_INIT LeNet_init
// #define MODEL_FORWARD LeNet_forward

#define SCALE 1
#define BATCH_SIZE 10
#define CLASS 10

int count_TP(int* x, unsigned char* y, int len) {
    int count = 0;
    for (int i = 0; i < len; i++) {
        // printf("%d %d\n", x[i], y[i]);
        if (x[i] == y[i])
            count++;
    }
    return count;
}

// LeNet MNIST
// int main(){
//     MODEL* net = (MODEL*)malloc(sizeof(MODEL));

//     MODEL_INIT(net);

//     int num_images, num_rows, num_cols;
//     IMG_TYPE* data_list = read_mnist_images(MNIST_DATA_PATH, &num_images, &num_rows, &num_cols);
//     unsigned char* label_list = read_mnist_labels(MNIST_LABEL_PATH);


//     int count = 0;
//     for(int i = 0; i < num_images / SCALE; i+=BATCH_SIZE){
//         Shape* shape = (Shape*)malloc(sizeof(Shape));
//         shape->N = BATCH_SIZE;shape->C = 1;shape->H = num_cols;shape->W = num_rows;

//         IMG_TYPE* batch_list = (IMG_TYPE*)malloc(sizeof(IMG_TYPE) * BATCH_SIZE * num_rows * num_cols);
//         memcpy(batch_list, &data_list[i * num_rows * num_cols], sizeof(IMG_TYPE) * BATCH_SIZE * num_rows * num_cols);

//         int* result = MODEL_FORWARD(*net, batch_list, shape, CLASS);
//         count += count_TP(result, &label_list[i], BATCH_SIZE);
//         free(shape);
//     }

//     printf("%f\n", count * 1.0 / (10000 / SCALE));
//     printf("successful\n");
//     return 0;
// }

// ResNet CIFAR10
int main(){
    MODEL* net = (MODEL*)malloc(sizeof(MODEL));

    MODEL_INIT(net);

    FILE *fp = fopen(CIFAR10_PATH, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Failed to open file %s\n", CIFAR10_PATH);
        return -1;
    }

    int count = 0;
    for(int i = 0; i < 10000 / SCALE; i+=BATCH_SIZE){
        // 每次只读取BatchSize大小的数据
        IMG_TYPE* data_list = (IMG_TYPE*)malloc(sizeof(IMG_TYPE) * BATCH_SIZE * CIFAR10_IMAGE_SIZE);
        unsigned char label_list[BATCH_SIZE];
        
        int num_images = 0;
        read_cifar10(fp, data_list, label_list, BATCH_SIZE, &num_images);
        if(num_images == 0)
            break;

        Shape shape;
        shape.N = num_images;shape.C = 3;shape.H = 32;shape.W = 32;

        int* result = MODEL_FORWARD(*net, data_list, &shape, CLASS);
        count += count_TP(result, &label_list[i], BATCH_SIZE);
    }

    fclose(fp);
    printf("%f\n", count * 1.0 / (10000 / SCALE));
    printf("successful\n");
    return 0;
}
