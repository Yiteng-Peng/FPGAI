#define _CRT_SECURE_NO_WARNINGS

#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int swap_endian(int val)
{
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

IMG_TYPE* read_mnist_images(const char *filename, int* num_images, int* num_rows, int* num_cols) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }
    int magic_number;
    fread(&magic_number, sizeof(int), 1, fp);
    fread(num_images, sizeof(int), 1, fp);
    fread(num_rows, sizeof(int), 1, fp);
    fread(num_cols, sizeof(int), 1, fp);

    magic_number = swap_endian(magic_number);*num_images = swap_endian(*num_images);
    *num_rows = swap_endian(*num_rows);*num_cols = swap_endian(*num_cols);

    printf("%d, %d, %d, %d\n", magic_number, *num_images, *num_rows, *num_cols);
    int len_images_array = *num_images * *num_rows * *num_cols;
    IMG_TYPE* images = (IMG_TYPE*)malloc(sizeof(IMG_TYPE)*len_images_array);

    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < *num_rows * *num_cols; j++) {
            unsigned char temp;
            fread(&temp, sizeof(unsigned char), 1, fp);
            images[i * *num_rows * *num_cols + j] = temp;
        }
    }
    fclose(fp);

    printf("data load successful\n");
    return images;
}

unsigned char* read_mnist_labels(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        printf("Failed to open file: %s\n", filename);
        exit(1);
    }
    int magic_number, num_labels;
    fread(&magic_number, sizeof(int), 1, fp);
    fread(&num_labels, sizeof(int), 1, fp);

    magic_number = swap_endian(magic_number);num_labels = swap_endian(num_labels);
    printf("%d, %d\n", magic_number, num_labels);
    int len_labels_array = num_labels;
    unsigned char* labels = (unsigned char*)malloc(sizeof(unsigned char)*len_labels_array);

    for (int i = 0; i < num_labels; i++) {
        unsigned char temp;
        fread(&temp, sizeof(unsigned char), 1, fp);
        labels[i] = temp;
    }
    fclose(fp);

    printf("label load successful\n");
    return labels;
}

// 读取CIFAR-10数据集的函数
int read_cifar10(FILE *fp, unsigned char *data, unsigned char *labels, int batch, int* num) {
    const int bytes_per_image = 1 + CIFAR10_IMAGE_SIZE; // 每张图片大小（3*32*32+1，其中1是标签)
    unsigned char buffer[1 + CIFAR10_IMAGE_SIZE];

    int i;
    for(i = 0; i < batch; i++){
        if (fread(buffer, bytes_per_image, 1, fp) == 1) {
            labels[i] = buffer[0];
            memcpy(&data[i*CIFAR10_IMAGE_SIZE], &buffer[1], sizeof(unsigned char)*CIFAR10_IMAGE_SIZE);
        } else {
            break;
        }
    }

    *num = i;
    return 0;
}