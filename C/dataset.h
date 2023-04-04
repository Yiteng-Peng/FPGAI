#ifndef DATASET
#define DATASET
#include <stdio.h>

#define IMG_TYPE unsigned char
#define MNIST_IMAGE_SIZE 784
#define CIFAR10_IMAGE_SIZE 3072

IMG_TYPE* read_mnist_images(const char *filename, int* num_images, int* num_rows, int* num_cols);
unsigned char* read_mnist_labels(const char *filename);
int read_cifar10(FILE *fp, unsigned char *data, unsigned char *labels, int batch, int* num);

#endif