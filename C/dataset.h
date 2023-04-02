#ifndef DATASET
#define DATASET

#define IMG_TYPE unsigned char

IMG_TYPE* read_mnist_images(const char *filename, int* num_images, int* num_rows, int* num_cols);
unsigned char* read_mnist_labels(const char *filename);

#endif