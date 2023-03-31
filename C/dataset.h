#ifndef DATASET
#define DATASET

int* read_mnist_images(const char *filename, int* num_images, int* num_rows, int* num_cols);
unsigned char* read_mnist_labels(const char *filename);

#endif