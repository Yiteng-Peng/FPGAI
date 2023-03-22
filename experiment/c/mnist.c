#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_TRAIN_IMAGES 60000
#define NUM_TEST_IMAGES 10000
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28

typedef struct {
    unsigned char data[IMAGE_WIDTH * IMAGE_HEIGHT];
    unsigned char label;
} mnist_image;

void read_mnist_images(const char *filename, mnist_image *images, int num_images) {
    FILE *file = fopen(filename, "rb");

    if (!file) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }

    unsigned char header[16];
    fread(header, sizeof(unsigned char), 16, file);

    for (int i = 0; i < num_images; i++) {
        fread(images[i].data, sizeof(unsigned char), IMAGE_WIDTH * IMAGE_HEIGHT, file);
    }

    fclose(file);
}

void read_mnist_labels(const char *filename, unsigned char *labels, int num_labels) {
    FILE *file = fopen(filename, "rb");

    if (!file) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }

    unsigned char header[8];
    fread(header, sizeof(unsigned char), 8, file);

    for (int i = 0; i < num_labels; i++) {
        fread(&labels[i], sizeof(unsigned char), 1, file);
    }

    fclose(file);
}

int main() {
    mnist_image train_images[NUM_TRAIN_IMAGES];
    mnist_image test_images[NUM_TEST_IMAGES];
    unsigned char train_labels[NUM_TRAIN_IMAGES];
    unsigned char test_labels[NUM_TEST_IMAGES];

    read_mnist_images("train-images-idx3-ubyte", train_images, NUM_TRAIN_IMAGES);
    read_mnist_images("t10k-images-idx3-ubyte", test_images, NUM_TEST_IMAGES);
    read_mnist_labels("train-labels-idx1-ubyte", train_labels, NUM_TRAIN_IMAGES);
    read_mnist_labels("t10k-labels-idx1-ubyte", test_labels, NUM_TEST_IMAGES);

    // do something with the MNIST data here

    return 0;
}
