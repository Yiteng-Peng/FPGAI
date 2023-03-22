#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// define the dimensions of the input image
#define IMG_WIDTH 32
#define IMG_HEIGHT 32
#define IMG_DEPTH 1

// define the number of filters in each convolutional layer
#define NUM_FILTERS_LAYER1 6
#define NUM_FILTERS_LAYER2 16

// define the dimensions of the filters in each convolutional layer
#define FILTER_SIZE_LAYER1 5
#define FILTER_SIZE_LAYER2 5

// define the size of the max pooling window
#define POOL_SIZE 2

// define the number of nodes in the fully connected layers
#define NUM_HIDDEN1 120
#define NUM_HIDDEN2 84

// define the number of classes in the output layer
#define NUM_CLASSES 10

// define the learning rate
#define LEARNING_RATE 0.01

// define the activation function (sigmoid)
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// define the derivative of the activation function (sigmoid)
double sigmoid_prime(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

// define the convolution operation
void conv(double input[][IMG_WIDTH][IMG_DEPTH], double filter[][FILTER_SIZE_LAYER1][FILTER_SIZE_LAYER1], double output[][IMG_WIDTH - FILTER_SIZE_LAYER1 + 1][NUM_FILTERS_LAYER1]) {
    for (int f = 0; f < NUM_FILTERS_LAYER1; f++) {
        for (int i = 0; i < IMG_WIDTH - FILTER_SIZE_LAYER1 + 1; i++) {
            for (int j = 0; j < IMG_WIDTH - FILTER_SIZE_LAYER1 + 1; j++) {
                double sum = 0.0;
                for (int k = 0; k < FILTER_SIZE_LAYER1; k++) {
                    for (int l = 0; l < FILTER_SIZE_LAYER1; l++) {
                        for (int d = 0; d < IMG_DEPTH; d++) {
                            sum += input[i+k][j+l][d] * filter[f][k][l];
                        }
                    }
                }
                output[i][j][f] = sigmoid(sum);
            }
        }
    }
}

// define the max pooling operation
void max_pool(double input[][IMG_WIDTH - FILTER_SIZE_LAYER1 + 1][NUM_FILTERS_LAYER1], double output[][IMG_WIDTH/POOL_SIZE - FILTER_SIZE_LAYER1/POOL_SIZE + 1][NUM_FILTERS_LAYER1]) {
    for (int f = 0; f < NUM_FILTERS_LAYER1; f++) {
        for (int i = 0; i < IMG_WIDTH/POOL_SIZE - FILTER_SIZE_LAYER1/POOL_SIZE + 1; i++) {
            for (int j = 0; j < IMG_WIDTH/POOL_SIZE - FILTER_SIZE_LAYER1/POOL_SIZE + 1; j++) {
                double max_val = input[i*POOL_SIZE][j*POOL_SIZE][f];
                for (int k = 0; k < POOL_SIZE; k++) {
                    for (int l = 0; l < POOL_SIZE; l++) {
                        max_val = fmax(max_val, input[i*POOL_SIZE+k][j*POOL_SIZE+l][f]);
                    }
                }
                output[i][j][f] = max_val;
            }
        }
    }
}

// define the fully connected layer
void fully_connected(double input[], double weights[][NUM_HIDDEN1], double output[], int num_inputs, int num_outputs) {
    for (int
