#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define IMAGE_WIDTH 224
#define IMAGE_HEIGHT 224
#define NUM_CLASSES 1000

typedef struct {
    float data[IMAGE_WIDTH * IMAGE_HEIGHT * 3];
} image;

typedef struct {
    float data[NUM_CLASSES];
} prediction;

typedef struct {
    float weights[3][3][3][64];
    float biases[64];
} conv1_params;

typedef struct {
    float weights[3][3][64][64];
    float biases[64];
} conv2_params;

typedef struct {
    float weights[3][3][64][128];
    float biases[128];
} conv3_params;

typedef struct {
    float weights[3][3][128][128];
    float biases[128];
} conv4_params;

typedef struct {
    float weights[3][3][128][256];
    float biases[256];
} conv5_params;

typedef struct {
    float weights[3][3][256][256];
    float biases[256];
} conv6_params;

typedef struct {
    float weights[3][3][256][512];
    float biases[512];
} conv7_params;

typedef struct {
    float weights[3][3][512][512];
    float biases[512];
} conv8_params;

typedef struct {
    float weights[512][NUM_CLASSES];
    float biases[NUM_CLASSES];
} fc_params;

typedef struct {
    conv_params conv1_params;
    conv_params conv2_params;
    conv_params shortcut_params;
    int in_channels;
    int out_channels;
    int height;
    int width;
} resblock_params;

void load_image(const char *filename, image *img) {
    FILE *file = fopen(filename, "rb");

    if (!file) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT * 3; i++) {
        unsigned char byte;
        fread(&byte, sizeof(unsigned char), 1, file);
        img->data[i] = (float)byte / 255.0f;
    }

    fclose(file);
}

void conv1(const image *input, float output[112][112][64], const conv1_params *params) {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 112; j++) {
            for (int k = 0; k < 112; k++) {
                output[j][k][i] = params->biases[i];
            }
        }
    }

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 3; l++) {
                    for (int m = 0; m < 112; m++) {
                        for (int n = 0; n < 112; n++) {
                            output[m][n][i] += input->data[(m+j)*IMAGE_WIDTH*3 + (n+k)*3 + l] * params->weights[j][k][l][i];
                        }
                    }
                }
            }
        }
    }
}

void conv2(const float input[112][112][64], float output[56][56][64], const conv2_params *params) {
    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 56; j++) {
            for (int k = 0; k < 56; k++) {
                output[j][k][i] = params->biases[i];
            }
        }
    }

    for (int i = 0; i < 64; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 64; l++) {
                    for (int m = 0; m < 56; m++) {
                        for (int n = 0; n < 56; n++) {
                            output[m][n][i] += input[m+j][n+k][l] * params->weights[j][k][l][i];
                        }
                    }
                }
            }
        }
    }
}

void conv3(const float input[56][56][64], float output[28][28][128], const conv3_params *params) {
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 28; j++) {
            for (int k = 0; k < 28; k++) {
                output[j][k][i] = params->biases[i];
            }
        }
    }

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 64; l++) {
                    for (int m = 0; m < 28; m++) {
                        for (int n = 0; n < 28; n++) {
                            output[m][n][i] += input[m+j][n+k][l] * params->weights[j][k][l][i];
                        }
                    }
                }
            }
        }
    }
}
        
void conv4(const float input[28][28][128], float output[14][14][128], const conv4_params *params) {
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 14; j++) {
            for (int k = 0; k < 14; k++) {
                output[j][k][i] = params->biases[i];
            }
        }
    }

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 128; l++) {
                    for (int m = 0; m < 14; m++) {
                        for (int n = 0; n < 14; n++) {
                            output[m][n][i] += input[m+j][n+k][l] * params->weights[j][k][l][i];
                        }
                    }
                }
            }
        }
    }
}

void conv5(const float input[14][14][128], float output[7][7][256], const conv5_params *params) {
    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 7; k++) {
                output[j][k][i] = params->biases[i];
            }
        }
    }

    for (int i = 0; i < 256; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                for (int l = 0; l < 128; l++) {
                    for (int m = 0; m < 7; m++) {
                        for (int n = 0; n < 7; n++) {
                            output[m][n][i] += input[m+j][n+k][l] * params->weights[j][k][l][i];
                        }
                    }
                }
            }
        }
    }
}

void resblock(const float *input, float *output, const resblock_params *params) {
    float *conv1_output;
    float *conv2_output;
    float *shortcut_output;

    conv1(input, conv1_output, &params->conv1_params);
    relu(conv1_output);

    conv2(conv1_output, conv2_output, &params->conv2_params);
    relu(conv2_output);

    conv3(conv2_output, output, &params->conv3_params);
    relu(output);

    // Shortcut connection
    conv4(input, shortcut_output, &params->shortcut_params);
    conv5(shortcut_output, shortcut_output, &params->conv5_params);

    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            for (int k = 0; k < 256; k++) {
                output[i][j][k] += shortcut_output[i][j][k];
            }
        }
    }

    relu(output);
}