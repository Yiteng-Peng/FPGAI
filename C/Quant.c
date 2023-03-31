#define _CRT_SECURE_NO_WARNINGS

#include "Quant.h"
#include "LeNet.h"
#include <stdio.h>
#include <stdlib.h>

int ListMax(int* x, int len){
    int max = INT_MIN;

    int i = 0;
    for(i = 0; i < len; i++){
        if(max < x[i])
            max = x[i];
    }
    return max;
}

int ListMean(int* x, int len){
    int sum = 0;

    int i = 0;
    for(i = 0; i < len; i++){
        sum += x[i];
    }
    return sum / len;
}

int* GetMatrix2d(int* x, Shape shape, int n, int c, int h_start, int h_end, int w_start, int w_end){
    int slice_len = (h_end-h_start)*(w_end-w_start);
    int* slice = (int*)malloc(sizeof(int)*slice_len);
    int base_index = n*shape.C*shape.H*shape.W + c*shape.H*shape.W;

    int i,j; int count = 0; int index = 0;
    for(i = h_start; i < h_end; i++){
        for(j = w_start; j < w_end; j++){
            index = base_index + i * shape.W + j;
            slice[count] = x[index];
            count++;
        }
    }
    return slice;
}

int LinearMultiple(int* x, unsigned char* y, unsigned char zero_point, int len){
    int result = 0;
    int i = 0;
    for(i = 0; i < len; i++){
        result += x[i] * y[i];
        result -= x[i] * zero_point;
    }
    return result;
}

void Pad(int** x, int padding, Shape* shape){
    if(padding == 0){return;}

    int* in_x = *x;
    int* out_x = (int*)malloc(sizeof(int)*shape->N*shape->C*(shape->H+2*padding)*(shape->W+2*padding));

    int N_i,C_i,H_i,W_i;
    for(N_i = 0; N_i < shape->N; N_i++){
        for(C_i = 0; C_i < shape->C; C_i++){
            int pad_base_index = (shape->H + 2 * padding) * (shape->W + 2 * padding);
            pad_base_index = N_i * shape->C * pad_base_index + C_i * pad_base_index;
            int base_index = shape->H * shape->W;
            base_index = N_i * shape->C * base_index + C_i * base_index;

            for(H_i = 0; H_i < shape->H + 2*padding; H_i++){
                for(W_i = 0; W_i < shape->W + 2*padding; W_i++){
                    int pad_index = H_i * (shape->W+2*padding) + W_i + pad_base_index;
                    int index = (H_i-padding) * shape->W + W_i-padding + base_index;
                    if (H_i < padding || W_i < padding || H_i >= shape->H + padding || W_i >= shape->W + padding){
                        out_x[pad_index] = 0;
                    } else {
                        out_x[pad_index] = in_x[index];
                    }
                }
            }
        }
    }

    free(in_x);
    shape->H = shape->H+2*padding; shape->W = shape->W+2*padding;
    *x = out_x;
}

void QuantLinear_forward(QuantLinear fc, int** x, int N){
    int* in_x = *x;
    int* out_x = (int*)malloc(sizeof(int) * N * fc.out_features);

    int i, j;
    for(i = 0; i < N; i++){
        for(j = 0; j < fc.out_features; j++){
            int index = i*fc.out_features + j;
            out_x[index] = LinearMultiple(&in_x[i*fc.in_features], &fc.quant_weight[j*fc.in_features], fc.zero_point, fc.in_features);
            out_x[index] = out_x[index] + fc.quant_bias[j];
            out_x[index] = out_x[index] * fc.scale;
            out_x[index] = out_x[index] >> fc.shift;
        }
    }

    free(in_x);
    *x = out_x;
}

void QuantAvePool2d_forward(QuantAvePool2d pool, int** x, Shape* shape){
    int H_out = (shape->H - pool.kernel_size) / pool.stride + 1;
    int W_out = (shape->W - pool.kernel_size) / pool.stride + 1;

    int* in_x = *x;
    int* out_x = (int*)malloc(sizeof(int) * shape->N * shape->C * H_out * W_out);

    int num_C = H_out * W_out;
    int num_N = shape->C * num_C;
    int base_index = 0;

    int H_i, W_i, N_i, C_i;
    for(H_i = 0; H_i < H_out; H_i++){
        for(W_i = 0; W_i < W_out; W_i++){
            int h_start = H_i * pool.stride;
            int h_end = h_start + pool.kernel_size;
            int w_start = W_i * pool.stride;
            int w_end = w_start + pool.kernel_size;

            base_index = H_i * W_out + W_i;

            for(N_i = 0; N_i < shape->N; N_i++){
                for(C_i = 0; C_i < shape->C; C_i++){
                    int* slice = GetMatrix2d(*x, *shape, N_i, C_i, h_start, h_end, w_start, w_end);
                    out_x[N_i*num_N + C_i*num_C + base_index] = ListMean(slice, pool.kernel_size*pool.kernel_size);
                    free(slice);
                }
            }
        }
    }

    free(in_x);
    shape->H = H_out;shape->W = W_out;
    *x = out_x;
}

void QuantMaxPool2d_forward(QuantMaxPool2d pool, int** x, Shape* shape){
    int H_out = (shape->H - pool.kernel_size) / pool.stride + 1;
    int W_out = (shape->W - pool.kernel_size) / pool.stride + 1;

    int* in_x = *x;
    int* out_x = (int*)malloc(sizeof(int) * shape->N * shape->C * H_out * W_out);

    int num_C = H_out * W_out;
    int num_N = shape->C * num_C;
    int base_index = 0;

    int H_i, W_i, N_i, C_i;
    for(H_i = 0; H_i < H_out; H_i++){
        for(W_i = 0; W_i < W_out; W_i++){
            int h_start = H_i * pool.stride;
            int h_end = h_start + pool.kernel_size;
            int w_start = W_i * pool.stride;
            int w_end = w_start + pool.kernel_size;

            base_index = H_i * W_out + W_i;

            for(N_i = 0; N_i < shape->N; N_i++){
                for(C_i = 0; C_i < shape->C; C_i++){
                    int* slice = GetMatrix2d(*x, *shape, N_i, C_i, h_start, h_end, w_start, w_end);
                    out_x[N_i*num_N + C_i*num_C + base_index] = ListMax(slice, pool.kernel_size*pool.kernel_size);
                    free(slice);
                }
            }
        }
    }

    free(in_x);
    shape->H = H_out;shape->W = W_out;
    *x = out_x;
}

void QuantConv2d_forward(QuantConv2d conv, int** x, Shape* shape){
    Pad(x, conv.padding, shape);
    int H_out = (shape->H - conv.kernel_size) / conv.stride + 1;
    int W_out = (shape->W - conv.kernel_size) / conv.stride + 1;

    int* in_x = *x;
    int* out_x = (int*)malloc(sizeof(int) * shape->N * conv.out_channels * H_out * W_out);

    int num_C = H_out * W_out;
    int num_N = conv.out_channels * num_C;
    int base_index = 0;

    int H_i, W_i, N_i, C_i, C_o;
    for(H_i = 0; H_i < H_out; H_i++){
        for(W_i = 0; W_i < W_out; W_i++){
            int h_start = H_i * conv.stride;
            int h_end = h_start + conv.kernel_size;
            int w_start = W_i * conv.stride;
            int w_end = w_start + conv.kernel_size;

            base_index = H_i * W_out + W_i;

            for(N_i = 0; N_i < shape->N; N_i++){
                for(C_o = 0; C_o < conv.out_channels; C_o++){
                    int index = N_i*num_N + C_o*num_C + base_index;
                    out_x[index] = 0;
                    int conv_size = conv.kernel_size * conv.kernel_size;
                    for(C_i = 0; C_i < shape->C; C_i++){
                        int* slice = GetMatrix2d(*x, *shape, N_i, C_i, h_start, h_end, w_start, w_end);
                        out_x[index] = out_x[index] + LinearMultiple(slice, &conv.quant_weight[(C_o*shape->C + C_i)*conv_size], conv.zero_point, conv_size);
                        free(slice);
                    }
                    out_x[index] = out_x[index] + conv.quant_bias[C_o];
                    out_x[index] = out_x[index] * conv.scale;
                    out_x[index] = out_x[index] >> conv.shift;
                }
            }
        }
    }

    free(in_x);
    shape->C = conv.out_channels;shape->H = H_out;shape->W = W_out;
    *x = out_x;
}