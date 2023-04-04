#ifndef LOAD
#define LOAD
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Quant.h"

void txt_load_quant(FILE* fp, unsigned char* scale, unsigned char* shift, unsigned char* zero_point);

void txt_load_tag_check(FILE* fp, char check);
void txt_load_uint8(unsigned char** list, FILE* fp, int num);
void txt_load_int(int** list, FILE* fp, int num);

void bin_load_uint8(unsigned char** list, FILE* fp, int num);
void bin_load_int(int** list, FILE* fp, int num);

void bin_load_conv(QuantConv2d* layer, FILE* fp_weight, FILE* fp_bias);
void bin_load_fc(QuantLinear* layer, FILE* fp_weight, FILE* fp_bias);

#endif