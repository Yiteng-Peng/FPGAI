#ifndef QUANT
#define QUANT

typedef struct{
    int N;
    int C;
    int H;
    int W;
}Shape;

int ListMax(int* x, int len);
int ListMean(int* x, int len);
int* GetMatrix2d(int* x, Shape shape, int n, int c, int h_start, int h_end, int w_start, int w_end);
int LinearMultiple(int* x, unsigned char* y, unsigned char zero_point, int len);
void Pad(int** x, int padding, Shape* shape);

typedef struct{
    unsigned int in_features;
    unsigned int out_features;

    unsigned char scale;
    unsigned char shift;
    unsigned char zero_point;
    
    unsigned char* quant_weight;
    int* quant_bias;
}QuantLinear;

void QuantLinear_forward(QuantLinear fc, int** x, int N);

typedef struct{
    unsigned char kernel_size;
    unsigned char stride;
}QuantMaxPool2d;

void QuantMaxPool2d_forward(QuantMaxPool2d pool, int** x, Shape* shape);

typedef struct{
    unsigned char kernel_size;
    unsigned char stride;
}QuantAvePool2d;

void QuantAvePool2d_forward(QuantAvePool2d pool, int** x, Shape* shape);

typedef struct{
    unsigned int in_channels;
    unsigned int out_channels;
    unsigned int kernel_size;
    
    char stride;
    char padding;
    
    unsigned char scale;
    unsigned char shift;
    unsigned char zero_point;
    
    unsigned char* quant_weight;
    int* quant_bias;
}QuantConv2d;

void QuantConv2d_forward(QuantConv2d conv, int** x, Shape* shape);

#endif