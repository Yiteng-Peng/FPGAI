#ifndef QUANT
#define QUANT

#define TYPE unsigned char
#define TYPE_MIN UCHAR_MAX

typedef struct{
    int N;
    int C;
    int H;
    int W;
}Shape;

TYPE ListMax(TYPE* x, int len);
TYPE ListMean(TYPE* x, int len);
TYPE* GetMatrix2d(TYPE* x, Shape shape, int n, int c, int h_start, int h_end, int w_start, int w_end);
int LinearMultiple(TYPE* x, unsigned char* y, unsigned char zero_point, int len);
void Pad(TYPE** x, int padding, Shape* shape);

typedef struct{
    unsigned int in_features;
    unsigned int out_features;

    unsigned char scale;
    unsigned char shift;
    unsigned char zero_point;
    
    unsigned char* quant_weight;
    int* quant_bias;
}QuantLinear;

int* QuantLinear_forward(QuantLinear fc, TYPE* x, Shape* shape);

typedef struct{
    unsigned char kernel_size;
    unsigned char stride;
}QuantMaxPool2d;

TYPE* QuantMaxPool2d_forward(QuantMaxPool2d pool, TYPE* x, Shape* shape);

typedef struct{
    unsigned char kernel_size;
    unsigned char stride;
}QuantAvePool2d;

TYPE* QuantAvePool2d_forward(QuantAvePool2d pool, TYPE* x, Shape* shape);

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

int* QuantConv2d_forward(QuantConv2d conv, TYPE* x, Shape* shape);

#endif