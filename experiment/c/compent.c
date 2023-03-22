#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 定义一个函数来实现卷积操作
void conv2d(float *input, float *output, float *kernel, int input_height, int input_width, int input_channels, int output_height, int output_width, int output_channels, int kernel_size, int stride) {
    // 循环遍历输出特征图的每个通道
    for (int oc = 0; oc < output_channels; oc++) {
        // 循环遍历输出特征图的每个像素位置
        for (int oh = 0; oh < output_height; oh++) {
            for (int ow = 0; ow < output_width; ow++) {
                // 初始化输出特征图的当前像素值为0
                float sum = 0.0;
                // 循环遍历输入特征图的每个通道
                for (int ic = 0; ic < input_channels; ic++) {
                    // 循环遍历卷积核的每个元素
                    for (int kh = 0; kh < kernel_size; kh++) {
                        for (int kw = 0; kw < kernel_size; kw++) {
                            // 计算输入特征图中对应的像素位置
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            // 计算输入特征图和卷积核在内存中的索引
                            int input_index = ic * input_height * input_width + ih * input_width + iw;
                            int kernel_index = oc * input_channels * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                            // 累加输入特征图和卷积核对应元素的乘积
                            sum += input[input_index] * kernel[kernel_index];
                        }
                    }
                }
                // 计算输出特征图在内存中的索引
                int output_index = oc * output_height * output_width + oh * output_width + ow;
                // 将累加结果赋值给输出特征图
                output[output_index] = sum;
            }
        }
    }
}

// 定义一个函数来实现池化操作（最大池化）
void maxpool2d(float*input,float*output,int input_height,int input_width,int input_channels,int output_height,int output_width,int pool_size,int stride){
    // 循环遍历输入特征图的每个通道
    for(int ic=0;ic<input_channels;ic++){
        // 循环遍历输出特征图的每个像素位置
        for(int oh=0;oh<output_height;oh++){
            for(int ow=0;ow<output_width;ow++){
                // 初始化输出特征图的当前像素值为负无穷大（或者一个足够小的数）
                float max=-INFINITY;//或者float max=-100000.0;
                // 循环遍历池化窗口内的每个元素
                for(int ph=0;ph<pool_size;ph++){
                    for(int pw=0,pw<pool_size,pw++){
                        // 计算输入特征图中对应的像素位置
                        int ih=oh*stride+ph;
                        int iw=ow*stride+pw;
                        // 计算输入特征图在内存中的索引
                        int input_index=ic*input_height*input_width+ih*input_width+iw;
                        // 更新输出特征图当前像素值为池化窗口内最大值（如果有必要）
                        if(input[input_index]>max){
                            max=input[input_index];
                        }
                    }
                }
                //



// 定义一个函数来实现残差链接操作
void residual_connection(float *input, float *output, float *residual, int height, int width, int channels) {
    // 循环遍历输入特征图和残差特征图的每个通道
    for (int c = 0; c < channels; c++) {
        // 循环遍历输入特征图和残差特征图的每个像素位置
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                // 计算输入特征图、残差特征图和输出特征图在内存中的索引
                int index = c * height * width + h * width + w;
                // 将输入特征图和残差特征图对应元素相加，并赋值给输出特征图
                output[index] = input[index] + residual[index];
            }
        }
    }
}