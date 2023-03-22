#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// 以下是一个简单的lenet5的结构定义
typedef struct {
    int input_width;
    int input_height;
    int map_size;
    int in_channels;
    int out_channels;
    float**** map_data; // 4维数组，存储每个输出特征图的数据
    float**** kernel_data; // 4维数组，存储每个卷积核的数据
    float* bias_data; // 1维数组，存储每个输出特征图的偏置项
} conv_layer;

typedef struct {
    int input_width;
    int input_height;
    int map_size;
    int in_channels;
    int out_channels;
    float**** map_data; // 4维数组，存储每个输出特征图的数据
} pool_layer;

typedef struct {
    int input_num; // 输入向量的维数
    int output_num; // 输出向量的维数
    float** weight_data; // 2维数组，存储权重矩阵
    float* bias_data; // 1维数组，存储偏置向量
} fc_layer;

// 定义一个lenet5结构体，包含两个卷积层、两个池化层和三个全连接层
typedef struct {
   conv_layer* C1;
   pool_layer* S2;
   conv_layer* C3;
   pool_layer* S4;
   fc_layer* F5;
   fc_layer* F6;
   fc_layer* OUTPUT;
} lenet5;

// 初始化一个lenet5网络，并分配内存空间
lenet5* initial_lenet()
{
     lenet5* net = (lenet5*)malloc(sizeof(lenet5));
     net->C1 = initial_conv_layer(32,32,5,1,6); 
     net->S2 = initial_pooling_layer(28,28,2,6);
     net->C3 = initial_conv_layer(14,14,5,6,16);
     net->S4 = initial_pooling_layer(10,10,2,16);
     net->F5 = initial_fc_layer(400,120);
     net->F6 = initial_fc_layer(120.84);
     net->OUTPUT = initial_fc_output(84.10);

     return net;

}

// 使用CUDA编程模型，在GPU上实现卷积层的前向传播函数（仅展示部分代码）
__global__ void conv_forward_kernel(float**** input,float**** output,float**** kernel,float* bias,int width,int height,int kernel_size,int in_channel,int out_channel)
{
      // 获取线程索引（x,y,z）
      const unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x; 
      const unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y; 
      const unsigned int zIndex = blockIdx.z * blockDim.z + threadIdx.z;

      if(xIndex < width && yIndex < height && zIndex < out_channel) 
      {
          float sum=0.0f;//用于累加卷积结果

          for(int i=0;i<in_channel;i++)//遍历输入特征图通道数
          {
              for(int m=0;m<kernel_size;m++)//遍历卷积核行数
                for(int n=0;n<kernel_size;n++)//遍历卷积核列数
                    {
                        //计算卷积结果
                        sum+=input[i][yIndex+m][xIndex+n]*kernel[zIndex][i][m][n];
                    }
                }
            }

          //加上偏置项，并使用激活函数（这里使用ReLU）
          output[zIndex][yIndex][xIndex]=max(0.0f,sum+bias[zIndex]);
      }

// 在主函数中，调用卷积层前向传播函数，并将数据从CPU复制到GPU
void conv_forward(conv_layer* layer,float**** input)
{
     //定义GPU上的变量指针
     float**** d_input;
     float**** d_output;
     float**** d_kernel;
     float* d_bias;

     //计算输出特征图的宽度和高度
     int output_width=layer->input_width-layer->map_size+1;
     int output_height=layer->input_height-layer->map_size+1;

     //为GPU上的变量分配内存空间，并将CPU上的数据复制到GPU
     cudaMalloc((void**)&d_input,sizeof(float)*layer->in_channels*layer->input_height*layer->input_width);
     cudaMemcpy(d_input,input,sizeof(float)*layer->in_channels*layer->input_height*layer->input_width,cudaMemcpyHostToDevice);

     cudaMalloc((void**)&d_output,sizeof(float)*layer->out_channels*output_height*output_width);
     
     cudaMalloc((void**)&d_kernel,sizeof(float)*layer->out_channels*layer->in_channels*layer->map_size*layer->map_size);
     cudaMemcpy(d_kernel,layer->kernel_data,sizeof(float)*layer->out_channels*layer->in_channels*layer->map_size*layer->map_size,cudaMemcpyHostToDevice);

     cudaMalloc((void**)&d_bias,sizeof(float)*layer-out_channels);
     cudaMemcpy(d_bias, layer-bias_data,sizeof(float)* layer-out_channels,cudaMemcpyHostToDevice);

    //定义线程块和网格的大小
    dim3 threadsPerBlock(16,16,4); 
    dim3 numBlocks(output_width / threadsPerBlock.x, output_height / threadsPerBlock.y, out_channel / threadsPerBlock.z); 

    //在GPU上执行卷积层前向传播函数
    conv_forward_kernel<<<numBlocks,threadsPerBlock>>>(d_input,d_output,d_kernel,d_bias,output_width,output_height,kernel_size,in_channel,out_channel);

    //将输出特征图从GPU复制回CPU，并释放GPU上的内存空间
    cudaMemcpy(layer-map_data,d_output,sizeof(float)* layer-out_channels * output_height * output_width,cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    cudaFree(d_bias);

}