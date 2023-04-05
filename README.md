# FPGAI

本仓库实现了一个简单的量化算法，下面给出训练量化模型的方法：

## python

model文件夹下保存模型，量化模型和numpy模型的源码，data文件夹下保存数据集。

### 1、训练

- 运行train_model，训练模型，其中模型的选择和配置可以在config.py文件中配置，默认情况下模型保存为./pretrained/{model_name}_raw.pth

### 2、量化

> 需存在训练好的raw模型

- 运行quant_model，进行模型的量化，量化后模型默认保存至./quantization/{model_name}_quant.tuple，tuple中为（torch.state_dict, quant_list）

### 3、测试

> 测试前需存在对应的模型（）

- 运行test_model测试未量化模型、量化模型和numpy量化模型，也可更改config文件以只测试其中一个

### 4、导出

- 运行export_quant_model，导出量化模型的txt文本保存。权重和量化参数的对应默认保存路径为：`./export_model/{model_name}_weight.txt` 和 `./export_model/{model_name}_quant.txt`。

  其中weight的格式为，按层数以此导出，每10个数据一行，逗号分割。每层的weight前有单独一行#，每层的bias前有单独一行$。

  其中bias的格式为，每行三个数据，逗号分割，分别对应scale，shift，zero_point。
  
- 考虑到模型体积问题，增加以二进制格式导出的方式，weight为8位无符号整型，bias为32位有符号整型，quant依然为文本格式导出

## C

### 1、推理

在量化模型训练好的情况下，编译inference.c并运行，其中所欲运行的模型可在inference最前面部分选择

## 模型

目前已经支持的模型有：

| 模型名称     | raw  | quant         | np quant | C·INT8 |
| ------------ | ---- | ------------- | -------- | ------ |
| lenet        | √    | √             | √        | √      |
| mobilenet_v2 | √    | √             |          |        |
| ResNet       | √    | √             |          | √      |
| vgg          | √    | √（还需调试） |          |        |

笔者在export目录下提供了可以直接在C程序中运行的权重和bias的bin文件以及对应的quant的txt文件，以支持直接运行

****

**lenet网络结构**：

![img](https://img-blog.csdnimg.cn/d0db1f76faf044cfb9eeff0026df5f5c.png)

**mobilenet_v2网络结构**：

其中第三层，即参数为6 24 2 2的层，当所用数据集为CIFAR-10的时候，stride需要改为1

![img](https://img-blog.csdnimg.cn/20210516115659558.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDc1MTI5NA==,size_16,color_FFFFFF,t_70)

**ResNet结构**：

18和34用的是BasicBlock，50层以后的用的是Bottleneck

![img](https://pic1.zhimg.com/v2-181cd2dc1d4dc7f3cb05f844d96017f4_b.jpg)

**注：下图为 https://www.jianshu.com/p/085f4c8256f1 中的参考图，ResNet50图片细节似乎有些问题，只做理解参考**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191219110451136.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NoZXVuZ2xlaWxlaQ==,size_16,color_FFFFFF,t_70)

## 参考资料

[1] https://blog.csdn.net/weixin_43530173/article/details/124598231
