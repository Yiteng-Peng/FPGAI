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

- 运行export_quant_model，导出量化模型的txt文本保存。权重和量化参数的对应默认保存路径为：\./txt\_model/\{model\_name\}\_weight\.txt和\./txt\_model/\{model_name\}\_quant\.txt。

  其中weight的格式为，按层数以此导出，每10个数据一行，逗号分割。每层的weight前有单独一行#，每层的bias前有单独一行$。

  其中bias的格式为，每行三个数据，逗号分割，分别对应scale，shift，zero_point。

## C

### 1、推理

在量化模型训练好的情况下，编译inference.c并运行，其中所欲运行的模型可在config.h中选择

## 参考资料

[1] https://blog.csdn.net/weixin_43530173/article/details/124598231