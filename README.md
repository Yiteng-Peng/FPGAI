# FPGAI

本仓库实现了一个简单的量化算法，下面给出训练量化模型的方法

## python

1、首先运行train_model，训练模型，其中模型的选择和配置可以在config.py文件中配置，默认情况下模型保存为./pretrained/{model_name}_raw.pth

2、运行quant_model，进行模型的量化，量化后模型保存至./quantization/{model_name}_quant.pth

3、运行test_model测试量化模型和训练好的模型，也可更改config文件以只测试其中一个

## C

1、在量化模型训练好的情况下，编译inference.c并运行，其中所欲运行的模型可在config.h中选择

## 参考资料

[1] https://blog.csdn.net/weixin_43530173/article/details/124598231