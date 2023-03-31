## 模型数据存储格式

两个txt文件

**{model name}_param.txt**

存储模型的量化后weight和bias

每层参数对应两行，第一行存储该层weight，第二行存储该层bias

**{model name}_quant.txt**

存储模型的量化参数

每行3个数字，用逗号隔开，不加空格

分别代表：scale，shift，zero_point