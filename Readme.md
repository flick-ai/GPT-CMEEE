# AI3612:GPT-CMEEE

这是我们知识表征与推理课程第二次大作业的github储存仓库，用于实现从CMEEE实体识别。

## 环境说明
运行本次作业代码的环境与第一次大作业的环境基本一直，但是您需要额外的两个库：openai和kmeans—pytorch。对于第二个库，我们进行了一些私人的更改，这里建议您从我们自己的仓库git下来并安装：
```shell
pip install openai # 安装openai
git clone https://github.com/flick-ai/kmeans_pytorch.git # git我们的kmeans—pytorch
cd kmeans_pytorch
pip install --editable . # 安装kmeans—pytorch
```
## 文件说明
我们最终的代码目录包含以下文件夹：

+ `dataset.py`:构建我们需要使用的数据格式
+ `get_embedding.py`:构建训练池的脚本
+ `get_response.py`:与ChatGPT交互的脚本
+ `main.py`:获取dev集上结果的脚本
+ `metric.py`:进行指标评价的脚本
## 代码运行
如果您想执行我们的代码，请按照如下格式执行：
```shell
python get_embedding.py # 构建训练样本池（如果您已经有select_train.json,则不需要运行该脚本）
python main.py # 获得dev集上的结果文件（如果您已经有output.json,则不需要运行该脚本）
python metric.py # 计算dev集上指标
```

## Acknowledgement

我们的实现参考了我们自己的第一次大作业代码: [CMEEE](https://github.com/1364406834/Knowledge-Representation-and-Reasoning). 如果您对我们的工作感兴趣，请访问第一次作业的仓库了解更多。