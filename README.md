# AI学习文档整理

## 基础

* numpy
* python

## 算力

* Aliyun
* Huaweicloud
* AWS
* Azure
* PC

## 环境

由于需要在多个按量计费的云环境中动态使用GPU主机（不用的时候停止，以节省费用），希望能构建一个可以在多个AI主机上快速切换的conda环境，多个主机共享conda和用户环境，可以任意启动多个实例，docker容器停止后用户或应用工作成果不丢失，切换到其他主机后可以继续工作。

构建三个基础容器镜像：

1. [cuda](): 安装宿主机驱动，安装和验证cuda docker运行环境

2. [conda](): 基于cuda环境构建公共conda环境

3. [jupyterlab](): 基于conda环境构建一个公共的jupyterlab, ai 试验环境。

> 后续所有试验内容基于本环境进行

## 模型

* chatglm

* echomimic
* funasr
* SAM

## 框架

* pytorch

* jax
* tensorflow

## 训练

* wandb
* deepspeed

## 运行部署

* accelerate
* ollama
* huggingface

## 应用

* aigateway
* 数字人

## 工具

* 数据标注



## 其他