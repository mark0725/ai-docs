# AI学习笔记

## 基础

* numpy:
* python:
* jupyterlab:JupyterLab is the latest web-based interactive development environment for notebooks, code, and data. [官网](https://jupyter.org/)

## 算力

* Aliyun
* Huaweicloud： [Modelarts](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard)提供免费GPU学习环境，但是gpu比较老，适合简单基础学习。长时间运行模型最好购买GPU服务器。
* AWS
* Azure
* PC

## 环境

由于需要在多个按量计费的云环境中动态使用GPU主机（不用的时候停止，以节省费用），希望能构建一个可以在多个AI主机上快速切换的conda环境，多个主机共享conda和用户环境，可以任意启动多个实例，docker容器停止后用户或应用工作成果不丢失，切换到其他主机后可以继续工作。

* 使用NAS实现多主机文件共享

构建三个基础容器镜像：

1. [cuda](): 安装宿主机驱动，安装和验证cuda docker运行环境

2. [conda](): 基于cuda环境构建公共conda环境

3. [jupyterlab](): 基于conda环境构建一个公共的jupyterlab, ai 试验环境。

> 后续所有试验内容基于本环境进行

## 模型

* llm(NLP)
  * qwen2:[github](https://github.com/QwenLM/Qwen2):

  * chatglm

  * llama3

* video
  * echomimic: [github](https://github.com/BadToBest/EchoMimic):

* audio
  * funasr
  * Whisper: [github](https://github.com/openai/whisper)
  * Coqui XTTS

* cv
  * SAM


## 框架

* pytorch

* jax
* tensorflow

## 训练

* wandb
* deepspeed

## 模型优化

* 微调
* 知识蒸馏

## 运行部署

* accelerate
* ollama: Get up and running with large language models.[github](https://github.com/ollama/ollama)|[ollama模型库](https://ollama.com/library)
* huggingface: The platform where the machine learning community collaborates on models, datasets, and applications. **AI model's github**[官网](https://huggingface.co/)
* vLLM:OpenAI-compatible API server. [github](https://github.com/vllm-project/vllm)|[文档](https://docs.vllm.ai/)

## 应用

* AI-Gateway
* RAG
* langchain: Applications that can reason. Powered by LangChain.[官网](https://www.langchain.com/)|[github](https://github.com/langchain-ai)
* Dify
* 数字人
* 提示词: [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 工具

* 数据标注
  * 


## 待解决问题

* [ ] 模型任务调度平台, 模型或算力是稀有资源，如何在多个应用及用户合理分配资源，同时保证用户体验。



## 其他

* 向量数据库
  * Qdrant
* FastAPI