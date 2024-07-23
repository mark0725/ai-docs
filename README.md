# AI学习笔记

## 环境

由于需要在多个按量计费的云环境中动态使用GPU服务器（不用的时候停止，以节省费用），希望能构建一个可以在多个AI服务器上快速切换的conda环境，多个服务器共享conda环境和用户工作内容，可以任意启动多个实例，docker容器停止后用户或应用工作成果不丢失，切换到其他主机后可以继续工作。

* NAS服务：使用NAS实现多主机文件共享. [NFS挂载](./infra/nas.md)

* GPU服务器: 阿里云V100GPU服务器环境安装.[Ubuntu 22.04.1 LTS上安装cuda环境](./cuda.md)

* docker环境: 安装GPU Docker容器环境.[NVIDIA Docker工具包安装官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)|[验证环境官方文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html)|[安装笔记](./cuda.md)

* 构建三个基础容器镜像：
  1. [cuda](): 安装和验证cuda docker运行环境
  2. [conda](): 基于容器cuda环境构建公共conda环境
  3. [jupyterlab](): 基于conda环境构建一个公共的jupyterlab, ai 试验环境。


> 后续所有试验内容基于本环境进行

## 算力

**运行AI模型最好能够使用GPU算力，稍微大一点的模型CPU基本跑不动。**

* Aliyun
* Huaweicloud： [Modelarts](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard)提供免费GPU学习环境，但是gpu比较老，适合简单基础学习。长时间运行模型最好购买GPU服务器。
* AWS
* Azure
* AutoDL: 价格便宜[官网](https://www.autodl.com/home)
* 算力互联: 价格便宜，卡多[官网](https://www.casdao.com/)
* 智星云: 价格便宜[官网](https://www.ai-galaxy.cn/home)
* ucloud:  [官网](https://www.ucloud.cn/site/active/new/gpu.html)
* colab：google，提供部分免费学习算力。
* PC: windows,mac,linux

## 基础

* python:
* numpy:
* jupyterlab: JupyterLab is the latest web-based interactive development environment for notebooks, code, and data. [官网](https://jupyter.org/)
* opencv2:

## 模型

### 开源模型

* llm(NLP)
  * Linly-AI: [github](https://github.com/CVI-SZU/Linly)
  * qwen: [github](https://github.com/QwenLM/Qwen2)
  * GLM-4: [github](https://github.com/THUDM/GLM-4)
  * llama3: [github](https://github.com/meta-llama/llama3)
  * code-llama:  [github](https://github.com/meta-llama/codellama)|[提示词](https://www.promptingguide.ai/models/code-llama)
  * clip: Predict the most relevant text snippet given an image.[github](https://github.com/openai/CLIP)
* avatar-(数字人)
  * echomimic: [github](https://github.com/BadToBest/EchoMimic)|[官网](https://badtobest.github.io/echomimic.html)|[笔记]()
  * SadTalker: [github](https://sadtalker.github.io)
  * Wav2Lip: [github](https://github.com/Rudrabha/Wav2Lip)
  * ER-NeRF: 是使用最新的NeRF技术构建的数字人，拥有定制数字人的特性，只需要一个人的五分钟左右到视频即可重建出来.[github](https://github.com/Fictionarry/ER-NeRF)
* 多模态
  * Stable Diffusion

  * Kolors: 快手可图,文生图大模型.[github](https://github.com/Kwai-Kolors/Kolors)|[modelscope](https://www.modelscope.cn/models/Kwai-Kolors/Kolors)
* video
  * 
* audio
  * funasr: 阿里的`FunASR`的语音识别效果也是相当不错，而且时间也是比whisper更快的.[github](https://github.com/modelscope/FunASR)
  * Whisper: Whisper is a general-purpose speech recognition model. [github](https://github.com/openai/whisper)
  * GPT-SoVITS: 低成本AI音色克隆软件。目前只有TTS（文字转语音）功能，将来会更新变声功能[github](https://github.com/RVC-Boss/GPT-SoVITS)|[文档](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
  * Coqui XTTS: Coqui XTTS是一个领先的深度学习文本到语音任务（TTS语音生成模型）工具包，通过使用一段5秒钟以上的语音频剪辑就可以完成声音克隆*将语音克隆到不同的语言*。 [github](https://github.com/coqui-ai/TTS)|[huggingface](https://huggingface.co/spaces/coqui/xtts)
  * SenseVoice:  体验有情感识别、声音事件检测、语音识别等功能的音频理解模型. [github](https://github.com/FunAudioLLM/SenseVoice)|[modelscope](https://www.modelscope.cn/studios/iic/SenseVoice)
  * CosyVoice: TTS [github](https://github.com/FunAudioLLM/CosyVoice)|[modelscope](https://www.modelscope.cn/models/speech_tts/CosyVoice-300M/summary)|[demo](https://fun-audio-llm.github.io/)
* cv
  * SAM: [github](https://github.com/facebookresearch/segment-anything)
* ocr
  * PaddleOCR: [github](https://github.com/PaddlePaddle/PaddleOCR)
  * tesseract-ocr: [github](https://github.com/tesseract-ocr/tesseract)

### 试验

* transformer
* cnn
* rnn
* lstm
* gan
* seq2seq
* vae
* unet

### 框架

* pytorch: Tensors and Dynamic neural networks in Python with strong GPU acceleration. [官网](https://pytorch.org/)|[github](https://github.com/pytorch/pytorch)
* jax: Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/TPU, and more[官网](http://jax.readthedocs.io/)|[github](https://github.com/google/jax)
* tensorflow: [官网]()
* candle：Minimalist ML framework for Rust [github](https://github.com/huggingface/candle)

### 训练工具

* wandb: The AI developer platform,Train and fine-tune models, manage models from experimentation to production, and track and evaluate LLM applications.[官网](https://wandb.ai/)
* DeepSpeed: DeepSpeed is a deep learning optimization library that makes distributed training and inference easy, efficient, and effective.[github](https://github.com/microsoft/DeepSpeed)|[官网](https://www.deepspeed.ai/)

### 模型训练

* 预训练
* 微调
* 知识蒸馏

### 数据集

* 图像分类
  - **MNIST**: 包含70,000张手写数字图像（60,000张训练集和10,000张测试集）。每张图片是28x28的灰度图像，标签为0-9的数字。
  - **CIFAR-10/100**: CIFAR-10包含60,000张32x32的彩色图像，分为10类；CIFAR-100有100个类别，每类600张图片。
  - **ImageNet**: 一个大型图像数据库，拥有超过1400万张图像，包含2万多个类。常用于大型图像分类任务和预训练模型的转移学习。
* 目标检测
  - **PASCAL VOC**: 提供图像和物体识别标记（bounding boxes），涵盖20类物体。常用于目标检测和分割任务。 [官网](http://host.robots.ox.ac.uk/pascal/VOC/)
  - **COCO (Common Objects in Context)**: 包含320,000张图像和超过200万个标注（annotations），涵盖80个类别。丰富的实例分割和对象检测标注使其成为许多检测任务的标准数据集。
* 语义/实例分割
  - **Cityscapes**: 包含来自50个不同城市的驾驶场景图像，专注于语义分割任务，特别是自动驾驶的应用。
  - **ADE20K**: 包含20,000张用于训练和2,000张用于验证的图像，覆盖150个语义类别，用于场景解析。
* 文本处理
  - **IMDB**: 包含50,000条电影评论用于情感分析任务，分为正面和负面两类。
  - **20 Newsgroups**: 包含约20,000篇新闻文章，分为20个新闻组，用于文本分类。
  - **SQuAD (Stanford Question Answering Dataset)**: 包含超过100,000个问答对，主要用于阅读理解和问答系统。
* 语音识别
  - **LibriSpeech**: 一个大规模的语音数据集，包含约1000小时的英语读书音频，主要用于语音识别任务。
  - **TIMIT**: 提供时间标注的话语和标注，为不同的语音识别任务提供参考。
* 时间序列预测
  - **UCI Machine Learning Repository**: 提供各种各样的时间序列数据集，如空气质量、股票价格等。
* 推荐系统
  - **MovieLens**: 包含数百万条电影评分数据，可以用于推荐系统任务。

## 运行部署

* accelerate: A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration[github](https://github.com/huggingface/accelerate)
* ollama: Get up and running with large language models.[github](https://github.com/ollama/ollama)|[官网](https://ollama.com/)|[docker](https://hub.docker.com/r/ollama/ollama)|[ollama模型库](https://ollama.com/library)|[笔记](./ollama.md)
* huggingface: The platform where the machine learning community collaborates on models, datasets, and applications. **AI model's github**[官网](https://huggingface.co/)
* 魔塔： 类似huggingface, [官网](www.modelscope.cn)
* vLLM:OpenAI-compatible API server. [github](https://github.com/vllm-project/vllm)|[文档](https://docs.vllm.ai/)
* Triton: [文档](https://www.nvidia.cn/gpu-cloud/ngc-nvidia-triton/)

## 应用

* RAG
* langchain: Applications that can reason. Powered by LangChain.[官网](https://www.langchain.com/)|[github](https://github.com/langchain-ai)
* Dify: [github](https://github.com/langgenius/dify)|[官网](https://dify.ai/)
* ComfyUI
* 数字人
* 提示词: [Prompt Engineering Guide](https://www.promptingguide.ai/)
* AI-Gateway
* LlamaIndex: LlamaIndex is a data framework for your LLM applications. [github](https://github.com/run-llama/llama_index)

## 工具

* 数据标注
  * jTessBoxEditor:  [jTessBoxEditor](https://vietocr.sourceforge.net/training.html)is a box editor and trainer for [Tesseract OCR](https://github.com/tesseract-ocr)
  * labelme: [github](https://github.com/labelmeai/labelme)
  * AnyLabeling: 是一款自动标注工具
  * 精灵标注: 图片标注[官网](http://www.jinglingbiaozhu.com/)
  * CVAT:  is an interactive video and image annotation tool for computer vision.[官网](https://app.cvat.ai/)
  * makesense： Free to use online tool for labelling photos. [github](https://github.com/SkalskiP/make-sense)|[官网](https://www.makesense.ai/)
* WebUI
  * gradio: Gradio是一个Python库,提供了一种简单的方式将机器学习模型作为交互式Web应用程序来部署。
  * Open WebUI: User-friendly WebUI for LLMs [github](https://github.com/open-webui/open-webui)

## 待解决问题

* [ ] 模型任务调度平台, 模型或算力是稀有资源，如何给多个应用及用户合理分配资源，同时保证用户体验。关键指标：分区隔离、限流、降级、动态平衡、多任务、集群、可观察、多模型供应商。
* [ ] 多个云的算力资源混合管理和调度。
* [ ] 多个云之间数据资源统一管理、共享、同步。

## 其他

* 向量数据库
  * faiss: [github](https://github.com/facebookresearch/faiss)
  * Qdrant: [github](https://github.com/qdrant/qdrant)
  * Chroma: [github](https://github.com/chroma-core/chroma)
* FastAPI
* nvtop: GPU & Accelerator process monitoring for AMD, Apple, Huawei, Intel, NVIDIA and Qualcomm. [github](https://github.com/Syllo/nvtop?tab=readme-ov-file)
* nvitop: An interactive NVIDIA-GPU process viewer and beyond, the one-stop solution for GPU process management.[github](https://github.com/XuehaiPan/nvitop)|[doc](https://github.com/XuehaiPan/nvitop)
* Tokenizers: Fast State-of-the-Art Tokenizers optimized for Research and Production.[github](https://github.com/huggingface/tokenizers)|[文档](https://huggingface.co/docs/tokenizers)
* ffmpeg
* SRS:SRS is a simple, high-efficiency, real-time video server supporting RTMP, WebRTC, HLS, HTTP-FLV, SRT, MPEG-DASH, and GB28181.[github](https://github.com/ossrs/srs)
* NAS: 实现多服务器文件共享，可以使用云厂商NAS服务或自己搭建NFS Server[阿里云搭建NAS笔记](./nas.md)