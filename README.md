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

* Aliyun
* Huaweicloud： [Modelarts](https://console.huaweicloud.com/modelarts/?region=cn-north-4#/dashboard)提供免费GPU学习环境，但是gpu比较老，适合简单基础学习。长时间运行模型最好购买GPU服务器。
* AWS
* Azure
* AutoDL: [官网](https://www.autodl.com/home)
* 算力互联: [官网](https://www.casdao.com/)
* colab
* PC

## 基础

* python:
* numpy:
* jupyterlab: JupyterLab is the latest web-based interactive development environment for notebooks, code, and data. [官网](https://jupyter.org/)
* opencv2:

## 模型

### 开源模型

* llm(NLP)
  * qwen2: [github](https://github.com/QwenLM/Qwen2)
  * GLM-4: [github](https://github.com/THUDM/GLM-4)
  * llama3: [github](https://github.com/meta-llama/llama3)
  * code-llama:  [github](https://github.com/meta-llama/codellama)|[提示词](https://www.promptingguide.ai/models/code-llama)
  * clip: Predict the most relevant text snippet given an image.[github](https://github.com/openai/CLIP)
* avatar-(数字人)
  * echomimic: [github](https://github.com/BadToBest/EchoMimic)|[官网](https://badtobest.github.io/echomimic.html)|[笔记]()
  * SadTalker: [github](https://sadtalker.github.io)
  * Wav2Lip: [github](https://github.com/Rudrabha/Wav2Lip)
  * ER-NeRF: [github](https://github.com/Fictionarry/ER-NeRF)
* video
  * 

* audio
  * funasr: 阿里的`FunASR`的语音识别效果也是相当不错，而且时间也是比whisper更快的.[github](https://github.com/modelscope/FunASR)
  * Whisper: Whisper is a general-purpose speech recognition model. [github](https://github.com/openai/whisper)
  * GPT-SoVITS: 低成本AI音色克隆软件。目前只有TTS（文字转语音）功能，将来会更新变声功能[github](https://github.com/RVC-Boss/GPT-SoVITS)|[文档](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e)
  * Coqui XTTS: Coqui XTTS是一个领先的深度学习文本到语音任务（TTS语音生成模型）工具包，通过使用一段5秒钟以上的语音频剪辑就可以完成声音克隆*将语音克隆到不同的语言*。 [github](https://github.com/coqui-ai/TTS)|[huggingface](https://huggingface.co/spaces/coqui/xtts)
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

* PASCAL VOC: Visual Object Classes [官网](http://host.robots.ox.ac.uk/pascal/VOC/)

## 运行部署

* accelerate: A simple way to launch, train, and use PyTorch models on almost any device and distributed configuration[github](https://github.com/huggingface/accelerate)
* ollama: Get up and running with large language models.[github](https://github.com/ollama/ollama)|[官网](https://ollama.com/)|[docker](https://hub.docker.com/r/ollama/ollama)|[ollama模型库](https://ollama.com/library)|[笔记](./ollama.md)
* huggingface: The platform where the machine learning community collaborates on models, datasets, and applications. **AI model's github**[官网](https://huggingface.co/)
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