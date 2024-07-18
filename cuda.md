# CUDA

## cuda命令

查看cuda版本

```
nvcc --version
```

查看gpu和驱动信息

```
nvidia-smi
```

##

检查pytoch cuda情况

```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_properties("cuda:0"))
print(torch.cuda.get_device_name("cuda:0"))
#测试cuda版本和gpu是否匹配
torch.zeros(1).cuda()
```



```
import cpm_kernels

A = np.random.randn(100, 100)
B = np.random.randn(100, 100)

C = cpm_kernels.matrix_multiplication(A, B)
```

## Ubuntu 22.04.1 LTS上安装cuda和docker pytorch环境

### 更新系统

```
sudo apt update
sudo apt upgrade
```

### 安装docker

```
sudo apt install docker.io docker-compose

docker -v
```

> 设置docker代理

### 禁用默认驱动

* 查看默认驱动

```
lsmod | grep nouveau
```

* 在安装NVIDIA驱动以前需要禁止系统自带显卡驱动nouveau

```
## 在终端输入命令打开blacklist.conf文件。
## gedit/vim/vi 均可
sudo gedit /etc/modprobe.d/blacklist.conf
```

* 在文件末尾增加以下内容，并保存。
```
blacklist nouveau
options nouveau modeset=0
```

* 更新 initramfs，并重启电脑

```
## 更新 initramfs
sudo update-initramfs -u
## 重启
sudo reboot
```

* 检查

```
## 如果没有输出，则说明已禁用nouveau
lsmod | grep nouveau
```



### 安装nvidia显卡驱动

* 先把之前的nvidia驱动卸载干净:

```
sudo apt-get remove --purge nvidia*
```

* 使用lspci 命令查询一下GPU是否存在、型号信息是什么

```
sudo lspci |grep -i nvidia
```

* 在终端里输入下面的命令查看可选择的驱动：

```
sudo ubuntu-drivers devices
```

* 下载你想下载的nvidia驱动版本：

```
sudo apt install nvidia-driver-535
```

* 重启电脑(这一步很重要，不重启没有效果):

```
reboot
```

* 重启后，输入命令查看nvidia驱动是否安装好了，要看到下面的进程。

```
nvidia-smi

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla V100S-PCIE-32GB          Off | 00000000:00:0D.0 Off |                    0 |
| N/A   32C    P0              25W / 250W |      0MiB / 32768MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```

### 安装和配置NVIDIA Container Toolkit

* Configure the production repository:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

* Update the packages list from the repository:

```bash
sudo apt-get update
```

* Install the NVIDIA Container Toolkit packages:

```bash
sudo apt-get install -y nvidia-container-toolkit
```

* Configure the container runtime by using the `nvidia-ctk` command:

```
sudo nvidia-ctk runtime configure --runtime=docker
```

* Restart the Docker daemon:

```
sudo systemctl restart docker
```

* Run a sample CUDA container:

```
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

### 下载容器

* 再docker hub中查找对应版本的docker: https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=12.1

```
docker pull nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
```

* 查看镜像

```
docker images
```

* 启动容器

```
docker run -i -t --gpus all nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04  /bin/bash
```

* 测试镜像

```
nvidia-smi
nvcc -V
```

### 安装pytorch环境

* 宿主机创建目录

```
mkdir -p /work/app
mkdir -p /work/user
mkdir -p /work/data
chmod -R a+rw /work
```

* 启动docker

```
docker run -i -t --gpus all  --name pytorch2 -v "/work:/work" \
nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 /bin/bash 
```

* 初始化user home

```
cp /root/.bashrc /work/user/
cp /root/.profile /work/user/
```

* 重新进入容器

```
docker run -i -t --gpus all  --name pytorch2 \
  -v "/work:/work" \
  -v "/work/user:/root" \
nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 /bin/bash 
```

* 更新docker环境

```
apt-get update
apt-get install -y wget 
rm -rf /var/lib/apt/lists/*
```

* 安装conda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
/bin/bash /tmp/miniconda.sh -b -p /work/app/conda
rm /tmp/miniconda.sh
/work/app/conda/bin/conda clean -tip -y
export PATH=/work/app/conda/bin:$PATH;conda init
```

* 安装python

```
conda create -n python3.10 python=3.10 -y
conda activate python3.10
```

* 安装notebook

```
pip install jupyterlab -i 'https://pypi.tuna.tsinghua.edu.cn/simple'
jupyter lab --generate-config
```

* 修改jupyter_lab_config.py
* 启动 jupyter lab

```
jupyter lab --no-browser --allow-root
```

* 启动容器

```
docker run -d --rm --gpus all  --name pytorch2-1 \
  -p 9999:2300 \
  -v "/work:/work" \
  -v "/work/user:/root" \
nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04 /bin/bash -c 'source /work/app/conda/bin/activate python3.10 && jupyter lab --no-browser --allow-root'
```

> 查看内核列表:jupyter kernelspec list

* 安装pytorch

```
pip3 install torch torchvision torchaudio -i 'https://pypi.tuna.tsinghua.edu.cn/simple'
```

* 验证环境

```
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.device_count())
print(torch.cuda.get_device_properties("cuda:0"))
print(torch.cuda.get_device_name("cuda:0"))
```



### 制作自己的pytorch容器

* Dockerfile:`Dockerfile.pytorch`

```dockerfile
# 使用官方Ubuntu基础镜像
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04
 
# 安装依赖
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

# 下载Miniconda安装脚本
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh

# 安装Miniconda
RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh && \
    /opt/conda/bin/conda clean -tip -y

# 将Conda添加到PATH，以便conda命令可以直接使用
ENV PATH /opt/conda/bin:$PATH

# 使用国内源
COPY condarc  /root/.condarc
RUN /opt/conda/bin/conda config --set show_channel_urls yes && conda clean -iy

# 创建python环境
RUN /opt/conda/bin/conda create -n python3.10 python=3.10 -y
RUN /bin/bash -c 'conda init'

ENV PATH /opt/conda/envs/python3.10/bin:$PATH

# 安装pytorch
#RUN /bin/bash -c 'source /opt/conda/bin/activate python3.10 && conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y'
RUN /bin/bash -c 'source /opt/conda/bin/activate python3.10 && pip3 install torch torchvision torchaudio -i https://pypi.tuna.tsinghua.edu.cn/simple'
#RUN /bin/bash -c 'source /opt/conda/bin/activate python3.10 && pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 -i https://download.pytorch.org/whl/cu121'
    
```



* 国内conda源配置

```yaml
channels:
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
 - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
 - defaults
show_channel_urls: true
```

> pip 国内镜像源
>
> https://pypi.tuna.tsinghua.edu.cn/simple
> https://mirrors.aliyun.com/pypi/simple/
> https://pypi.mirrors.ustc.edu.cn/simple/

* 构建docker 镜像

```bash
docker build -f Dockerfile.pytorch -t pytorch2.2-cuda12.1-cudnn8-ubuntu20.04 .
```

* 启动容器

```
```

