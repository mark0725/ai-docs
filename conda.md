# conda

## 安装conda

### MAC

```
brew install miniconda
```

### linux



### windows

## 使用

* 帮助
```bash
conda -h              # 获取帮助
conda env -h            # 获取环境相关命令的帮助

```
* 更新
```bash
conda --version           # 查看版本信息
conda update conda          
conda update anaconda      
conda update --all                  # 更新全部包 
conda update xxx            # 更新xxx文件包

```
* 创建环境
```bash
conda env list                      # 显示所有的虚拟环境
conda create --name newname --clone oldname  # 创建一个newname的新环境,里面的包与oldname相同
conda create -n xxxx python=3.9     # 创建python3.9的xxxx虚拟环境,创建新环境的时候最好指定python具体版本,不要创建空环境
conda activate xxxx                 # 激活虚拟环境,可用于不同环境互相切换
conda list                  # 查看当前环境中已经安装的包
conda deactivate          # 退出环境
```

* 卸载包
```bash
conda search package_name     # 可以在安装具体的某款包前查找conda库中是否有对应的版本
conda instal xxx          # 安装xxx文件包
conda uninstall xxx           # 卸载xxx文件包
conda remove -n xxxx --all      # 删除xxxx虚拟环境
# 下面的删除就是清理缓存
conda clean -p              # 删除没有用的包,清理缓存
conda clean -t            # 删除tar安装包
```