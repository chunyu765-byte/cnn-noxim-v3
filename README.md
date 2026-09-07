# cnn-noxim-v3 快速说明

本仓库是 CNN-NoC/Noxim 仿真实验工程。更完整的新电脑安装步骤见 `新电脑安装与运行.md`，常用运行命令见 `常用命令.md`。

## 新电脑环境

推荐系统：Ubuntu 24.04 LTS。

优先使用系统包安装 SystemC：

```bash
sudo apt update
sudo apt install -y build-essential git make wget tar pkg-config libsystemc-dev python3 python3-venv python3-pip
```

如果工位电脑不方便使用 `sudo`，也可以把新版 SystemC 安装到用户目录，例如：

```text
/home/neil-pc/.local/systemc-3.0.2
```

当前 `bin/Makefile.defs` 会优先使用 `pkg-config systemc`；如果没有系统包，会自动检查 `$(HOME)/.local/systemc-3.0.2`，最后再回退到旧路径 `/usr/local/systemc-2.3.1`。因此这台新电脑上直接执行 `make` 即可。

## 编译

```bash
cd ~/cnn-noxim-v3/bin
make -j"$(nproc)"
```

编译成功后生成：

```text
bin/build/noxim
```

## 多线程参数

当前 Noxim 支持两个线程参数：

```text
-precompute_threads N   坐标/通信缓存预计算线程数
-pe_compute_threads N   精确 PE 计算线程数
```

线程上限会按当前机器在线 CPU 数自动判断。例如这台 20 核心线程机器上，`./build/noxim -help` 会显示 `1..20`。

## ResNet8 精确仿真示例

```bash
cd ~/cnn-noxim-v3/bin
./build/noxim \
  -dimx 8 -dimy 8 -dimz 1 \
  -NNmodel resnet8_cifar10/resnet8_model.txt \
  -NNweight resnet8_cifar10/resnet8_weight_fc_wb2parser.txt \
  -NNweight_scale resnet8_cifar10/resnet8_weight_scale.txt \
  -NNinput resnet8_cifar10/resnet8_input.txt \
  -NNapprox resnet8_cifar10/resnet8_approx.txt \
  -NNapprox_Level_Table resnet8_cifar10/resnet8_approx_level_table.txt \
  -NNlabel resnet8_cifar10/resnet8_label.txt \
  -mapping dir_x \
  -groupsize 4096 \
  -pe_log 0 \
  -sim 4000000 \
  -stop_on_infer_done 1 \
  -precompute_threads 20 \
  -pe_compute_threads 20 \
  -thermal_update 0 \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 \
  -acdc_abdtr 0 -is_sap_rle 0 -is_sap_rle_v2 0
```

## VGG11 精确仿真示例

```bash
cd ~/cnn-noxim-v3/bin
./build/noxim \
  -dimx 8 -dimy 8 -dimz 1 \
  -NNmodel vgg11_cifar10/vgg11_model.txt \
  -NNweight vgg11_cifar10/vgg11_weight_fc_wb2parser.txt \
  -NNweight_scale vgg11_cifar10/vgg11_weight_scale.txt \
  -NNinput vgg11_cifar10/vgg11_input.txt \
  -NNapprox vgg11_cifar10/vgg11_approx.txt \
  -NNapprox_Level_Table vgg11_cifar10/vgg11_approx_level_table.txt \
  -NNlabel vgg11_cifar10/vgg11_label.txt \
  -mapping dir_x \
  -groupsize 4096 \
  -pe_log 0 \
  -sim 4000000 \
  -stop_on_infer_done 1 \
  -precompute_threads 20 \
  -pe_compute_threads 20 \
  -thermal_update 0 \
  -isapprox 0 -allzeropacket 0 -zero_skip 0 \
  -acdc_abdtr 0 -is_sap_rle 0 -is_sap_rle_v2 0
```

## VSCode 头文件路径

如果 VSCode 不能识别 `systemc.h`，在 C/C++ 配置里加入当前 SystemC include 路径之一：

```text
/home/neil-pc/.local/systemc-3.0.2/include
/usr/local/systemc-2.3.1/include
```

---

# 常用：
./build/noxim -dimx 8 -dimy 8 -dimz 1   -NNmodel lenet5/model.txt   -NNweight lenet5/weight.txt   -NNweight_scale lenet5/weight_scale.txt   -NNinput lenet5/input1_new.txt   -NNapprox lenet5/approx.txt   -NNlabel lenet5/label1.txt   -mapping dir_x   -groupsize 1024   -sim 20000 > run_log/run_log_26_04_14_2.txt
# 使用方法：
bin目录下打开终端
./build/noxim
-dimx 8 指定NoC的x尺寸
-dimy 8 指定NoC的y尺寸
-dimz 1 指定NoC的z尺寸
-NNmodel model.txt 指定模型结构文件
-NNinput input.txt 指定输入文件
-NNweight weight.txt 指定权重文件
-NNweight_scale weight_scale.txt 每通道的权重缩放因子
-NNapprox approx.txt 指定近似阈值文件
-NNapprox_Level_Tabel approx_level_table.txt 指定阈值配置表
-NNlabel label.txt 指定标签列表
-mapping random 指定映射算法
-approx_compute 0 近似计算模式
-isapprox 0 不启用近似通信
-groupsize 1024 1024个神经元（乘累加）一组
-tdm 1 时分复用次数
-sim 10000 仿真10000周期


#vscode配置 添加systemc头文件路径
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/local/systemc-2.3.1/include"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++17",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}


#关于输入文件格式
## model.txt
### Input层
Input 宽度 高度 通道数 输入缩放（浮点数）
### Convolution层
Convolution 输出宽度 输出高度 输出通道数 卷积核宽度 卷积核高度 卷积核通道数（输入通道数） 卷积步长 0填充大小 激活函数 权重缩放 近似阈值（等级） 不知道什么玩意 输入缩放（浮点数,等于上一层的输出缩放） 输入零点位置 输出缩放（浮点数,等于权重缩放） 输出零点位置（默认0）
### Pooling层
Pooling 输出宽度 输出高度 输出通道数 池化宽度 池化高度 池化步长 池化函数 近似阈值（等级）
### Dense层
Dense 输出大小 激活函数 权重缩放 近似阈值（等级） 输出缩放 输入缩放（浮点数，等于上一层的输入缩放） 输入零点位置 输出缩放（浮点数，等于权重缩放） 输出零点位置（默认0）
#### model.txt 中的 weight_scale 参数可能没有被使用
## input.txt
宽度、高度、通道数的顺序存放量化后的输入数据
## weight.txt
### 卷积层
按卷积核宽度、高度、输入通道数、存放一个输出通道的巻积核权重数据，再存放一个偏置数据，在按顺序存放剩余通道的，均是量化后的权重和偏执
### 全连接层
按输入通道的顺序存放所有权重数据外加一个偏置，再按输出通道顺序存放其他的，均为量化后的权重和偏置

### weight_scale.txt
存放每层每个通道的权重缩放（浮点数）



noxim中是通过把4个近似阈值信息放在数据包中实现数据恢复的，并不是本地存储了上一层的近似阈值
具体见 **NoximMain.h**的struct NoximPacket

## 必须要有result及其内部的文件夹目录
