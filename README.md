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
