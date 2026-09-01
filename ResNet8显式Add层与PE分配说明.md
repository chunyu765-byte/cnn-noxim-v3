# ResNet8 显式 Add 层与 PE 分配说明

本文记录当前 `cnn-noxim-v2` 中 ResNet8 残差结构的实现方式，方便后续理解、调试和继续修改近似通信逻辑。

## 1. 当前总体方案

当前实现采用 **ResNet8 专用显式 Add 层**。

也就是说，残差加法不是隐式写在卷积层内部，也不是复用主分支 PE 完成，而是作为一个新的网络层写入 noxim 模型文件：

```text
Add H W C src0 src1 act out_scale out_zp
```

其中：

- `H W C`：Add 输出特征图尺寸。
- `src0 src1`：两个输入来源层号。
- `act`：Add 后激活函数，ResNet8 中使用 `relu`。
- `out_scale out_zp`：Add 输出量化参数。

当前 Add 层会被当作独立层参与：

- 神经元分组。
- PE 分配。
- 数据接收。
- 加法计算。
- ReLU。
- 输出发送到下一层。

## 2. 1x1 Shortcut 分支是否单独分配 PE

答案：**是，1x1 卷积分支单独分配 PE 计算。**

在 ResNet8 的 block2 和 block3 中，由于 stride 或通道数变化，shortcut 不是 identity，而是 projection shortcut：

```text
shortcut(x) = 1x1 Conv(x)
```

当前实现中，这两个 1x1 shortcut 都被写成普通 `Convolution` 层，因此会独立分配 PE。

它们不是：

- 在主分支 Conv PE 上计算。
- 在 Add PE 上计算。
- 在前一层 PE 上原地计算。

而是作为独立卷积层执行。

## 3. 当前 ResNet8 层结构

当前导出的 ResNet8 noxim 层结构如下：

```text
0  Input
1  conv1

2  block1.conv1
3  block1.conv2
4  Add1: src0=3, src1=1

5  block2.conv1
6  block2.conv2
7  block2.shortcut.0  1x1 Conv, src_layer=4
8  Add2: src0=6, src1=7

9   block3.conv1
10  block3.conv2
11  block3.shortcut.0  1x1 Conv, src_layer=8
12  Add3: src0=10, src1=11

13  AvgPool
14  FC
```

注意：

- layer4 是第一个显式 Add 层。
- layer8 是第二个显式 Add 层。
- layer12 是第三个显式 Add 层。
- layer7 和 layer11 是 projection shortcut 的 1x1 Conv。

## 4. 主分支和 Shortcut 分支的数据流

### block1

block1 是 identity shortcut，没有 1x1 Conv。

```text
layer1 conv1  ----------------\
                               -> layer4 Add1 -> ReLU
layer2 block1.conv1 -> layer3 --/
```

其中：

- 主分支：`layer3 -> layer4`
- shortcut 分支：`layer1 -> layer4`

### block2

block2 的 shortcut 是 1x1 Conv。

```text
layer4 Add1 -> layer5 block2.conv1 -> layer6 block2.conv2 --\
                                                              -> layer8 Add2 -> ReLU
layer4 Add1 -> layer7 block2.shortcut.0 1x1 Conv ------------/
```

其中：

- 主分支：`layer6 -> layer8`
- shortcut 分支：`layer7 -> layer8`
- shortcut 1x1 Conv 输入来源：`layer4`

### block3

block3 的 shortcut 也是 1x1 Conv。

```text
layer8 Add2 -> layer9 block3.conv1 -> layer10 block3.conv2 --\
                                                               -> layer12 Add3 -> ReLU
layer8 Add2 -> layer11 block3.shortcut.0 1x1 Conv ------------/
```

其中：

- 主分支：`layer10 -> layer12`
- shortcut 分支：`layer11 -> layer12`
- shortcut 1x1 Conv 输入来源：`layer8`

## 5. 当前验证时的 PE 分配

验证命令使用：

```text
-dimx 8 -dimy 8 -dimz 1 -groupsize 4096 -mapping dir_x
```

对应日志：

```text
bin/run_log/run_log_26_08_31_resnet8_cifar10_exact_pre2_pe2_cpu2-3_8x8x1_gs4096_sim4e6.txt
```

当前 floorplan 中各层 PE 分配如下：

```text
layer1  conv1                 PE0  - PE3
layer2  block1.conv1          PE4  - PE7
layer3  block1.conv2          PE8  - PE11
layer4  Add1                  PE12 - PE15

layer5  block2.conv1          PE16 - PE17
layer6  block2.conv2          PE18 - PE19
layer7  block2.shortcut.0     PE20 - PE21
layer8  Add2                  PE22 - PE23

layer9   block3.conv1         PE24
layer10  block3.conv2         PE25
layer11  block3.shortcut.0    PE26
layer12  Add3                 PE27

layer13  AvgPool              PE28
layer14  FC                   PE29
```

因此可以明确看到：

- Add 层使用独立 PE。
- 1x1 shortcut Conv 使用独立 PE。
- 主分支 Conv 和 shortcut Conv 不共享 PE。
- Add PE 接收来自主分支和 shortcut 分支的输入后再执行加法。

## 6. 为什么需要显式 source layer

原始 AlexNet 是线性结构，因此每层默认从前一层接收输入即可：

```text
layer N receives from layer N-1
```

ResNet 不满足这个假设。

例如 layer7 是 block2 的 shortcut 1x1 Conv，它在模型文件顺序上位于 layer6 后面，但它真正的输入不是 layer6，而是 layer4：

```text
layer7 source = layer4
```

因此当前 `Convolution` 行额外支持一个可选字段：

```text
Convolution ... out_scale out_zp [src_layer]
```

如果没有写 `src_layer`，仍然默认使用前一层，兼容 AlexNet。

ResNet8 中关键 shortcut Conv 写法是：

```text
layer7  block2.shortcut.0: src_layer=4
layer11 block3.shortcut.0: src_layer=8
```

## 7. 对近似通信设计的影响

当前实现非常适合后续按边区分近似通信，特别是残差结构中的以下边：

```text
main branch:
layer3  -> layer4
layer6  -> layer8
layer10 -> layer12

shortcut branch:
layer1  -> layer4
layer7  -> layer8
layer11 -> layer12
```

如果后续要分别控制主分支和 shortcut 分支的近似强度，不要只按 Add 层配置一个阈值，而是使用 edge 级配置。

当前已经支持两类边级近似配置。FAS 使用 `Edge`，SAP-RLE/SAP-RLE-V2 使用 `SapEdge`，不要混用。

FAS 边级配置文件格式为：

```text
Edge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5
```

含义：

- `src_layer dst_layer`：模型文件中的源层号和目标层号。
- `config_sel`：这条边独立选择的配置列，范围是 `0..5`。
- `th0 th1 th2 th3`：这条边自己的 4 个 FAS 阈值。
- `lv0..lv5`：这条边在 6 个配置列下允许的最大近似等级，范围是 `-1..3`，其中 `-1` 表示这条边不做 FAS 压缩。

命令行使用：

```text
-NNfas_edge_approx resnet8_cifar10/resnet8_fas_edge_approx.txt
```

`-NNedge_approx` 仍然保留为兼容旧命令的别名，但后续建议写 `-NNfas_edge_approx`。

SAP-RLE/SAP-RLE-V2 边级配置文件格式为：

```text
SapEdge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5
```

含义：

- `src_layer dst_layer`：模型文件中的源层号和目标层号。
- `config_sel`：这条边独立选择的配置列，范围是 `0..5`。
- `th0 th1 th2 th3`：这条边自己的 4 个 SAP delta 候选值。
- `lv0..lv5`：这条边在 6 个配置列下选择哪个 delta，范围是 `-1..3`，其中 `-1` 表示这条边关闭 SAP 压缩。

命令行使用：

```text
-sap_edge_approx resnet8_cifar10/resnet8_sap_edge_approx.txt
```

如果不传 `-sap_edge_approx`，SAP 继续使用按层配置 `-sap_threshold_file` 和 `-sap_level_table_file`。

当前 ResNet8 导出器会生成：

```text
FAS:
resnet8_cifar10/resnet8_fas_threshold.txt
resnet8_cifar10/resnet8_fas_level_table.txt
resnet8_cifar10/resnet8_fas_edge_approx.txt

SAP:
resnet8_cifar10/resnet8_sap_threshold.txt
resnet8_cifar10/resnet8_sap_level_table.txt
resnet8_cifar10/resnet8_sap_edge_approx.txt
```

为了兼容旧脚本，导出器仍然生成这些别名：

```text
approx.txt / resnet8_approx.txt -> FAS threshold
approx_level_table.txt / resnet8_approx_level_table.txt -> FAS level table
edge_approx.txt / resnet8_edge_approx.txt -> FAS edge table
sap_threshold.txt -> SAP threshold
sap_level_table.txt -> SAP level table
sap_edge_approx.txt -> SAP edge table
```

默认 FAS/SAP 边表都列出全部真实通信边：

```text
Edge 1 2 0 0 0 0 0 0 1 2 3 3 3
Edge 2 3 0 0 0 0 0 0 1 2 3 3 3
Edge 3 4 0 0 0 0 0 0 1 2 3 3 3
Edge 1 4 0 0 0 0 0 0 1 2 3 3 3
Edge 4 5 0 0 0 0 0 0 1 2 3 3 3
Edge 5 6 0 0 0 0 0 0 1 2 3 3 3
Edge 4 7 0 0 0 0 0 0 1 2 3 3 3
Edge 6 8 0 0 0 0 0 0 1 2 3 3 3
Edge 7 8 0 0 0 0 0 0 1 2 3 3 3
Edge 8 9 0 0 0 0 0 0 1 2 3 3 3
Edge 9 10 0 0 0 0 0 0 1 2 3 3 3
Edge 8 11 0 0 0 0 0 0 1 2 3 3 3
Edge 10 12 0 0 0 0 0 0 1 2 3 3 3
Edge 11 12 0 0 0 0 0 0 1 2 3 3 3
Edge 12 13 0 0 0 0 0 0 1 2 3 3 3
Edge 13 14 0 0 0 0 0 0 1 2 3 3 3
```

SAP 文件中同样的边使用 `SapEdge` 开头。

这样后续论文实验可以清楚比较：

- 只近似主分支。
- 只近似 shortcut 分支。
- 主分支和 shortcut 分支使用不同阈值。
- Add 后输出再近似。

## 8. 当前实现边界

当前版本已经验证：

- AlexNet 原线性路径未被破坏。
- ResNet8 显式 Add 精确仿真可以跑完。
- projection shortcut 1x1 Conv 可以从非前一层 source 正确取输入。
- FAS 可以按边独立选择阈值和配置列。
- SAP-RLE/SAP-RLE-V2 可以按边覆盖按层 delta。

当前版本暂未完整实现：

- main/shortcut 分支独立近似策略。
- Add 层近似计算能耗模型。
- shortcut Conv 和主分支 Conv 的 PE 复用优化。

这些可以作为下一阶段工作。

## 9. 后续修改建议

短期建议优先做 edge 级近似通信表，只覆盖残差 Add 输入边。

原因：

- 改动范围小。
- 实验主题明确。
- 可以快速比较 main branch 与 shortcut branch 的误差敏感性。
- 不需要马上重构整个 NNModel 为完全通用动态图。

如果后续想进一步减少通信开销，可以再考虑 PE 复用方案，例如：

- 让 Add 层复用主分支最后一层 Conv 的 PE。
- 让 identity shortcut 数据直接进入主分支 PE 做本地加法。
- 让 projection shortcut 1x1 Conv 和 Add 做局部融合。

但这些都会改变当前清晰的通信边界，不建议作为第一周内的主线。
