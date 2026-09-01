# ResNet8 近似通信 Python 搜索规则

## 1. 目标

在 CIFAR10 完整测试集上，为 ResNet8 搜索 FAS、SAP-RLE、SAP-RLE-V2 和 ABDTR 的近似通信配置。

精度约束定义为：

```text
Acc_approx >= Acc_exact - 2%
```

搜索目标是在满足 2% 精度下降约束的前提下，尽量提高近似强度，减少通信量。

## 2. 基本边界

Python 仿真只模拟激活值的 16bit 量化和近似通信误差传播，用于得到每层/每边的近似等级选择。

noxim 中还包含权重量化、PE 计算、NoC 发包、路由、缓存和能耗统计，因此 Python 中统计出的阈值数值不直接等同于 noxim 最终阈值。Python 结果主要用于指导：

- 每层候选阈值等级。
- 每条边使用哪个配置等级。
- 哪些层/边应该保守，哪些层/边可以激进。

## 3. 通用数据集

精度评估使用完整 CIFAR10 测试集。

搜索过程中每次配置变更都要记录：

- 方案名称。
- 当前全局等级或逐层配置。
- 每层阈值。
- 每层/每边等级选择。
- CIFAR10 测试集分类精度。
- 是否满足 2% 精度约束。
- 与上一轮相比是否回退。

## 4. Shortcut 规则

shortcut 相关边默认不做近似通信。

ResNet8 当前显式 Add 图中的 shortcut 边包括：

```text
1 -> 4
4 -> 7
7 -> 8
8 -> 11
11 -> 12
```

说明：

- `1 -> 4`、`7 -> 8`、`11 -> 12` 是 shortcut 分支进入 Add 的边。
- `4 -> 7`、`8 -> 11` 是 projection shortcut 1x1 Conv 的输入边。
- 这些边在 FAS/SAP-RLE/SAP-RLE-V2 中默认配置为关闭近似。

## 5. FAS 阈值统计

FAS 对每一层单独统计阈值。

统计对象是该层 16bit 量化激活值的绝对值：

```text
abs(activation_q16)
```

统计步骤：

1. 收集该层输出激活值。
2. 转成 16bit 量化整数。
3. 取绝对值。
4. 去掉 0。
5. 升序排序。
6. 取 20%、40%、60%、80% 四个位置作为候选阈值：

```text
th0 = sorted_abs_values[20%]
th1 = sorted_abs_values[40%]
th2 = sorted_abs_values[60%]
th3 = sorted_abs_values[80%]
```

对应 noxim 文件中的：

```text
th0 th1 th2 th3
```

## 6. SAP-RLE 阈值统计

SAP-RLE 和 SAP-RLE-V2 对每一层单独统计阈值。

统计对象是同一通道内连续激活值的差分绝对值，不跨通道：

```text
abs(activation_q16[i] - activation_q16[i-1])
```

注意：SAP-RLE 阈值统计时不按 6 个数分组。6 个数一组只用于后续模拟近似通信压缩/恢复时。

统计步骤：

1. 收集该层输出激活值。
2. 转成 16bit 量化整数。
3. 按通道展开。
4. 只在同一通道内计算连续相邻差分。
5. 对差分取绝对值。
6. 去掉 0。
7. 升序排序。
8. 取 20%、40%、60%、80% 四个位置作为候选 delta：

```text
th0 = sorted_abs_diffs[20%]
th1 = sorted_abs_diffs[40%]
th2 = sorted_abs_diffs[60%]
th3 = sorted_abs_diffs[80%]
```

对应 noxim SAP 文件中的：

```text
th0 th1 th2 th3
```

## 7. FAS/SAP 等级搜索流程

FAS、SAP-RLE、SAP-RLE-V2 使用相同的等级搜索流程。

SAP-RLE/SAP-RLE-V2 在模拟近似通信时，默认每 6 个数为一组，并且分组不跨通道。

第一阶段：全局统一等级搜索。

1. 所有非 shortcut 边使用同一个等级。
2. 从低等级开始测试。
3. 如果满足 2% 精度约束，继续提高全局等级。
4. 如果不满足约束，回退到上一档全局等级。

第二阶段：按层逐步提高等级。

1. 以第一阶段得到的全局等级为初始配置。
2. 按层输出神经元数量从大到小排序。
3. 逐层尝试提高该层输出边的近似等级。
4. 如果提高后仍满足 2% 精度约束，则保留。
5. 如果提高后不满足 2% 精度约束，则回退该层，并跳过该层。
6. 直到所有层都尝试完成。

等级含义：

```text
-1: 关闭该边近似
 0: 使用 th0
 1: 使用 th1
 2: 使用 th2
 3: 使用 th3
```

## 8. ABDTR 搜索流程

ABDTR 使用丢弃间隔进行搜索。

第一阶段：全局间隔搜索。

1. 从全局间隔 `1` 开始。
2. 如果不满足 2% 精度约束，则逐步增大全局间隔。
3. 直到刚好满足精度约束。

这里的含义是：

- 间隔越小，丢弃越频繁，近似越激进。
- 间隔越大，丢弃越少，近似越保守。

第二阶段：按层降低间隔。

1. 以第一阶段得到的全局间隔为初始配置。
2. 按层输出神经元数量从大到小排序。
3. 逐层尝试降低该层间隔。
4. 如果降低后仍满足 2% 精度约束，则保留。
5. 如果降低后不满足 2% 精度约束，则回退该层，并跳过该层。
6. 直到所有层都尝试完成。

## 9. 输出文件要求

Python 搜索脚本、统计数据和结果文档放在：

```text
Resnet8_CIFAR10/
```

最终 markdown 结果至少包含：

- exact baseline accuracy。
- FAS 每层阈值和最终等级配置。
- SAP-RLE 每层 delta 阈值和最终等级配置。
- SAP-RLE-V2 每层 delta 阈值和最终等级配置。
- ABDTR 全局/逐层间隔配置。
- 每轮关键搜索记录。
- 最终分类精度。
- shortcut 边关闭近似的说明。
- Python 阈值与 noxim 阈值不完全一致的说明。

## 10. noxim 配置映射

FAS 推荐写入：

```text
resnet8_fas_threshold.txt
resnet8_fas_level_table.txt
resnet8_fas_edge_approx.txt
```

SAP-RLE/SAP-RLE-V2 推荐写入：

```text
resnet8_sap_threshold.txt
resnet8_sap_level_table.txt
resnet8_sap_edge_approx.txt
```

ABDTR 推荐写入：

```text
resnet8_drop.txt
```

其中 FAS 使用 `Edge` 行，SAP-RLE/SAP-RLE-V2 使用 `SapEdge` 行，二者不要混用。
