# ResNet8 非 Shortcut 边阈值等级修改说明

## 1. 结论

如果要对 ResNet8 的每条非 shortcut 边独立选择近似等级，优先修改 noxim 输入目录下的逐边配置文件：

- FAS：`bin/resnet8_cifar10/resnet8_fas_edge_approx.txt`
- SAP-RLE / SAP-RLE-V2：`bin/resnet8_cifar10/resnet8_sap_edge_approx.txt`

运行命令中对应使用：

- FAS：`-NNfas_edge_approx bin/resnet8_cifar10/resnet8_fas_edge_approx.txt`
- SAP-RLE / SAP-RLE-V2：`-sap_edge_approx bin/resnet8_cifar10/resnet8_sap_edge_approx.txt`

逐边文件优先级高于逐层默认阈值表和等级表。也就是说，只要某条真实边在逐边文件里出现，PE 发包时会使用这条边自己的阈值和等级。

## 2. 行格式

FAS 行格式：

```text
Edge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5
```

SAP-RLE / SAP-RLE-V2 行格式：

```text
SapEdge src_layer dst_layer config_sel th0 th1 th2 th3 lv0 lv1 lv2 lv3 lv4 lv5
```

字段含义：

- `src_layer`：发送激活值的源层编号。
- `dst_layer`：接收激活值的目标层编号。
- `config_sel`：当前选择第几个等级配置，取值范围 `0..5`。
- `th0..th3`：该边使用的 4 个阈值。
- `lv0..lv5`：6 个候选近似等级，取值范围 `-1..3`。
- 实际生效等级是：`lv[config_sel]`。
- `-1` 表示该边不近似，保持精确通信。

## 3. 推荐手动修改方式

如果 GPU 搜索结果给出“每条边一个最终等级”，推荐固定 `config_sel=0`，然后只改 `lv0`：

```text
Edge 5 6 0 602 1293 2149 3351 2 1 2 3 3 3
```

上例中 `config_sel=0`，所以实际使用 `lv0=2`。后面的 `lv1..lv5` 不会影响本次运行。

如果你想保留 6 套候选配置用于批量 sweep，则保持 `lv0..lv5 = 0 1 2 3 3 3`，然后只改每条边的 `config_sel`：

```text
Edge 5 6 2 602 1293 2149 3351 0 1 2 3 3 3
```

上例中 `config_sel=2`，所以实际使用 `lv2=2`。

## 4. ResNet8 非 Shortcut 边列表

这些边可以手动配置近似等级：

| 边 | 含义 | 修改位置 |
| --- | --- | --- |
| `1 -> 2` | `conv1` 到 `block1.conv1` | 对应 `Edge/SapEdge 1 2 ...` 行 |
| `2 -> 3` | `block1.conv1` 到 `block1.conv2` | 对应 `Edge/SapEdge 2 3 ...` 行 |
| `3 -> 4` | `block1.conv2` 到 `block1.add` | 对应 `Edge/SapEdge 3 4 ...` 行 |
| `4 -> 5` | `block1.add` 到 `block2.conv1` | 对应 `Edge/SapEdge 4 5 ...` 行 |
| `5 -> 6` | `block2.conv1` 到 `block2.conv2` | 对应 `Edge/SapEdge 5 6 ...` 行 |
| `6 -> 8` | `block2.conv2` 到 `block2.add` | 对应 `Edge/SapEdge 6 8 ...` 行 |
| `8 -> 9` | `block2.add` 到 `block3.conv1` | 对应 `Edge/SapEdge 8 9 ...` 行 |
| `9 -> 10` | `block3.conv1` 到 `block3.conv2` | 对应 `Edge/SapEdge 9 10 ...` 行 |
| `10 -> 12` | `block3.conv2` 到 `block3.add` | 对应 `Edge/SapEdge 10 12 ...` 行 |
| `12 -> 13` | `block3.add` 到 `avgpool` | 对应 `Edge/SapEdge 12 13 ...` 行 |
| `13 -> 14` | `avgpool` 到 `fc` | 对应 `Edge/SapEdge 13 14 ...` 行 |

## 5. ResNet8 Shortcut 边列表

这些边按当前规则不做近似，逐边文件中应保持 `lv0..lv5 = -1 -1 -1 -1 -1 -1`：

| 边 | 含义 |
| --- | --- |
| `1 -> 4` | `conv1` identity shortcut 到 `block1.add` |
| `4 -> 7` | `block1.add` 到 `block2.shortcut.0` 的 1x1 卷积分支输入 |
| `7 -> 8` | `block2.shortcut.0` 输出到 `block2.add` |
| `8 -> 11` | `block2.add` 到 `block3.shortcut.0` 的 1x1 卷积分支输入 |
| `11 -> 12` | `block3.shortcut.0` 输出到 `block3.add` |

注意：`4 -> 7` 和 `8 -> 11` 虽然是送入 1x1 shortcut 卷积的边，但按当前实验规则也视为 shortcut 相关边，不做近似。

## 6. 当前已写入的阈值文件

Python 统计得到的激活 q16 阈值已经写入：

- FAS 阈值：`bin/resnet8_cifar10/resnet8_fas_threshold.txt`
- FAS 兼容别名：`bin/resnet8_cifar10/resnet8_approx.txt`、`bin/resnet8_cifar10/approx.txt`
- SAP 阈值：`bin/resnet8_cifar10/resnet8_sap_threshold.txt`
- SAP 兼容别名：`bin/resnet8_cifar10/sap_threshold.txt`
- FAS 逐边模板：`bin/resnet8_cifar10/resnet8_fas_edge_approx.txt`
- FAS 逐边兼容别名：`bin/resnet8_cifar10/resnet8_edge_approx.txt`、`bin/resnet8_cifar10/edge_approx.txt`
- SAP 逐边模板：`bin/resnet8_cifar10/resnet8_sap_edge_approx.txt`
- SAP 逐边兼容别名：`bin/resnet8_cifar10/sap_edge_approx.txt`

原文件备份目录：`backup_resnet8_python_thresholds_20260901_113512`。

## 7. 代码读取位置

相关 noxim 代码位置：

- `src/NNModel.cpp`：解析 `Edge` / `SapEdge` 逐边配置文件。
- `src/NoximProcessingElement.cpp`：发包时根据目标 PE 推断 `dst_layer`，再按 `(src_layer, dst_layer)` 取逐边阈值和等级。
- `src/NoximCmdLineParser.cpp`：命令行参数 `-NNfas_edge_approx`、`-NNedge_approx`、`-sap_edge_approx`。


## 8. ABDTR 逐边配置

现在 ABDTR 也支持逐边配置，优先修改：

- `bin/resnet8_cifar10/resnet8_abdtr_edge_drop.txt`
- 兼容短别名：`bin/resnet8_cifar10/abdtr_edge_drop.txt`

运行命令中加入：

```text
-acdc_abdtr 1 -abdtr_drop_file resnet8_cifar10/resnet8_drop.txt -abdtr_edge_drop_file resnet8_cifar10/resnet8_abdtr_edge_drop.txt
```

逐边文件格式：

```text
AbdtrEdge src_layer dst_layer interval
```

字段含义：

- `src_layer`：发送激活值的源层编号。
- `dst_layer`：接收激活值的目标层编号。
- `interval`：ABDTR 丢弃间隔。
- `interval >= 0`：该边使用逐边 interval，约每 `interval + 1` 个 body flit 丢弃一个。
- `interval = -1`：该边不使用 ABDTR，保持精确通信。
- 文件中没写到的边，会回退到逐层 `resnet8_drop.txt`。

当前模板里，非 shortcut 边默认 `14`，shortcut 相关边默认 `-1`。

## 9. ABDTR 代码读取位置

- `src/NoximCmdLineParser.cpp`：新增 `-abdtr_edge_drop_file` 参数。
- `src/NNModel.cpp`：解析 `AbdtrEdge src dst interval` 文件。
- `src/NoximProcessingElement.cpp`：发包压缩前按 `(src_layer, dst_layer)` 查询逐边 interval；命中则覆盖逐层 interval。

验证记录：短运行已经确认 noxim 能加载逐边文件，日志显示 `ABDTR per-edge drop intervals loaded ... (edges=16)`。后续短测在热模型 `Thermal_IF::hs_temperature` 处触发已有段错误，和逐边配置解析无关。
