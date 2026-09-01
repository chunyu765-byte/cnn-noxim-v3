# cnn-noxim PE 计算并行加速修改思路

## 1. 问题背景

当前 cnn-noxim 在执行 AlexNet、VGG11、VGG16 等 CNN 模型仿真时，运行时间主要消耗在 PE 内部的卷积层和全连接层计算部分。虽然 NoC 硬件结构中 PE、router、link 等模块具有并行性，但当前仿真程序基于 SystemC 事件驱动执行，实际软件仿真过程基本是单线程推进。因此在 Linux 中通常表现为 `noxim` 进程占用约 100% CPU，即吃满一个 CPU 核心。

如果希望一次仿真使用多个 CPU 核心加速，不能简单依靠 `taskset`。`taskset -c 0,1,2` 只能限制进程允许运行在哪些核心上，不能自动把单线程程序变成多线程程序。真正的加速需要在代码中引入并行计算。

## 2. 总体原则

推荐的原则是：

```text
不并行 SystemC 时间推进；
只并行 PE 内部纯计算部分。
```

也就是说，不改 router、link、VC、buffer、flit 收发、SystemC 事件调度这些和时序强相关的模块，只在某个 PE 已经满足接收条件、准备执行当前层计算时，把该 PE 内部的输出神经元计算循环并行化。

这样做的目标是减少真实运行时间，也就是墙钟时间，不改变仿真统计出来的 cycle。

## 3. 不建议并行的部分

以下部分暂时不建议并行：

- SystemC 仿真主循环。
- router 仲裁和 flit 转发。
- link delay 和 router delay。
- VC/buffer 状态更新。
- flit 注入和接收。
- 自动停止逻辑。
- 多个近似通信方案同时运行。
- 多张图片并行推理。

原因是这些部分和仿真时间、事件顺序、共享状态高度相关，贸然并行容易导致竞态、非确定性结果、统计错误或崩溃。

## 4. 优先并行的部分

优先考虑以下三个位置：

1. 第一层卷积计算。
2. 普通卷积层计算。
3. 全连接层计算。

这些部分共同特点是：当输入数据已经准备好后，不同输出神经元之间通常没有数据依赖，可以按输出神经元维度并行。

## 5. 卷积层并行思路

当前卷积层大致逻辑是：

```cpp
for (int bg = 0; bg < Use_Neu; bg++) {
    value = 0;
    for (int bl = 0; bl < conv_z; bl++) {
        for (int fg = 0; fg < size_conv; fg++) {
            value += input_val * weight;
        }
    }
    res[wz][bg] = activation(value);
}
```

可以改为按 `bg` 并行：

```cpp
#pragma omp parallel for
for (int bg = 0; bg < Use_Neu; bg++) {
    long long local_acc = 0;

    for (int bl = 0; bl < conv_z; bl++) {
        for (int fg = 0; fg < size_conv; fg++) {
            local_acc += input_val * weight;
        }
    }

    local_res[bg] = activation(local_acc);
}
```

推荐不要在并行区域里直接写复杂共享结构，而是：

```text
并行区：只计算 local_res[bg]
并行区外：统一写回 res[wz][bg]
```

## 6. 全连接层并行思路

当前全连接层如果是：

```cpp
for (int j = 0; j < receive; j++) {
    for (int i = 0; i < Use_Neu; i++) {
        res[wz][i] += input[j] * weight[i][j];
    }
}
```

这个结构不适合直接并行，因为多个线程可能同时写 `res[wz][i]`。

建议改成按输出神经元 `i` 并行：

```cpp
#pragma omp parallel for
for (int i = 0; i < Use_Neu; i++) {
    long long local_acc = 0;

    for (int j = 0; j < receive; j++) {
        local_acc += input[j] * weight[i][j];
    }

    local_res[i] = activation(local_acc);
}
```

这种写法每个线程只负责自己的输出神经元，避免多个线程同时累加同一个结果。

## 7. 能耗统计处理

当前 PE 计算中存在：

```cpp
stats.power.compute(...)
```

这类统计函数如果直接放进 OpenMP 并行区域，可能不是线程安全的。

推荐方案是：

```text
并行区内：每个线程只统计本地乘法次数、近似乘法次数、shift 次数等；
并行区外：把各线程计数汇总后，再统一调用 stats.power.compute。
```

如果 `stats.power.compute()` 目前不支持批量调用，可以先串行补记：

```cpp
for (long long k = 0; k < total_mul_count; k++) {
    stats.power.compute(EXACT_MUL);
}
```

这样虽然能耗统计部分仍然串行，但能避免线程安全问题。后续可以再增加批量统计接口。

## 8. 日志打印处理

并行区域内不能保留大量 `cout`，否则会出现：

- 日志顺序混乱。
- 多线程争用输出锁。
- 性能下降。
- 调试信息难以阅读。

推荐做法：

```text
并行区内不打印；
并行区外只打印层级信息、PE 信息、总计信息。
```

如需调试某一个 PE 或某一层，可以用编译期开关或运行时开关限制打印范围。

## 9. 对 cycle 统计的影响

理论上不会影响仿真 cycle。

原因是 cnn-noxim 的计算 cycle 不是由真实 CPU 运行时间自动决定，而是由代码中的 `formula_time` / `computation_time` 计算得到。例如：

```cpp
formula_time = receive * ((Use_Neu + 31) / 32) + ...
computation_time = formula_time;
```

OpenMP 只减少真实执行时间，不应该改变：

- `formula_time`
- `computation_time`
- 当前层开始发送时间
- 下一层开始计算时间
- router/link delay
- Global average delay
- Total simulation time

前提是并行化不改变输出数据、不改变 flit 数量、不改变事件调度顺序。

## 10. 对通信统计的影响

理论上也不会影响通信统计。

通信统计主要由以下内容决定：

- 数据包数量。
- flit 数量。
- flit 注入时间。
- router buffer 状态。
- router delay。
- link delay。
- 拓扑和映射。
- 当前层计算完成后的发送时间。

只要 PE 内部计算结果和 `computation_time` 不变，通信行为应该保持一致。

## 11. 主要风险

主要风险包括：

- 并行区内写共享变量导致竞态。
- `stats.power.compute()` 线程不安全。
- 多线程写 `res`、`output_tmp`、`receive_data` 导致结果不稳定。
- `cout` 并行输出导致日志混乱。
- 计算顺序变化导致浮点累加误差。
- 对近似计算方案的能耗统计产生偏差。

其中浮点累加误差需要特别注意。如果保持整数乘加，并且每个输出神经元内部的累加顺序不变，则结果应当稳定。如果改成浮点并改变累加顺序，可能导致微小差异，Tanh 下可能被放大。

## 12. 推荐实现阶段

### 阶段一：只做精确计算并行

先只针对 `approx_compute == 0` 的精确 PE 计算路径做并行。

目标：

- AlexNet 精确仿真输出完全一致。
- Total received flits 完全一致。
- Total simulation time 完全一致。
- Global average delay 完全一致或极接近。
- Compute energy 完全一致。

### 阶段二：扩展到通信近似方案

通信近似方案包括：

- ABDTR
- FAS/isapprox
- SAP-RLE
- SAP-RLE-V2

这些方案主要影响传输数据，不应改变 PE 精确计算本身。只要接收端恢复后的 `receive_data` 一致，PE 并行计算可以复用精确路径。

### 阶段三：处理近似计算方案

如果后续还要支持 `DRUM6`、`DCY_MUL`、`BIASED_MUL`、`UNBIASED_MUL` 等近似乘法，则需要单独处理能耗统计和阈值判断逻辑。

### 阶段四：优化数据结构

在 OpenMP 验证正确后，可以进一步优化数据结构：

- 将计算热点中的 `deque` 替换为 `vector`。
- 减少内层循环里的线性查找。
- 预先构建输入索引表。
- 预先展开权重访问顺序。
- 减少每个输出神经元重复构造临时容器。

这部分可能比 OpenMP 更稳定，但需要更多代码整理。

## 13. 编译配置

如果使用 OpenMP，需要在编译参数中加入：

```text
-fopenmp
```

链接阶段也需要包含 `-fopenmp`。需要检查当前 Makefile，把该参数同时加入 CXXFLAGS 和 LDFLAGS 或对应变量中。

运行时可以限制线程数：

```bash
export OMP_NUM_THREADS=3
```

也可以配合：

```bash
taskset -c 0,1,2
```

推荐运行方式：

```bash
OMP_NUM_THREADS=3 taskset -c 0,1,2 ./build/noxim ...
```

## 14. 验证方法

建议先用 AlexNet 做最小验证。

验证顺序：

1. 运行修改前 AlexNet 精确仿真，保存日志。
2. 运行修改后 AlexNet 精确仿真，保存日志。
3. 对比最终 `output.txt`。
4. 对比 `Total received flits`。
5. 对比 `Total simulation time`。
6. 对比 `Global average delay`。
7. 对比 `Compute energy`。

如果精确仿真完全一致，再测试：

1. ABDTR。
2. FAS/isapprox。
3. SAP-RLE。
4. SAP-RLE-V2。

最后再迁移到 VGG11 和 VGG16。

## 15. 推荐结论

最稳妥的改进路线是：

```text
先不改 SystemC 通信仿真；
先不改 flit 和 router；
先只并行 PE 内部输出神经元计算；
把统计、日志、全局状态写入留在并行区外；
从 AlexNet 精确模式验证完全一致后，再推广到其他方案。
```

这样可以最大程度降低对 cycle 统计和通信统计的影响，同时利用多核 CPU 减少真实等待时间。
