# NoC推理CNN的过程

本文基于当前工程源码整理（`/home/neil_chun/cnn-noxim`），按真实执行时序说明一次 CNN 推理在 NoC 上是如何完成的。内容覆盖：

- 程序入口与命令行参数
- 模型/权重/输入的加载与映射
- PE 的接收、计算、发送主循环
- 第一层“直接计算”与后续层“先收后算”的区别
- 近似通信（阈值近似、all-zero、ABDTR、drop/trunc）的压缩与恢复
- 日志与输出文件的对应关系

---

## 1. 运行入口与总体时序

### 1.1 编译与入口

- 可执行入口脚本：`bin/noxim`
- 实际程序：`bin/build/noxim`
- 由 `bin/Makefile` 控制编译输出目录 `build/`

即：

1. `./bin/compile`（内部调用 `make -C bin`）
2. `./bin/noxim ...`（转发到 `./bin/build/noxim`）

### 1.2 仿真主入口

`sc_main` 在 `src/NoximMain.cpp`：

1. `parseCmdLine(...)` 解析参数
2. 创建 `NoximNoC` 实例（构造时会 `buildMesh()`）
3. `reset=1` 跑复位周期
4. `reset=0` 进入正式仿真 `sc_start(simulation_time * CYCLE_PERIOD)`
5. 仿真结束后统计并退出

---

## 2. 命令行参数如何影响推理

参数解析在 `src/NoximCmdLineParser.cpp`。

### 2.1 与 CNN 推理最相关的参数

- `-NNmodel`：模型结构（如 `lenet5/model_new.txt`）
- `-NNweight`：权重文件（如 `lenet5/weight_fc_wb2parser.txt`）
- `-NNinput`：输入数据
- `-NNlabel`：标签（用于 accuracy）
- `-mapping`：映射方式（`dir_x/dir_y/random/table`）
- `-groupsize`：每组神经元数
- `-packet_size`：每包 body flit 数（包总 flit = body + 2）
- `-sim`：仿真周期

### 2.2 近似通信互斥检查

代码明确规定以下通信近似方案互斥，一次只能启用一个：

- `-isapprox 1`（阈值近似）
- `-allzeropacket 1`（全0数据包）
- `-acdc_abdtr 1`（ABDTR方案）
- `-is_drop_trunc 1`（截位）

如果同时开启多套，会直接报错退出（防止恢复元信息冲突）。

---

## 3. 模型/权重/输入加载（NNModel::load）

核心在 `src/NNModel.cpp -> NNModel::load()`。

## 3.1 模型文本解析

按行解析 `Input / Convolution / Pooling / Dense`，写入：

- `all_leyer_type[tdm][layer]`：层类型（`i/c/p/f`）
- `all_leyer_size[tdm][layer][...]`：层参数
- `all_layer_in_scales / all_layer_output_scales`
- `all_layer_in_zp / all_layer_out_zp`

> 当前 `bin/lenet5/model_new.txt` 的层序（示例）：
>
> 0: Input  
> 1: Conv  
> 2: Pool  
> 3: Conv  
> 4: Pool  
> 5: Dense  
> 6: Dense  
> 7: Dense

## 3.2 关于 `weight_scale.txt`

当前代码里 `weight_scale.txt` 的读取逻辑被注释掉了（未生效）。  
当前缩放主要来自 `model_new.txt` 每层定义的 `in_scale/out_scale`。

## 3.3 近似阈值与等级表

- `approx.txt` 读入 `all_layer_approx`（每层4个阈值）
- `approx_level_table.txt` 读入 `all_layer_approx_level_table`（每层配置档位）
- `drop.txt` 读入 `drop_rate_new`（ABDTR 间隔参数）

## 3.4 权重与偏置读取

### 卷积层

- 先读每个输出通道的卷积核权重（按输入通道 × kernel）
- 再读该层所有输出通道 bias
- 存到 `all_conv_weight` 与 `all_conv_bias`

### 全连接层

按每个输出神经元：

1. 读完上一层所有输入权重
2. 再补一个 bias 到该神经元 `weight.back()`

即 FC 的 `NeuInfo.weight = [w0...wN-1,bias]`。

## 3.5 分组与映射（关键）

神经元被切成 group，group 再映射到 PE：

- `Group_table`：每个 group 里有哪些神经元
- `mapping_table[group_id] = pe_id`
- `all_leyer_ID_Group[layer]`：某层有哪些 group

这里有个非常关键的层号定义：

- `Input` 在模型里是 layer 0
- 参与 NoC 计算的第一层（例如第一个 Conv）是 `ID_layer = 1`

因此日志里经常看到 `ID_layer` 从 1、2、3...，不会出现输入层作为计算层。

## 3.6 坐标缓存与收发预计算

加载/计算：

- `all_conv_coord`：卷积每个输出位置对应上一层哪些输入位置
- `all_pool_coord`：池化窗口坐标

并做全局收发预计算（很关键）：

- `PE_send_list`：某 PE 发送到哪些目标 PE（去重后）
- `PE_send_req_list`：对应每个目标 PE 需要多少数据
- `PE_send_conv_list / PE_send_pool_list`：保留原始连接信息（含重复）供精确发包
- `PE_receive_conv_list`：每个本地输出神经元要接收哪些源神经元

---

## 4. NoC 构建（NoximNoC::buildMesh）

`src/NoximNoC.cpp` 中：

1. `nnmodel.load()` 完成模型侧准备
2. 按 `mesh_dim_x/y/z` 实例化 Tile
3. 每个 Tile 的 PE 拿到：
   - `local_id`
   - `NN_Model` 指针
4. 连接 Router/PE 各方向信号
5. 清边界信号、初始化统计数组

`NoximNoC.cpp` 里统计数组目前是固定 16 维：

- `cnt_neighbor_total[16]`
- `cnt_received_total[16]`
- `cnt_local_total[16]`
- `tot_cnt_local[16][16][16]`
- `tot_cnt_neighbor[16][16][16]`
- `tot_cnt_received[16][16][16]`

---

## 5. PE 级推理主流程（最核心）

PE 有两个并行 SystemC 进程：

- `rxProcess()`：接收/恢复/触发计算
- `txProcess()`：初始化、直接计算第一层、发包发 flit

## 5.1 复位阶段（reset=1）

### txProcess 复位做什么

- 清状态、队列、计数器
- 根据 `mapping_table` 找到当前 PE 对应 `ID_group`
- 从 `Group_table` 取本 PE 的神经元表 `PE_table`
- 得到 `ID_layer / Type_layer / Use_Neu`
- 从预计算表填充：
  - 发往哪些 PE
  - 每个目标发送多少
  - 本地需要接收哪些神经元 ID

### rxProcess 复位做什么

- 初始化接收缓存与恢复辅助队列
- 构造 `receive_Neu_ID`（去重）
- 初始化 `should_receive[pic]` 与 `receive_data[pic][...]`

---

## 6. 第一层为什么“直接计算”

在 `txProcess()` 复位分支里，若当前 PE 的 `Type_layer=='c'` 且 `ID_layer==1`，会直接从输入内存计算：

- 输入来自 `NN_Model->all_data_in[0][pic][input_idx]`
- 不等收包（因为输入层不是 NoC 计算层，没有上游 PE 发包）
- 计算完直接把结果写到 `res[pic][local_neuron]`
- 然后进入正常“发包到下一层”阶段

这就是你日志里常见现象的根因：

- 第一层是“先算再发”
- 后续层是“先收齐再算”
- 所以很多“开始计算”的日志首先出现在 `ID_layer=2`（第二个计算层）处

---

## 7. 后续层的统一流程：先收齐再算

### 7.1 收包写入

`rxProcess()` 每收到 flit：

1. 记录到 `r_flit_vector[vc]`
2. 如果是近似包，先读取 head 中的恢复元信息
3. tail 到达后，执行整包恢复（若启用）
4. 遍历包内 body flit，根据 `src_Neu_id` 定位 `receive_Neu_ID` 索引
5. 写入 `receive_data[pic][idx]`
6. `should_receive[pic]--`

当 `should_receive[pic]==0`，触发该 PE 当前层计算。

### 7.2 层计算公式（当前实现）

### 全连接（f）

对每个输出神经元 i：

1. `acc += (x_j - input_zp) * w_ij`
2. 在加 bias 前：`acc *= input_data_scale`
3. `acc += bias_i`
4. 激活（ReLU/Tanh/Sigmoid/Softmax）
5. 最后一层写 `output.txt`，并按 `label.txt` 统计 accuracy

### 卷积（c）

对每个输出神经元：

1. 按 `receive_neu_ID_conv` 取到卷积窗输入
2. `acc += (x - input_zp) * w`
3. 在加 bias 前：`acc *= input_data_scale`
4. `acc += bias(channel)`
5. 激活（常见 ReLU）

### 池化（p）

- `MAX`：窗口内取最大
- `AVG`：窗口求和后除窗口大小

---

## 8. 发包与装 flit

## 8.1 何时开始发

`txProcess()` 在以下条件满足后开始发：

- `PE_enable`
- 不是最后层
- 该图片该 PE 计算完成 (`flag_p[pic]==1`)
- 当前时间达到 `temp_computation_time + computation_time`

## 8.2 packet 生成

每个目标 PE 生成若干 packet：

- `packet.size = body_count + 2`（head+tail）
- packet 携带该层 4 个近似阈值 `approx_threshold`

## 8.3 nextFlit 装载数据

`nextFlit()`：

- 先出 HEAD，再 BODY，再 TAIL
- BODY 的 `src_Neu_id` 与 `data` 按层类型从 `res` 取值
- `src_Neu_id` 不是“顺序自增假设”，是按映射/连接关系确定

这也是恢复必须依赖 `src_Neu_id`（以及 `approx_src_id`）的原因。

---

## 9. 近似通信：压缩与恢复

## 9.1 四种方案（互斥）

一次只能启用一个：

1. 阈值近似 `-isapprox 1`
2. 全零包 `-allzeropacket 1`
3. ABDTR `-acdc_abdtr 1`
4. drop/trunc `-is_drop_trunc 1`

## 9.2 阈值近似（isapprox）

发送端：

- 对 body flit 按阈值判断
- 满足条件就删掉该 body
- 在 head 记录：
  - `approx_pos`
  - `approx_level`
  - `approx_src_id`

接收端（tail 到达后）：

- 按 `approx_pos` 在包内插回缺失 body
- `src_Neu_id` 优先用 `approx_src_id`
- 数据值按等级恢复：
  - level0 -> 0
  - level1 -> th0
  - level2 -> th1
  - level3 -> th2

## 9.3 ABDTR（acdc_abdtr）

发送端：

- 从 `drop.txt` 读出该层 `interval`
- `drop_interval = interval + 1`
- 对 body flit 按计数 `counter % drop_interval == 0` 丢弃
- 丢弃位置与 `src_Neu_id` 写入 head 元信息

注意：

- 计数是 body-flit 计数，不含 head/tail
- 计数按图片号在 PE 内累积，不是“每个包重置”
- 所以总 flit 不会严格等于 “原总 flit / 2”

接收端：

- 在缺失位置插回 body
- 数据恢复使用线性插值：
  - `data = (prev_body + next_body)/2`
  - 缺左/缺右则退化拷贝邻居
  - 两边都无则置 0

## 9.4 drop/trunc

发送端会做位截断与多种 combine 模式打包；
接收端按 `combine_mode` 展开恢复。
这套逻辑与 ABDTR 不同，不应混用。

---

## 10. 关于 ID_layer：从几开始，何时变化

## 10.1 从几开始

- 输入层是 layer 0（不作为 PE 计算层）
- 第一计算层（通常第一卷积）是 `ID_layer=1`

来源：`NNModel::load()` 中分组构建时 `temp_layer` 从 1 开始。

## 10.2 何时变化

- 单次仿真中，某个 PE 的 `ID_layer` 在 reset 时由映射固定
- 进入下一 TDM 子网络时，在 `tdm_reset()` 中切换到下一份 `Group_table`，`ID_layer` 可能变化

---

## 11. 输出文件与日志对应关系

## 11.1 `output.txt`

最后层输出写入：

- `pic_no: X No.Y output neuron result: ...`
- 并可能附带 `accuracy: ...`

## 11.2 `PE_log/PE_R_x` 与 `PE_log/PE_T_x`

- 接收日志：`PE_R_x`（复位时会删）
- 发送日志：`PE_T_x`（当前代码是追加写，不会在复位里自动删）

因此若多次运行想避免历史混杂，建议手动清理 `bin/PE_log/PE_T_*`。

## 11.3 终端 run_log

你通常用重定向保存命令行输出到 `bin/run_log/run_log_*.txt`，该日志包含：

- `ID_layer`
- `input_data_scale / output_scale`
- 每层开始计算/发送时刻
- 近似模式启用信息

---

## 12. 你当前这套定点缩放链路在代码里的落点

你的核心设定是：

- 输入放大 `1/input_scale`
- 权重和偏置放大 `1/weight_scale`
- 在加 bias 前乘 `input_scale` 或对应层 scale

代码里对应位置是 Conv/FC 的这一步：

- 先做整数 MAC
- 再 `value *= input_data_scale`
- 再 `+ bias`

因此从“流程设计”上，它确实是“先把 MAC 拉回 bias 同域，再加 bias”的路径。

---

## 13. 一次完整推理的时序图（文字版）

1. 启动程序，解析命令行  
2. 加载模型、权重、输入、近似配置  
3. 构建 Group 与 PE 映射，预计算收发列表  
4. 全网 reset  
5. 第一层 Conv PE 直接从输入做卷积  
6. 第一层把结果打包成 flit 发送  
7. 第二层开始收包，收齐后计算  
8. 重复“收齐->计算->发包”直到最后层  
9. 最后层写 `output.txt` 并计算准确率  
10. 仿真结束，输出统计

---

## 14. 典型命令模板

## 14.1 精确通信

```bash
./bin/noxim \
  -dimx 8 -dimy 8 -dimz 1 \
  -NNmodel lenet5/model_new.txt \
  -NNweight lenet5/weight_fc_wb2parser.txt \
  -NNinput lenet5/input7_new.txt \
  -NNlabel lenet5/label7.txt \
  -NNapprox lenet5/approx.txt \
  -NNapprox_Level_Table lenet5/approx_level_table.txt \
  -mapping dir_x -groupsize 1024 \
  -packet_size 4 -sim 20000 \
  -isapprox 0 -allzeropacket 0 -acdc_abdtr 0 -is_drop_trunc 0
```

## 14.2 仅 ABDTR

```bash
./bin/noxim \
  -dimx 8 -dimy 8 -dimz 1 \
  -NNmodel lenet5/model_new.txt \
  -NNweight lenet5/weight_fc_wb2parser.txt \
  -NNinput lenet5/input7_new.txt \
  -NNlabel lenet5/label7.txt \
  -NNapprox lenet5/approx.txt \
  -NNapprox_Level_Table lenet5/approx_level_table.txt \
  -mapping dir_x -groupsize 1024 \
  -packet_size 4 -sim 20000 \
  -isapprox 0 -allzeropacket 0 -is_drop_trunc 0 -acdc_abdtr 1
```

---

## 15. 关键结论（便于快速回看）

1. `ID_layer` 的计算层从 1 开始，输入层是 0 但不走 PE 计算。  
2. 第一层卷积在 `txProcess` reset 分支里“直接计算”，不是先收包。  
3. 后续层统一是“收齐所需神经元 -> 触发计算 -> 发包”。  
4. 近似通信四种模式互斥，混开会被命令行检查拦截。  
5. ABDTR 恢复当前采用线性插值（前后 body 平均）。  
6. `weight_scale.txt` 当前未参与加载，缩放来源是 `model_new.txt`。  

---

## 16. TXT 文件排布规范（重点）

这一章专门回答“各种 txt 文件到底该怎么排”的问题。  
下面所有规则都按当前源码真实读取逻辑整理。

## 16.1 `model_new.txt` 的排布

解析函数：`NNModel::load()`。

### Input 行格式

```txt
Input X Y C input_scale
```

含义：

- `X Y C`：输入宽高通道
- `input_scale`：输入层 scale（float）

你的 `lenet5/model_new.txt` 第1行示例：

```txt
Input 28 28 1 3.0519440883843007996093511566868e-5
```

### Convolution 行格式

```txt
Convolution out_x out_y out_ch kx ky kz stride pad act weight_scale approx_threshold bn in_scale in_zp out_scale out_zp
```

注意点：

- `kz` 是输入通道数
- `weight_scale/in_scale/out_scale` 是 float
- 当前代码把 `weight_scale` 存进 `all_leyer_size`（int 容器）时会发生截断，但真实计算主要用 `all_layer_in_scales / all_layer_output_scales`

`lenet5/model_new.txt` 第2行示例：

```txt
Convolution 28 28 6 5 5 1 1 2 relu 1.712739021954845e-05 0 0 3.0519440883843007996093511566868e-5 0 1.712739021954845e-05 0
```

### Pooling 行格式

```txt
Pooling out_x out_y ch kx ky stride mode approx_threshold
```

示例：

```txt
Pooling 14 14 6 2 2 2 maximum 0
```

### Dense 行格式

```txt
Dense out_neuron act weight_scale approx_threshold output_scale_int_placeholder in_scale in_zp out_scale out_zp
```

注意点：

- 这里第5个字段 `output_scale_int_placeholder` 是历史兼容字段（常见写 16）
- 当前真实输出 scale 主要来自后面的 `out_scale`（float）

示例（第8行）：

```txt
Dense 10 none 1.2331280021587152e-05 0 16 1.0506284099751971e-05 0 1.2331280021587152e-05 0
```

---

## 16.2 `weight*.txt` 的排布（最容易出错）

解析函数：`NNModel::load()`。  
权重文件是“按 token 流读取”，换行只影响可读性，不影响解析。

### 总体顺序

1. 先按模型顺序读取所有卷积层（每层：weights 后 bias）
2. 再按模型顺序读取所有全连接层（每层：先 bias，再所有权重）

### 卷积层排布（每一层）

顺序如下：

1. 对每个输出通道 `oc`
2. 对每个输入通道 `ic`
3. 连续写 `kx*ky` 个 kernel 权重
4. 所有 `oc` 的 weights 写完后，写该层 `out_ch` 个 bias

单层 token 数：

```txt
conv_tokens = out_ch * kz * (kx*ky) + out_ch
```

### 全连接层排布（每一层，当前 parser 期望）

顺序如下：

1. 先写这一层全部输出神经元的 bias（数量 = `out_neuron`）
2. 再按输出神经元顺序写权重：
   - 第0个输出神经元写 `prev_neuron` 个权重
   - 第1个输出神经元写 `prev_neuron` 个权重
   - ...

单层 token 数：

```txt
fc_tokens = out_neuron + out_neuron * prev_neuron
```

### 你的 lenet5 数量核对（当前模型）

- Conv1: `6*1*5*5 + 6 = 156`
- Conv2: `16*6*5*5 + 16 = 2416`
- FC1: `120 + 120*400 = 48120`
- FC2: `84 + 84*120 = 10164`
- FC3: `10 + 10*84 = 850`
- 总计：`61706`

你当前文件统计也正好是 `61706` tokens。

### `weight.txt` 与 `weight_fc_wb2parser.txt` 的关系（当前工程实测）

- 两者都 61706 token，但 FC 段排布不同
- 当前 parser + 当前模型下，`weight_fc_wb2parser.txt` 与 `output_original.txt` 完全匹配
- `weight.txt` 在同样配置下会给出明显不同结果

结论：当前请优先使用 `weight_fc_wb2parser.txt`。

---

## 16.3 `input*.txt` 的排布

解析函数：`NNModel::load()` 读取输入。

规则：

- 解析是纯 token 流
- 每 `input_size` 个数切成一张图
- 换行不是硬约束，只是可读性

对 `Input 28 28 1`：

- 每张图必须正好 `784` 个数
- 2张图必须 `1568` 个数

你常用的“每行28个，56行”是人类友好格式，程序完全支持。

### 多图顺序

`input1_7.txt = input1_new + input7_new` 时：

- 前784个 token 是 `pic_no: 0`
- 后784个 token 是 `pic_no: 1`

所以标签文件必须同顺序。

---

## 16.4 `label*.txt` 的排布

解析在最后层输出阶段进行。

规则：

- 空白分隔整数标签
- 标签个数应 >= 输入图片张数
- 第 `wz` 张图用 `label[wz]`

例如两图：

```txt
1
7
```

---

## 16.5 `approx.txt` 的排布

每个“计算层”（Conv/Pool/Dense）一行，Input 不写。  
lenet5 共有 7 个计算层，所以应有 7 行。

格式：

```txt
LayerType v0 v1 v2 v3
```

当前实现会把读入顺序反转存储为 `[th0,th1,th2,th3]`：  
文件里常见写法是 `th3 th2 th1 th0`，例如：

```txt
Convolution 1024 768 512 256
```

会在内部变成：

- `th0=256`
- `th1=512`
- `th2=768`
- `th3=1024`

---

## 16.6 `approx_level_table.txt` 的排布

每个计算层一行，6 个配置值：

```txt
LayerType cfg0 cfg1 cfg2 cfg3 cfg4 cfg5
```

运行时使用：

```txt
all_layer_approx_level_table[layer-1][config_sel]
```

默认 `config_sel=0`，所以常见第一列为 0 表示“基本不近似”。

---

## 16.7 `drop.txt` 的排布（ABDTR）

每个计算层一个整数（可放一行或多行，空白分隔均可）。

当前逻辑：

```txt
interval = drop_rate_new[layer-1]
drop_interval = interval + 1
每当 body_flit_counter % drop_interval == 0 时丢弃
```

例如 `9` 表示“每 10 个 body flit 丢 1 个”。

---

## 16.8 `output.txt` 的排布

格式：

```txt
pic_no: X No.Y output neuron result: V
accuracy: R
```

注意：

- 程序默认追加写
- 不清空就会混入历史结果
- 多图推理时同一文件会有 `pic_no:0`、`pic_no:1` ...

---

## 16.9 `PE_R_x.txt` / `PE_T_x.txt` 的排布

目录：`bin/PE_log/`

- `PE_R_x.txt`：接收 flit 日志
- `PE_T_x.txt`：发送 flit 日志

行格式示例：

```txt
cycle: ProcessingElement[id] SENDING/RECEIVING [type:..., seq:..., src->dst], src_Neu_id=..., data=..., pic_no=..., vc_id=...
```

---

## 17. 文件排布快速自检命令

以下命令可以直接检查常见格式错误。

### 输入 token 数

```bash
awk '{for(i=1;i<=NF;i++) c++} END{print c}' bin/lenet5/input1_7.txt
```

对 2 张 28x28x1 应输出 `1568`。

### 输入是否每行 28 个（可读性检查）

```bash
awk 'NF!=28{print NR,NF}' bin/lenet5/input1_7.txt
```

无输出即通过。

### 权重 token 数

```bash
awk '{for(i=1;i<=NF;i++) c++} END{print c}' bin/lenet5/weight_fc_wb2parser.txt
```

lenet5 当前应为 `61706`。

### 标签数是否匹配图片数

```bash
awk '{for(i=1;i<=NF;i++) c++} END{print c}' bin/lenet5/label1_7.txt
```

2图应至少有 `2`。

---

## 18. 这次你遇到的“不一致”归因（具体到文件）

你之前两图合并时误用了：

- `input1_new.txt + input7.txt`

而不是：

- `input1_new.txt + input7_new.txt`

这会让第二张图输入域与单图基准不一致，因此 `output` 不一致。  
改成 `input7_new` 后，`pic_no:1` 与 `output_original.txt` 已逐项一致。

---

## 19. 最小可复现建议

为了避免后续再踩文件排布坑，建议固定一套“最小验证包”：

1. 固定模型：`model_new.txt`
2. 固定权重：`weight_fc_wb2parser.txt`
3. 单图基准：`input7_new.txt + label7.txt`，保存标准 `output_original.txt`
4. 多图验证：`input1_new + input7_new`，检查 `pic_no:1` 必须逐项等于基准
5. 每次跑前清空 `output.txt`

---

## 20. 每层时间估算公式与合理性解释（源码对应）

这一章专门回答“每层时间到底怎么估、为什么这么估、怎么变成仿真 cycle”的问题。

### 20.1 先有公式还是先有 cycle 统计

当前实现是“先公式估算，再用公式门控发送时刻”，不是先逐 cycle 做详细计算再反推统计。

关键流程（`src/NoximProcessingElement.cpp`）：

1. `rxProcess()` 收齐本层所需输入后，立即进入该层计算。
2. 计算完成后得到 `formula_time`。
3. 记录计算开始时刻：`temp_computation_time[wz] = sc_simulation_time()`。
4. 写入估算计算时长：`computation_time = formula_time`。
5. `txProcess()` 只有在  
   `sc_simulation_time() >= temp_computation_time[pic] + computation_time`  
   时才允许开始发包/发 flit。

因此，`computation_time` 是“计算延迟模型”，直接参与时序推进。

### 20.2 公式里各变量的物理意义

- `Use_Neu`：当前 PE 在当前层负责计算的输出神经元数量（本地输出规模）。
- `receive`：当前 PE 在当前层需要接收的输入激活数量。
- `kernel_size`：卷积核空间大小 `kx * ky`。
- `kernel_z`：卷积输入通道数。
- `((Use_Neu+31)/32)`：按 32 路并行执行分组后的组数（向上取整）。
- `(Use_Neu+31)%32`：尾组补偿项（当前代码按该表达式建模）。

### 20.3 全连接层时间公式

#### 精确计算（`approx_compute == 0`）

```txt
T_fc = receive * ((Use_Neu + 31) / 32) + (Use_Neu + 31) % 32 + 1
```

含义拆解：

1. `receive * ((Use_Neu+31)/32)`：每个输入激活都要与本 PE 全部输出神经元做一次乘加，按 32 路并行折算成组执行。
2. `+(Use_Neu+31)%32`：尾组/对齐补偿。
3. `+1`：固定开销（控制/收尾）。

#### 近似计算（`approx_compute > 0`）

```txt
T_fc_approx = count_zero.cal_cycle * ((Use_Neu + 31) / 32)
            + (Use_Neu + 31) % 32 + 1
            + count_zero.sequencesOfTen * 2
```

其中：

- `count_zero.cal_cycle`、`count_zero.sequencesOfTen` 来自
  `countZerosAndSequences(...)` 或 `counte_zero_skip_cycle(...)`。
- 直观上是把“有效计算周期”与“压缩/恢复附加成本”引入延迟模型。

### 20.4 卷积层时间公式

#### 精确计算

```txt
T_conv = kernel_size * kernel_z * ((Use_Neu + 31) / 32)
       + (Use_Neu + 31) % 32 + 1
```

含义拆解：

1. `kernel_size * kernel_z`：单个输出神经元的 MAC 次数。
2. 乘上 `((Use_Neu+31)/32)`：本 PE 负责的所有输出神经元按 32 路并行折算。
3. `+(...)%32 + 1`：尾项与固定开销。

#### 近似计算

```txt
T_conv_approx = conv_cal_cycle + (Use_Neu + 31) % 32 + 1 + all_sequenceten_zero * 2
```

其中 `conv_cal_cycle` 与 `all_sequenceten_zero` 来自逐输出神经元统计后聚合，体现稀疏/近似对有效计算周期的影响。

### 20.5 池化层时间公式

- 平均池化：

```txt
T_pool_avg = (3 + 1) * ((Use_Neu + 31) / 32)
```

- 最大池化：

```txt
T_pool_max = 3 * ((Use_Neu + 31) / 32)
```

设计意图是：平均池化包含除法，代价高于 max，比 max 多建模一个周期量级。

### 20.6 第一层卷积（直接计算路径）时间公式

第一层卷积不等上游 PE 发包，而是在 `txProcess()` reset 分支直接从输入数据计算，时间公式仍采用卷积通式：

```txt
T_layer1 = kernel_size * kernel_z * ((Use_Neu + 31) / 32)
         + (Use_Neu + 31) % 32 + 1
```

这就是“第一层先算再发、后续层先收再算”的源码根因。

### 20.7 一个具体对齐示例（你之前提到的最后 FC 层）

若最后 FC 层在某 PE 上：

- `receive = 84`
- `Use_Neu = 10`

代入精确公式：

```txt
((10+31)/32) = 1
(10+31)%32   = 9
T_fc         = 84*1 + 9 + 1 = 94
```

若日志显示该层从 `18860` 到 `18954`，差值也是 `94`，与模型一致。

### 20.8 为什么“公式正确”但日志层间隔有时不完全等于公式

这是合理的，原因是公式只建模“计算等待”，全层完成时间还叠加通信与调度因素：

1. 网络注入、路由、仲裁、拥塞、VC 可用性会增加等待。
2. 发送端有 `packet_queue/flit_queue` 过程，通信阶段可能跨多个周期。
3. 代码中当 `packet_queue` 非空时会执行 `temp_computation_time[i]++`，会拉长观测到的起发时间。

所以：

- `formula_time` 更像“计算内核延迟模型”
- 日志上的层总耗时是“计算延迟 + 通信/资源竞争延迟”

### 20.9 合理性总结

当前公式体系从工程建模角度是自洽的：

1. 用 `MAC 工作量 / 并行度` 估主要计算成本（Conv/FC）。
2. 用尾项和常数项补偿边界开销。
3. 对近似计算单独引入稀疏与恢复相关开销项。
4. 通过 `temp_computation_time + computation_time` 统一映射到 SystemC 时间门控。

因此它不是“统计型后验模型”，而是“前馈型时序估算模型”。这也解释了你看到的“先有公式，再推进 cycles”。
