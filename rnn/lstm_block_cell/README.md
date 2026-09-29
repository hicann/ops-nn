# LstmBlockCell

LSTM（长短期记忆网络）**单步前向全融合算子**，对标TensorFlow的`LstmBlockCell`（`tf.raw_ops.LSTMBlockCell`单步语义，TensorFlow通路）：给定当前步输入`x`、上一步cell状态`cs_prev`与隐状态`h_prev`、合并投影权重`w`与偏置`b`，一次调用完成投影GEMM、ICFO四门激活与状态更新，输出`i`/`cs`/`f`/`o`/`ci`/`co`/`h`七个`[B, H]`张量。

本算子**GEIR-only交付**：通过GE图模式以`op::LstmBlockCell`构图调用，或经TensorFlow插件把TF侧同名算子`LstmBlockCell`自动映射到本算子。

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：LSTM单步cell前向计算（B=batch、I=inputSize、H=hiddenSize）。融合完成投影GEMM与ICFO门控激活；`i`/`f`/`o`/`ci`为门控与候选中间量，`cs`/`co`/`h`为新cell状态、新输出状态与新隐状态。
- **计算公式**（门序`i|ci|f|o`，σ为sigmoid，⊙为逐元素乘，`[x, h_prev]`表示沿axis 1拼接）：

$$
gates = [x,\ h_{prev}] \cdot w + b \in \mathbb{R}^{B \times 4H}
$$

$$
i = \sigma\left(gates[:,\ 0{:}H] + p_i\right), \qquad
ci = \tanh\left(gates[:,\ H{:}2H]\right)
$$

$$
f = \sigma\left(gates[:,\ 2H{:}3H] + f_b + p_f\right)
$$

$$
cs = \mathrm{clip}\left(f \odot cs_{prev} + i \odot ci,\ \pm c_{clip}\right)
$$

$$
co = \tanh(cs), \qquad
o = \sigma\left(gates[:,\ 3H{:}4H] + p_o\right), \qquad
h = o \odot co
$$

  其中`forget_bias`（$f_b$，默认1.0）在f门sigmoid之前加上；`cell_clip`（$c_{clip}$，默认3.0）大于0时对cs裁剪到$[-c_{clip}, c_{clip}]$，等于0不裁剪；$p_i/p_f/p_o$为peephole项，`use_peephole`（默认false）为true时分别取$cs_{prev} \odot wci$、$cs_{prev} \odot wcf$、$co \odot wco$，否则为0。
- **语义要点**：
  - **单步cell**：对标`tf.raw_ops.LSTMBlockCell`（8输入、无seq_len_max），不是`BlockLSTMV2`的多步循环形态。
  - **门序i|ci|f|o**：`w`列序前H列为i门、次H列为ci、再次H列为f门、末H列为o门（ICFO，非IFCO）。
  - **peephole权重恒传入**：`use_peephole`为false时`wci/wcf/wco`不参与计算，但8个输入槽位固定（TF侧约定零填充而非缺席）。
  - **特殊值语义**：±Inf经sigmoid/tanh饱和（σ(±Inf)∈{0,1}、tanh(±Inf)=±1）；门控激活与cell_clip钳位以Compares+Select实现（比较对NaN恒假、原值走else支路，NaN逐元素传播，与TF的std::min/std::max操作数序相同）。
  - **实现形态**：MIX 1 AIC : 2 AIV；M（batch）/N（hidden）/K（I+H）全轴切分，任意B/I/H≥1的合法shape均可执行，超大批次自动多轮消化（无容量拒绝）。

## 参数说明

|参数名|输入/输出/属性|描述|数据类型|数据格式|
|-----|-----------|----|---------|------|
|x|输入|当前步输入，shape [B, I]|FLOAT/FLOAT16|ND|
|cs_prev|输入|上一步cell状态，shape [B, H]，dim0与x一致|FLOAT/FLOAT16|ND|
|h_prev|输入|上一步隐状态，shape [B, H]，dim0与x一致|FLOAT/FLOAT16|ND|
|w|输入|合并投影权重，shape [I+H, 4H]（行：x块+h块；列序i\|ci\|f\|o）|FLOAT/FLOAT16|ND|
|wci|输入|i门peephole权重，shape [H]|FLOAT/FLOAT16|ND|
|wcf|输入|f门peephole权重，shape [H]|FLOAT/FLOAT16|ND|
|wco|输入|o门peephole权重，shape [H]|FLOAT/FLOAT16|ND|
|b|输入|偏置，shape [4H]（列序i\|ci\|f\|o）|FLOAT/FLOAT16|ND|
|i|输出|输入门激活，shape [B, H]|FLOAT/FLOAT16|ND|
|cs|输出|新cell状态，shape [B, H]|FLOAT/FLOAT16|ND|
|f|输出|遗忘门激活，shape [B, H]|FLOAT/FLOAT16|ND|
|o|输出|输出门激活，shape [B, H]|FLOAT/FLOAT16|ND|
|ci|输出|候选状态，shape [B, H]|FLOAT/FLOAT16|ND|
|co|输出|新输出状态（tanh(cs)），shape [B, H]|FLOAT/FLOAT16|ND|
|h|输出|新隐状态，shape [B, H]|FLOAT/FLOAT16|ND|
|forget_bias|属性|遗忘门偏置加成，f门sigmoid之前加上，默认1.0|浮点|-|
|cell_clip|属性|cell状态裁剪幅度，大于0时cs裁剪到[-cell_clip, cell_clip]，默认3.0|浮点|-|
|use_peephole|属性|是否启用peephole连接，默认false|布尔|-|

## 约束说明

- **数据类型**：不支持float32/float16以外的数据类型，8个输入与7个输出的dtype一致性经图级推导与tiling双层校验，违规返回`GRAPH_FAILED`。
- **数据格式**：仅ND。
- **shape约束**：x/cs_prev/h_prev/w的维度必须为2、wci/wcf/wco/b的维度必须为1（tiling阶段校验，违规返回`GRAPH_FAILED`并输出报错日志）；B/I/H取值[1, 2^24]；单张量总元素数须< 2^32（kernel以uint32元素偏移寻址）；shape契约（h_prev/cs_prev为[B, H]、w为[I+H, 4H]、b为[4H]、wci/wcf/wco为[H]）同样在tiling阶段校验。
- **空输入**：B/I/H均≥1，零值维度由tiling阶段拒绝。
- **属性**：3个属性任意取值均支持，无拒绝路径（`cell_clip=0`表示不裁剪）。
- **动态shape**：支持-1未知维与-2未知rank声明（执行期以实际供数shape校验并定形）。
- **实现形态**：fp32/fp16双档tilingKey（0/1）静态shape统一kernel；GM搬运无32B对齐要求，任意合法shape可执行。

## 调用说明

Ascend 950PR&950DT系列产品：

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE图模式 | [test_geir_lstm_block_cell](examples/arch35/test_geir_lstm_block_cell.cpp) | 通过[算子IR](op_graph/lstm_block_cell_proto.h)以`op::LstmBlockCell`构图，经`ge::Session`编译执行 |
| GE图模式（动态shape） | [test_geir_lstm_block_cell_dynamic](examples/arch35/test_geir_lstm_block_cell_dynamic.cpp) | -1未知维/-2未知rank声明，同一图多次`RunGraph`换shape执行 |
| TensorFlow原生接口 | [lstm_block_cell_tf_plugin](framework/lstm_block_cell_tf_plugin.cpp) | TF侧同名算子`LstmBlockCell`经TF插件自动映射为本算子 |

本算子仅提供上述两条GE图模式通路，不提供单算子C API直调。

### GE图模式调用

前置条件：构建并安装算子包后，GE经`$ASCEND_OPP_PATH/vendors/<vendor_name>_nn`解析本算子的原型与tiling库。

调用链：`op::LstmBlockCell`构图（8个`op::Data`输入+7个[B, H]输出描述）→`ge::Session::AddGraph`（编译：原型→InferShape→tiling→kernel binary→任务生成）→`RunGraph`（下发执行）→输出回收（i/cs/f/o/ci/co/h各[B, H]）。

```cpp
#include "graph.h"
#include "ge_api.h"    // ge::GEInitialize / ge::Session
#include "lstm_block_cell_proto.h"  // 算子原型（op::LstmBlockCell）

// 1) 构图：8输入 → LstmBlockCell → 7输出
auto lstmCellOp = op::LstmBlockCell("lstm_block_cell");
lstmCellOp.set_input_x(dataX);           // x      [B, I]
lstmCellOp.set_input_cs_prev(dataCsP);   // cs_prev [B, H]
lstmCellOp.set_input_h_prev(dataHP);     // h_prev  [B, H]
lstmCellOp.set_input_w(dataW);           // w       [I+H, 4H]
lstmCellOp.set_input_wci(dataWci);       // wci     [H]
lstmCellOp.set_input_wcf(dataWcf);       // wcf     [H]
lstmCellOp.set_input_wco(dataWco);       // wco     [H]
lstmCellOp.set_input_b(dataB);           // b       [4H]

ge::TensorDesc outDesc(ge::Shape({B, H}), ge::FORMAT_ND, ge::DT_FLOAT);
lstmCellOp.update_output_desc_i(outDesc);  // cs/f/o/ci/co/h同形 [B, H]

ge::Graph graph("lstm_block_cell");
graph.SetInputs({dataX, dataCsP, dataHP, dataW, dataWci, dataWcf, dataWco, dataB})
     .SetOutputs({lstmCellOp});

// 2) 会话编译 + 执行 + 输出回收（feeds为8个host输入张量，GE负责H2D/D2H）
std::map<ge::AscendString, ge::AscendString> options = {
    {"ge.exec.deviceId", "0"}, {"ge.graphRunMode", "1"}};
ge::GEInitialize(options);
ge::Session session(std::map<ge::AscendString, ge::AscendString>{});
session.AddGraph(graphId, graph, options);  // 编译（含InferShape/tiling/kernel binary）
session.RunGraph(graphId, feeds, outputs);  // 执行；outputs依次为i/cs/f/o/ci/co/h
```

完整可运行样例（含确定性输入构造与独立双精度golden数值比对，容差rtol=1e-5/atol=1e-6）见[test_geir_lstm_block_cell.cpp](examples/arch35/test_geir_lstm_block_cell.cpp)。

### TensorFlow接入

本算子内置TF插件（构建产物`libcust_tf_parsers.so`）。TF侧需自注册同名算子`LstmBlockCell`（8入7出3属性，槽位与本算子OpDef逐字一致，经`tf.load_op_library`加载），插件按位置自动映射输入、按名字匹配属性：

```cpp
// framework/lstm_block_cell_tf_plugin.cpp
REGISTER_CUSTOM_OP("LstmBlockCell")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("LstmBlockCell")
    .ParseParamsByOperatorFn(ParseLstmBlockCellByOpFn);
```

`ParseLstmBlockCellByOpFn`在`AutoMappingByOpFn`基础上把源节点输入dtype（TF属性`T`）前向设置到全部8输入与7输出，避免ATC为全fp32图选择fp16 kernel并用trans_Cast桥接全图。在线执行依赖TF-NPU适配器`npu_device`，且算子须在`tf.function`内调用：

```python
import npu_device
import tensorflow as tf

lstm_block_cell_module = tf.load_op_library("libcustom_ops.so")  # TF侧 LstmBlockCell 注册库

with npu_device.open().as_default():
    @tf.function(autograph=False)
    def call_lstm(x, cs_prev, h_prev, w, wci, wcf, wco, b):
        return lstm_block_cell_module.lstm_block_cell(
            x=x, cs_prev=cs_prev, h_prev=h_prev, w=w,
            wci=wci, wcf=wcf, wco=wco, b=b,
            forget_bias=1.0, cell_clip=3.0, use_peephole=False)

    i, cs, f, o, ci, co, h = call_lstm(x, cs_prev, h_prev, w, wci, wcf, wco, b)
```
