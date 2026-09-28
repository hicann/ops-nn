# QuantBatchMatmulInplaceAddTransposeFusionPass

## 融合模式

本融合规则将`QuantBatchMatmulInplaceAdd`的`x1`输入前的`Transpose`或`TransposeD`与目标算子融合，并取反`transpose_x1`。满足条件时，同时融合`x1_scale`前的转置或等价`Reshape`；中间的`Bitcast`保留。`x2`、`x2_scale`及`transpose_x2`保持不变。

图示约定：

- 输入为`x1`、`x2`、`x1_scale`、`x2_scale`和`yRef`；`yRef`为累加输入，对应算子输入`y`，输出`y`保持原地累加关系。
- `t1`、`t2`分别表示融合前的`transpose_x1`、`transpose_x2`，`!t1`表示逻辑取反。本规则仅在`t1=false`且`t2=false`时融合，融合后为`transpose_x1=true`、`transpose_x2=false`。
- `M`、`N`分别为结果矩阵的行数、列数，`K`为矩阵乘的归约维度。shape使用方括号表示；“单例维”表示长度为1的维度。
- 图中省略转置维度顺序及Reshape目标shape的输入。图示表达转换关系，实际输入及融合结果须符合算子规格。

### 仅融合x1侧转置

将`x1`前的`Transpose`或`TransposeD`融合到目标算子，取反`transpose_x1`。如果`x1_scale`前也存在满足条件的转置，则同时融合。`x2`和`x2_scale`保持原连接。

![仅融合x1侧转置](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_1.png)

### 仅x2侧存在转置时不融合

仅`x2`或`x2_scale`前存在转换节点时，不触发本规则，图结构保持不变。

![仅x2侧存在转置时不融合](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_2.png)

### 两侧存在转置时仅融合x1侧

两路数据前都有转置时，仅融合`x1`及满足条件的`x1_scale`转换。`x2`和`x2_scale`前的`Transpose`、`TransposeD`、`Reshape`、`Bitcast`及其连接保持不变。

![两侧存在转置时仅融合x1侧](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_3.png)

### MX量化中x1_scale的Reshape融合

静态shape下，数据类型为`FLOAT8_E8M0`的三维`x1_scale`，其`Reshape`输入、输出的前两维互换，且这两维中至少一维的长度为1时，可以视为等价转置。例如shape由`[1, M, 2]`变为`[M, 1, 2]`。融合`x1`转置时，可以同时融合该`Reshape`；`x2_scale`前的`Reshape`保留。

![MX量化中x1_scale的Reshape融合](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_4.png)

### 保留Bitcast的转置融合

对于“转置节点→`Bitcast`→目标算子”的结构，融合转置并保留`Bitcast`。`x1`和`x1_scale`可以分别包含一个`Bitcast`，不要求两路同时包含。`x2`和`x2_scale`支路保持不变。此模式的共享节点限制见使用约束。

![保留Bitcast的转置融合](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_5.png)

### 保留Bitcast的x1_scale Reshape融合

`x1_scale`支路可以为“`Reshape`→`Bitcast`→目标算子”。当`Reshape`满足等价转置条件，且带`Bitcast`的模式被识别时，融合该`Reshape`并保留`Bitcast`。图中展示静态MX量化场景；动态shape组合的限制见使用约束。

![保留Bitcast的x1_scale Reshape融合](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_6.png)

### 单元素scale保持原连接

以HIFLOAT8逐张量量化为例，`x1_scale`和`x2_scale`均为数据类型为FLOAT32、shape为`[1]`的一维张量，各包含一个量化系数。两路scale前没有转换节点时保持原连接，仅融合`x1`侧转置。这里的`[1]`不是零维张量的shape `[]`。

![单元素scale保持原连接](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_7.png)

### 动态shape下x1_scale的Reshape融合

当数据输入shape包含未知维度时，可以识别`x1_scale`前的`Reshape`，不要求满足静态shape的单例维条件。输入图须保证该`Reshape`与所需转置等价。图中不包含`Bitcast`；这描述的是融合模式，不代表算子支持任意未知维度输入。

![动态shape下x1_scale的Reshape融合](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_8.png)

### 其他满足单例维条件的Reshape

静态shape下，若`x1_scale`输入至少二维，且末两维中至少一维的长度为1，也可以识别其前方的`Reshape`，例如shape由`[8, 1]`变为`[1, 8]`。输入图须保证`Reshape`与转置等价，实际量化模式和参数组合仍须符合算子规格。

![其他满足单例维条件的Reshape](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_9.png)

### x1_scale没有转换节点

`x1_scale`直接连接目标算子、前方不存在可识别转换，或可选输入`x1_scale`未连接时，该支路保持不变，不因此阻止`x1`转置融合。`x2_scale`须保持有效输入连接。

![x1_scale没有转换节点](../../../docs/zh/figures/QuantBatchMatmulInplaceAddTransposeFusionPass_10.png)

## 使用约束

- 融合前须满足`transpose_x1=false`、`transpose_x2=false`，以保证融合后的转置属性满足后端要求；其他属性组合保持原图。
- `x1`前须存在可识别的`Transpose`或`TransposeD`，直接连接目标算子或经一个`Bitcast`连接。只有`Reshape`或只有`x2`侧转置不能触发本规则。
- 数据转置须与矩阵乘转置语义一致，scale转换须保持量化系数与数据的对应关系；不能将任意维度置换或任意Reshape视为等价转置。
- 含`Bitcast`的模式仅用于待融合转换节点不被其他计算支路共享的图；共享支路的数值正确性不在该模式的保证范围内。
- 带`Bitcast`的融合模式须通过目标平台的能力校验；不支持该模式的平台保持原图。此限制不影响不带`Bitcast`的模式。
- 每个目标节点根据自身`x1`、`x2`的shape独立识别动态shape，不受此前处理的节点或图影响。动态shape的“`Reshape`→`Bitcast`”组合仍须满足平台能力及其他融合条件。此处动态shape指含未知维度，与MX动态量化不是同一概念。
- 实际数据类型、量化模式、shape和转置属性须符合[QuantBatchMatmulInplaceAdd算子说明](../README.md)。该算子当前要求`transposeX1=true`、`transposeX2=false`；图示中的属性取反不扩展算子的支持范围。

## 支持的型号

<!-- npu="950" id1 -->
Ascend 950PR&950DT系列产品
<!-- end id1 -->
