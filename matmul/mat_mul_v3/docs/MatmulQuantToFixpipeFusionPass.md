# MatmulQuantToFixpipeFusionPass

## 融合模式

该融合规则利用矩阵乘结果写出时的Fixpipe后处理能力，将矩阵乘及其后的反量化或重新量化合并为一个算子，减少独立的后处理节点。矩阵乘节点的类型为`MatMul`、`MatMulV2`、`BatchMatMul`或`BatchMatMulV2`。下文的“矩阵乘节点”均指这四种类型。

本规则识别融合前计算图（下文称“源图”）中的以下输入组合，并按输入类型确定融合目标。“仅权重量化（weight-only）”指第一个数据输入为浮点类型、第二个数据输入为INT8权重的场景。

| 矩阵乘节点的第一个输入 | 矩阵乘节点的第二个输入 | 融合后的算子类型 |
| ---- | ---- | ---- |
| INT8 | INT8 | `QuantBatchMatmulV3` |
| FLOAT16 | INT8 | `WeightQuantBatchMatmulV2`，须满足仅权重量化场景的额外平台要求 |
| FLOAT32 | INT8 | 可识别此类源图，但当前`WeightQuantBatchMatmulV2`不支持FLOAT32输入，不能将其视为设备可执行的组合 |

融合规则的识别范围和目标算子的执行规格需要同时满足。当前`WeightQuantBatchMatmulV2`的第一个输入`x`只支持FLOAT16、BF16；其中BF16不在本融合规则的源图识别范围内。具体组合还须满足[数据类型与格式](#数据类型与格式)及[平台能力要求](#平台能力要求)。

以下三图均展示INT8×INT8输入、INT32矩阵乘结果的场景。图中符号的含义如下：

- `x1`、`x2`：矩阵乘的两个数据输入；`bias`：可选偏置，`optional`表示可以不提供该输入；`y`：图中融合前后对应的输出张量。
- `scale`：缩放参数张量的图示名称。在反量化图中对应`AscendDequant.deq_scale`输入，在重新量化图中对应`AscendRequant.req_scale`输入。`UINT64`表示该张量的数据按算子要求编码。
- `Const(scale)`：以常量节点提供的缩放参数张量；`scale_fused`：三节点融合后生成的缩放参数常量的图示名称，不是新增的算子参数。
- 第三图的`s`、`o`分别表示`AscendQuant.scale`、`AscendQuant.offset`两个标量属性值，与上述`scale`张量是不同的对象。
- 框中的数据类型表示对应张量或节点输出的数据类型。融合后`QuantBatchMatmulV3`的输入索引从0开始：0、1为数据输入，2为`scale`，4为`bias`。

### 矩阵乘与AscendDequant融合

矩阵乘结果直接输入`AscendDequant`时，将两个节点融合为表中对应的矩阵乘算子。`AscendDequant.deq_scale`张量连接到融合后的缩放参数输入，融合输出为FLOAT16。

![矩阵乘与AscendDequant融合，缩放参数和可选bias连接到融合节点](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_1.png)

该两节点模式不要求`deq_scale`为编译期常量。`deq_scale`由运行期节点生成时，保留该生成节点及其输入，将生成的张量连接到融合后的矩阵乘算子。

### 矩阵乘与AscendRequant融合

矩阵乘结果直接输入`AscendRequant`时，将两个节点融合为表中对应的矩阵乘算子。`AscendRequant.req_scale`张量连接到融合后的缩放参数输入，输出为INT8。

`req_scale`可以由运行期节点生成，该生成节点及其输入在融合后保留。

![矩阵乘与AscendRequant融合，输出为INT8](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_2.png)

### 矩阵乘、AscendDequant与AscendQuant融合

矩阵乘后依次连接`AscendDequant`和`AscendQuant`，且满足[使用约束](#使用约束)以及下述常量和输出连接要求时，将三个节点融合为表中对应的矩阵乘算子，输出为INT8。

三节点融合要求在编译期能够取得`AscendDequant.deq_scale`的常量值，以及`AscendQuant.scale`、`AscendQuant.offset`的属性值。编译期将反量化缩放因子与`AscendQuant.scale`相乘，并将乘积和取整后的`AscendQuant.offset`编码为一维UINT64常量张量，连接到融合后的缩放参数输入。偏移属性按最接近的整数取整，恰好位于两个整数中间时向远离零的方向取整，例如0.5取1、-0.5取-1。图中的`scale_fused`表示这个常量张量，用户无需额外提供该常量张量。

图中的`offset=o`是`AscendQuant`的属性，不是`QuantBatchMatmulV3.offset`输入。三节点融合将该偏移信息编码在新生成的缩放参数中，不单独连接`offset`输入，也不为参数合并新增运行期乘法或加法节点。

![三个节点融合，合并缩放和偏移参数并生成UINT64常量](../../../docs/zh/figures/matmul_quant_to_fixpipe_fusion_pass_3.png)

`AscendDequant`输出只连接到所匹配的`AscendQuant`时，才具备三节点融合的输出连接条件。平台不支持该模式、编译期无法取得所需常量，或`AscendDequant`输出还被其他节点使用时，规则会继续匹配两节点模式。两节点融合成功后，只合并矩阵乘与`AscendDequant`，保留`AscendQuant`及其属性；两节点匹配也未通过时，这条链路不发生融合。

### 输入与输出的对应关系

融合后的算子沿用矩阵乘的数据输入和可选偏置，缩放参数的对应关系如下。

| 融合前的输入 | QuantBatchMatmulV3输入 | WeightQuantBatchMatmulV2输入 | 对应关系 |
| ---- | ---- | ---- | ---- |
| 矩阵乘的第一个数据输入 | `x1` | `x` | 使用同一个输入张量 |
| 矩阵乘的第二个数据输入 | `x2` | `weight` | 使用同一个输入张量 |
| `AscendDequant.deq_scale`或`AscendRequant.req_scale` | `scale` | `antiquant_scale` | 两节点模式使用融合前的张量；三节点模式使用合并后的常量张量 |
| 矩阵乘的可选`bias` | `bias` | `bias` | 提供时使用同一个偏置张量，未提供时保持未连接 |

本规则不连接`QuantBatchMatmulV3`的`offset`、`pertoken_scale`，也不连接`WeightQuantBatchMatmulV2`的`antiquant_offset`、`quant_scale`和`quant_offset`。这些参数仍是目标算子的可选输入，但不会通过本规则新增到融合后的节点上。

两个数据输入的转置属性按下表对应。融合保留属性值，不新增转置节点，也不将属性取反。

| 矩阵乘节点类型 | 融合前的两路转置属性 | QuantBatchMatmulV3的对应属性 | WeightQuantBatchMatmulV2的对应属性 |
| ---- | ---- | ---- | ---- |
| `MatMul`、`MatMulV2` | `transpose_x1`、`transpose_x2` | `transpose_x1`、`transpose_x2` | `transpose_x`、`transpose_weight` |
| `BatchMatMul`、`BatchMatMulV2` | `adj_x1`、`adj_x2` | `transpose_x1`、`transpose_x2` | `transpose_x`、`transpose_weight` |

融合后的输出形状与融合前的矩阵乘结果形状一致，数据格式为ND。矩阵乘与`AscendDequant`的两节点融合固定输出FLOAT16；矩阵乘与`AscendRequant`的两节点融合、三节点融合均输出INT8。下游节点改为使用该输出张量。

## 使用约束

### 数据类型与格式

| 检查项 | 要求 |
| ---- | ---- |
| 矩阵乘节点的输入 | 数据类型符合[融合模式](#融合模式)中的输入识别范围；仅包含两个数据输入及可选的`bias`，不处理带其他输入的矩阵乘节点 |
| INT8×INT8输入的数据格式 | 第一个输入为ND、NCHW、NHWC、HWCN、CHWN、NDHWC、NCDHW、DHWCN或DHWNC；第二个输入支持这些格式及FRACTAL_NZ |
| 仅权重量化输入的数据格式 | 两个数据输入均为ND、NCHW、NHWC、HWCN、CHWN、NDHWC、NCDHW、DHWCN或DHWNC |
| `AscendDequant`两节点融合的输出 | 融合后固定为FLOAT16；即使融合前`AscendDequant`输出为FLOAT32，融合后也不会保留该输出类型 |
| `AscendRequant`的输出 | 按该算子的输入输出规格，输出类型为INT8 |
| 三节点融合 | `AscendQuant`输出为INT8；`deq_scale`使用UINT64编码张量；编译期能够取得该常量值和`AscendQuant.scale`、`AscendQuant.offset`属性值；生成的缩放参数须符合所用平台对`scale`或`antiquant_scale`的格式要求 |

输出类型需要与下游节点的输入要求一致。尤其是`AscendDequant`的两节点融合，不能假定融合后仍提供FLOAT32结果。

上述格式和类型是融合规则分别检查的条件，不能任意交叉组合。图中数据类型、形状、缩放参数和偏置的组合还须符合[QuantBatchMatmulV3](../../quant_batch_matmul_v3/README.md)或[WeightQuantBatchMatmulV2](../../weight_quant_batch_matmul_v2/README.md)的算子约束。

输入形状中存在未知维度时，不会仅因维度值未知而跳过本规则。动态维度和批次维度广播能否执行，仍须满足目标算子及所用安装包的支持范围。

### 后处理属性

本文的融合场景采用以下后处理属性设置。

| 被合并的节点 | 属性设置 |
| ---- | ---- |
| `AscendDequant` | `relu_flag=false`、`sqrt_mode=false`；输出类型按前文说明处理 |
| `AscendRequant` | `relu_flag=false` |
| `AscendQuant`，仅三节点融合 | `sqrt_mode=false`、`round_mode="Round"`，输出类型为INT8 |

融合只处理前述缩放参数及`AscendQuant.scale`、`AscendQuant.offset`，不会为上述非默认属性额外保留ReLU、平方缩放或其他舍入操作。这些属性设置是用户构图时的使用条件；不能假定设置非默认值后，融合就一定会自动跳过。

### 输出连接与共享场景

- 矩阵乘的输出张量只能接入一个下游输入，且该输入属于所匹配的`AscendDequant`或`AscendRequant`。例如，矩阵乘结果同时作为图输出并输入`AscendDequant`时，不融合这两个节点，以保留矩阵乘结果。
- 矩阵乘与`AscendDequant`的两节点模式允许`AscendDequant`输出被多个节点使用。融合后，各下游节点继续取得FLOAT16反量化结果。
- 多个`AscendQuant`共享同一个`AscendDequant`输出，或`AscendDequant`输出同时被`AscendQuant`和其他节点使用时，不进行三节点融合；满足两节点约束时，将矩阵乘与`AscendDequant`融合，并保留所有`AscendQuant`及其他下游节点。

共享矩阵乘结果时不执行本次融合；共享`AscendDequant`输出时，可以在两节点约束满足后融合矩阵乘与`AscendDequant`。

### 平台能力要求

- 平台须具有`Intrinsic_fix_pipe_l0c2out`能力，且不具有`Intrinsic_data_move_l0c2ub`能力。
- 若平台具有`Intrinsic_fix_pipe_l0c2ub`能力，还须具有支持BF16的`Intrinsic_data_move_l12bt`能力。
- 三节点模式和仅权重量化场景另要求平台能力表中的`Intrinsic_mmad`包含`s8s4`。`s8s4`在此处是平台能力标识，不是用户需要设置的算子参数；本规则的源输入类型仍以上文识别范围为准。INT8两节点模式不受这项额外限制。

未发生融合时，节点仍留在计算图中，需要按各独立算子的规格检查其输入类型、数据格式和形状。例如，独立`AscendDequant`的`x`输入要求INT32；如果保留的矩阵乘结果为FLOAT16，不能直接作为该输入执行。

## 支持的型号

以下为INT8两节点基本模式的适用产品。三节点模式和仅权重量化场景还须满足前述额外平台能力要求，不能仅凭产品名称判断所有模式均可融合。

<!-- npu="910b" id1 -->
Atlas A2系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR&950DT系列产品
<!-- end id3 -->
