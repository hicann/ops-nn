# QuantBatchMatmulV3TransposeFusionPass

## 融合模式

该融合规则将`QuantBatchMatmulV3`的`x1`或`x2`输入支路中的显式转置合入矩阵乘计算，也可以同时处理两条支路。可识别的转置节点类型为`Transpose`和`TransposeD`。融合`x1`支路时，将`QuantBatchMatmulV3`节点的`transpose_x1`属性取反；融合`x2`支路时，将该节点的`transpose_x2`属性取反。取反是指将`false`改为`true`，或将`true`改为`false`。

本文用“旁路”表示以下连接调整：将转换节点的输入直接连接到该转换节点的后继节点，使当前支路不再经过该转换节点。旁路后，转换节点仍有其他数据使用者时保留，否则删除。后继节点可以是`QuantBatchMatmulV3`，也可以是保留的`Bitcast`。

以下图示均以满足[输入与转置要求](#输入与转置要求)及[维度与平台限制](#维度与平台限制)为前提。图中变量说明如下：

- **输入对应关系**：分别展示`x1`、`x2`及两路`scale`。`x1_scale`表示输入`pertoken_scale`（IR索引5），`x2_scale`表示输入`scale`（IR索引2）；这两个名称是图示别名，不是新增算子参数，也不是算子属性。
- **符号定义**：`t1`、`t2`分别表示融合前`QuantBatchMatmulV3`的`transpose_x1`、`transpose_x2`属性值；`!`表示逻辑取反。
- **源张量**：融合前后的同名源张量表示同一个张量。
- **省略内容**：图中省略`bias`、`offset`、指定转置维度顺序的输入及指定重塑后形状的输入，只展示与融合相关的数据连接。

### 仅融合x1侧转置

`x1`支路和`x1_scale`支路中各存在一个转置节点时，分别旁路这两个转置节点，将它们的输入直接连接到`QuantBatchMatmulV3`的`x1`和`pertoken_scale`输入端口。同时，将`QuantBatchMatmulV3`节点的`transpose_x1`属性取反。`x2`和`scale`输入的连接，以及该节点的`transpose_x2`属性保持不变。

![仅融合x1及x1_scale转置，四路输入独立连接](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_1.png)

### 仅融合x2侧转置

`x2`支路和`x2_scale`支路中各存在一个转置节点时，分别旁路这两个转置节点，将它们的输入直接连接到`QuantBatchMatmulV3`的`x2`和`scale`输入端口。同时，将`QuantBatchMatmulV3`节点的`transpose_x2`属性取反。`x1`和`pertoken_scale`输入的连接，以及该节点的`transpose_x1`属性保持不变。

![仅融合x2及x2_scale转置，四路输入独立连接](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_2.png)

### 同时融合两侧转置

`x1`、`x2`、`x1_scale`和`x2_scale`四条支路中各存在一个转置节点，且`x1`、`x2`两侧均满足本节开头所列使用约束时，分别旁路四个转置节点。四个转置节点的输入分别连接到`QuantBatchMatmulV3`的`x1`、`x2`、`pertoken_scale`和`scale`输入端口；`QuantBatchMatmulV3`节点的`transpose_x1`和`transpose_x2`属性均取反。仅匹配图中结构不足以触发融合，还需满足数据类型、维度和平台能力等约束。

![两路数据和两路scale的四个转置分别融合](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_3.png)

### 普通scale的Reshape融合

`x1`、`x2`支路中各存在一个转置节点，两路`scale`支路中各存在一个等价于所需转置的`Reshape`时，分别旁路两个转置节点和两个`Reshape`。四条支路连接到各自对应的`QuantBatchMatmulV3`输入端口，`QuantBatchMatmulV3`节点的`transpose_x1`和`transpose_x2`属性均取反。静态普通`scale`的识别要求见[scale的Reshape要求](#scale的reshape要求)。图中展示两侧均可融合的情形，也可仅处理满足约束的一侧。

![两路普通scale的Reshape分别随对应数据转置融合](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_4.png)

### MX scale的Reshape融合

以`x2`侧为例：`x2`输入前为`Transpose`或`TransposeD`，`x2_scale`为`FLOAT8_E8M0`，其`Reshape`将(1, N, 2)变为(N, 1, 2)。融合时旁路`x2`支路的转置节点和`x2_scale`支路的`Reshape`，将二者的输入分别连接到`QuantBatchMatmulV3`的`x2`和`scale`输入端口，并将`QuantBatchMatmulV3`节点的`transpose_x2`属性取反；`x1`及`x1_scale`保持原连接。`x1`侧满足相应条件时采用相同的处理逻辑。

![MX场景分别展示x1、x2及两个scale和三维Reshape](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_5.png)

### 保留Bitcast的转置融合

`Bitcast`按目标数据类型重新解释数据的位表示。四路输入各自按“转置节点 → `Bitcast` → `QuantBatchMatmulV3`”连接时，融合会将四个转置节点的输入分别直接连接到后面的`Bitcast`，保留四个`Bitcast`及其到`QuantBatchMatmulV3`的连接，并将`QuantBatchMatmulV3`节点的`transpose_x1`和`transpose_x2`属性取反。融合后，`Bitcast`输出的数据类型保持不变。

![四路输入的转置分别融合并各自保留Bitcast](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_6.png)

各路输入不要求同时存在`Bitcast`。转置节点可以直接连接矩阵乘，也可以通过一个`Bitcast`连接矩阵乘；不支持越过连续多个`Bitcast`进行融合。对于`scale`支路，`Bitcast`前的节点也可以是满足下文要求的`Reshape`。平台限制和动态`Reshape`的适用边界见使用约束。

### 动态scale的Reshape融合

`x1`或`x2`存在未知维度时，按动态场景处理。以`x2`侧为例，融合时旁路`x2`支路的转置节点和`x2_scale`支路的`Reshape`，将二者的输入分别连接到`QuantBatchMatmulV3`的`x2`和`scale`输入端口，并将`QuantBatchMatmulV3`节点的`transpose_x2`属性取反；`x1`和`x1_scale`保持原连接。动态场景不执行静态`Reshape`的维度条件检查，但输入图仍须保证该`Reshape`与所需的`scale`转置等价。

![动态场景中四路输入独立展示，仅处理x2侧转置与Reshape](../../../docs/zh/figures/QuantBatchMatmulV3TransposeFusionPass_7.png)

各图中的`scale`转换节点是所示场景的具体结构，不要求每一路`scale`都存在转换节点。只有相应数据输入的`Transpose`或`TransposeD`被融合时，才处理该路`scale`；`pertoken_scale`未连接时不处理该输入。被旁路的节点仍有其他数据使用者时保留，否则从图中删除。

## 使用约束

### 输入与转置要求

- `x1`和`x2`的原始`shape`维数均不得小于2，至少一路数据输入存在`Transpose`或`TransposeD`，否则不触发融合。数据输入前仅有`Reshape`不能触发融合。
- `x1`、`x2`的数据类型均支持`INT8`、`INT4`、`FLOAT8_E4M3FN`、`FLOAT8_E5M2`、`HIFLOAT8`、`FLOAT4_E2M1`；输出数据类型支持`INT8`、`FLOAT16`、`BF16`、`INT32`、`FLOAT32`。实际组合还需满足目标产品的算子约束。
- 支持动态`shape`；`x1`或`x2`的`shape`存在未知维度时，按动态场景处理。
- 数据输入的转置必须只交换最后两维，其他维度的顺序保持不变。例如，形状为`(B, M, K)`的输入转置为`(B, K, M)`。交换批次维度等其他转置方式不属于本规则的适用范围。对应的`scale`转换也须保持量化系数与数据的对应关系，具体要求见下一节。
- `pertoken_scale`是可选输入，未连接时不处理该路`scale`。

### Bitcast场景的限制

部分平台的矩阵乘指令支持INT8与INT4混合输入。在这类平台上，只要四路输入中的一路符合下表结构，本规则就保留整个`QuantBatchMatmulV3`节点及其所有输入支路，不执行本次转置融合。即使当前节点没有采用INT8与INT4混合输入，上述限制也适用。

| 输入支路 | 触发上述限制的连接结构 |
| ---- | ---- |
| `x1`、`x2`或两路`scale`中的任一路 | `Transpose`或`TransposeD` → `Bitcast` → `QuantBatchMatmulV3` |
| 两路`scale`中的任一路 | 满足下一节融合要求的`Reshape` → `Bitcast` → `QuantBatchMatmulV3` |

仅有`Bitcast`、其前方没有上述转换节点时，不会因本条限制跳过融合。

动态`Reshape`与`Bitcast`组合使用时还存在识别限制：如果只有`scale`支路包含“动态`Reshape` → `Bitcast` → 矩阵乘”，且其他支路都没有可识别的上述带`Bitcast`结构，该动态`Reshape`不保证被融合。因此，本文的动态`Reshape`示例仅展示未经过`Bitcast`的连接。

### scale的Reshape要求

`scale`输入前的`Reshape`需满足下表条件。静态场景还要求输入图提供`Reshape`输入、输出的形状和数据类型信息，且对应`QuantBatchMatmulV3`输入至少为2维。

| 场景 | 识别条件 |
| ---- | ---- |
| 条件A：静态通用条件 | 对应`QuantBatchMatmulV3`输入`shape`的最后两维至少一维为1，此判断不限制`scale`的数据类型 |
| 条件B：静态MX条件 | 数据类型为`FLOAT8_E8M0`，`Reshape`输入、输出均为3维，前两维互换且其中至少一维为1，例如(1, N, 2)变为(N, 1, 2) |
| 动态`shape` | 不要求满足上述静态形状条件，但`Reshape`仍须与所需的`scale`转置等价 |

静态场景满足上表条件A或条件B中的任意一项即可。条件B指`FLOAT8_E8M0`三维`scale`的前两维交换条件；不满足条件B时，仍可通过条件A识别。动态场景按上表“动态`shape`”行处理。图中MX示例的末维2来自该示例的量化布局，并非条件B额外要求的固定值。满足表中的形状条件后，仍须保证`Reshape`与所需的`scale`转置等价，不能改变量化系数与数据的对应关系。本规则不单独删除用于计算动态形状的`Shape`、`Gather`、`Pack`等节点。

### 维度与平台限制

`QuantBatchMatmulV3TransposeFusionPass`是普通转置融合规则，`QuantBatchMatmulV3TransposeLimitFusionPass`是针对特定维度场景的补充规则。两者分别注册，均将符合条件的显式转置合入矩阵乘计算，通过调整输入连接并取反`QuantBatchMatmulV3`的相应转置属性实现融合，区别在于适用的产品和维度条件。Limit规则用于在普通规则被关闭时，仍为符合其条件的支路提供转置融合处理。

令`outer`为融合前`QuantBatchMatmulV3`对应数据输入原始`shape`的倒数第二维，`inner`为最后一维。两个规则均分别检查`x1`、`x2`，可只融合满足条件的一侧。本文所列产品的处理条件如下，实际融合还须满足前文的输入、转置节点及`scale`等使用约束。

| 产品 | 普通规则 | Limit补充规则 |
| ---- | ---- | ---- |
| Ascend 950PR/Ascend 950DT | `x1`、`x2`均不受本节65535维度阈值限制，满足其他使用约束时执行转置融合 | 跳过本补充规则，转置融合由普通规则处理 |
| Atlas A2训练系列产品/Atlas A2推理系列产品、Atlas A3训练系列产品/Atlas A3推理系列产品 | `x1`要求`outer` ≤ 65535；`x2`要求`outer` ≤ 65535或格式为`FRACTAL_NZ` | 分别检查`x1`、`x2`；仅处理满足0 < `outer` ≤ 65535且`inner` > 65535的支路 |

Limit规则的维度条件同样适用于`FRACTAL_NZ`格式，不因格式而放宽。参与判断的`outer`或`inner`为未知值`-1`时，该支路不满足Limit规则的维度条件。

### 版本要求

- 编译和运行时的图编译器版本号均须不小于90100000。
- 编译版本不满足要求时，不启用这两个转置规则；运行版本不满足要求时，两个规则均保持输入图不变。

## 支持的型号

产品的数据类型与格式支持范围参见[QuantBatchMatmulV3算子说明](../README.md)。本规则按平台指令能力判断融合条件，算子支持某产品并不表示该产品支持本规则的全部融合模式。

<!-- npu="910b" id1 -->
Atlas A2训练系列产品/Atlas A2推理系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3训练系列产品/Atlas A3推理系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR/Ascend 950DT
<!-- end id3 -->
