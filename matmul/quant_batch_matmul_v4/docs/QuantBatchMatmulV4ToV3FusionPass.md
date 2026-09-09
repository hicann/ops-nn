# QuantBatchMatmulV4ToV3FusionPass

## 融合模式

该融合将符合条件的`QuantBatchMatmulV4`节点替换为`QuantBatchMatmulV3`节点，按两种算子的输入定义重排连接，并保留输出的形状、数据类型、格式及相关属性，使量化矩阵乘计算由`QuantBatchMatmulV3`完成。如下图所示。

![QuantBatchMatmulV4转换为QuantBatchMatmulV3](../../../docs/zh/figures/QuantBatchMatmulV4ToV3FusionPass_1.png)

图中展示参与映射的输入，V3的可选输入未连接时保持缺省。`x2_scale`虽然在V4中为可选输入，但映射到V3的必选输入scale，因此可转换的输入图需要提供合法的`x2_scale`。输入映射如下，索引为算子IR定义中的索引。

| QuantBatchMatmulV4输入 | V4索引 | QuantBatchMatmulV3输入 | V3索引 |
| ---- | ---- | ---- | ---- |
| x1 | 0 | x1 | 0 |
| x2 | 1 | x2 | 1 |
| bias | 2 | bias | 4 |
| `x1_scale` | 3 | `pertoken_scale` | 5 |
| `x2_scale` | 4 | scale | 2 |
| `x2_offset` | 7 | offset | 3 |

转换时，各映射输入及输出`y`的形状、数据类型和格式保持不变，不增加数据类型转换等节点。因此，映射输入的数据类型必须同时满足V3对应输入的要求。属性处理如下。

| 属性 | 转换规则 |
| ---- | ---- |
| dtype | 传递V4的属性值 |
| `transpose_x1` | 传递V4的属性值，不取反 |
| `transpose_x2` | 传递V4的属性值，不取反 |
| `group_size` | V4值为-1时改为V3默认值0，其他值保持不变 |

V4特有的`compute_type`属性不传递给V3。

## 使用约束

- 该规则处理图中的`QuantBatchMatmulV4`节点。
- 可选输入的匹配结构包括全部提供，以及按算子定义的输入顺序从`bias`开始连续缺省的情况；不能理解为支持任意位置的可选输入组合。转换时，仅为能够取得形状、数据类型等信息的映射输入建立连接。
- 输入图必须提供`x1`、`x2`及输出`y`的形状、数据类型等信息，信息缺失时不执行转换。
- 输入图应属于QuantBatchMatmulV3能够承接的量化场景，并满足其数据类型、shape和格式约束，参见[QuantBatchMatmulV3算子说明](../../quant_batch_matmul_v3/README.md)。
- V4的`y_scale`、`x1_offset`、`y_offset`和`x2_table`不会连接到替换后的V3。如果计算依赖这些输入，就不属于本规则可等价转换的范围；不能依靠本规则自动拦截所有此类输入组合。

以下两类场景保留QuantBatchMatmulV4，不执行转换。同一行中的条件需同时满足。

| 保留V4的场景 | 数据类型条件 | shape条件 |
| ---- | ---- | ---- |
| V4原生支持场景 | x1为`FLOAT8_E4M3FN`，x2为`FLOAT4_E2M1`或FLOAT32，y为BF16或FLOAT16 | 满足左列数据类型条件即保留V4，无附加形状条件 |
| A8W8 G-B量化识别场景 | x1、x2均为INT8，`x1_scale`、`x2_scale`均存在且为FLOAT32，y为BF16 | x1与`x1_scale`的原始shape维数相同，且x2与`x2_scale`的原始shape维数相同 |

上表A8W8 G-B场景比较的是维数，不要求各维长度相等，也不以`group_size`属性值区分。除上表条件外，本规则不额外要求输入必须为静态形状，也不额外限定输入格式；可用形状和格式仍须同时满足V4与V3的算子约束。

## 支持的型号

产品支持范围需同时满足[QuantBatchMatmulV4算子说明](../README.md)和[QuantBatchMatmulV3算子说明](../../quant_batch_matmul_v3/README.md)。该pass自身没有按芯片型号进行分支判断，是否转换由上述输入条件决定。

<!-- npu="910b" id1 -->
Atlas A2训练系列产品/Atlas A2推理系列产品
<!-- end id1 -->

<!-- npu="A3" id2 -->
Atlas A3训练系列产品/Atlas A3推理系列产品
<!-- end id2 -->

<!-- npu="950" id3 -->
Ascend 950PR/Ascend 950DT
<!-- end id3 -->
