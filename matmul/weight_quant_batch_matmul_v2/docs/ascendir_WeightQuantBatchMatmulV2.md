# WeightQuantBatchMatmulV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：完成一个仅权重量化的矩阵乘计算（权重反量化与矩阵乘融合计算），并支持对输出进行量化计算。支持MX、pergroup、perchannel、pertensor等[量化模式](../../../docs/zh/context/quant_mode_introduction.md)，不同量化模式对应的输入shape及约束参见[aclnnWeightQuantBatchMatmulV2](aclnnWeightQuantBatchMatmulV2.md)的约束说明。
- 计算公式：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  其中：
  - $x$：左矩阵，维度为$(M, K)$；当transpose_x=true时，$x$传入的维度为$(K, M)$，计算时逻辑转置为$(M, K)$。
  - $weight$：伪量化场景的量化权重右矩阵，维度为$(K, N)$；当transpose_weight=true时，$weight$传入的维度为$(N, K)$，计算时逻辑转置为$(K, N)$。
  - $ANTIQUANT(weight)$：权重反量化计算，公式为$ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale$。
  - $bias$：维度为$(N)$或$(1, N)$的偏置向量。
  - $y$：输出矩阵，维度为$(M, N)$。M为$x$的行数和$y$的行数，N为$weight$的列数和$y$的列数，K为$x$的列数和$weight$的行数。

  当需要对输出进行量化处理时，其量化公式为：

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

## Ascend IR定义

Ascend IR定义所在头文件路径：[weight_quant_batch_matmul_v2_proto.h](../op_graph/weight_quant_batch_matmul_v2_proto.h)

```cpp
REG_OP(WeightQuantBatchMatmulV2)
    .INPUT(x, TensorType({DT_FLOAT16, DT_BF16}))
    .INPUT(weight, TensorType({DT_INT8, DT_INT4, DT_INT32, DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1}))
    .INPUT(antiquant_scale, TensorType({DT_FLOAT16, DT_BF16, DT_UINT64, DT_INT64, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(antiquant_offset, TensorType({DT_FLOAT16, DT_BF16, DT_INT32}))
    .OPTIONAL_INPUT(quant_scale, TensorType({DT_FLOAT, DT_UINT64}))
    .OPTIONAL_INPUT(quant_offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8}))
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(antiquant_group_size, Int, 0)
    .ATTR(dtype, Int, -1)
    .ATTR(inner_precise, Int, 0)
    .OP_END_FACTORY_REG(WeightQuantBatchMatmulV2)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x | 必选输入 | 矩阵乘运算的左矩阵，公式中的$x$。 | transpose_x=false时，shape为(M, K)；transpose_x=true时，shape为(K, M)，计算时逻辑转置为(M, K)参与计算。M取值范围为[1, 2147483647]，K取值至少为1。 | FLOAT16、BFLOAT16 | ND | 2 |
| weight | 必选输入 | 矩阵乘运算的量化权重右矩阵，公式中的$weight$。 | transpose_weight=false时，shape为(K, N)；transpose_weight=true时，shape为(N, K)。数据类型为INT4、FLOAT4_E2M1时，参与计算的内轴大小需为偶数；数据类型为INT32时，输入为INT4打包数据（每32位存放8个4-bit数据），shape支持(N, K/8)、(K, N/8)。 | INT8、INT4、INT32、FLOAT8_E4M3FN、HIFLOAT8、FLOAT4_E2M1 | ND、FRACTAL_NZ | 2 |
| antiquant_scale | 必选输入 | 反量化参数的缩放因子，公式中的$antiquantScale$。 | pergroup量化模式下shape为(⌈K / antiquant_group_size⌉, N)或(N, ⌈K / antiquant_group_size⌉)；perchannel量化模式下shape为(N,)、(1, N)或(N, 1)；pertensor量化模式下shape为(1,)或(1, 1)。数据类型为FLOAT16、BFLOAT16时需与x一致。 | FLOAT16、BFLOAT16、UINT64、INT64、FLOAT8_E8M0 | ND | 1~2 |
| antiquant_offset | 可选输入 | 反量化参数的偏置因子，公式中的$antiquantOffset$。 | shape与antiquant_scale一致；antiquant_scale数据类型为UINT64、INT64时，本参数数据类型为INT32。 | FLOAT16、BFLOAT16、INT32 | ND | 1~2 |
| quant_scale | 可选输入 | 输出量化参数的缩放因子，公式中的$quantScale$。 | shape支持(1,)、(1, N)、(N,)；输出y数据类型为INT8时本参数必选。 | FLOAT32、UINT64 | ND | 1~2 |
| quant_offset | 可选输入 | 输出量化参数的偏置因子，公式中的$quantOffset$。 | shape与quant_scale一致。 | FLOAT32 | ND | 1~2 |
| bias | 可选输入 | 矩阵乘运算结果上累加的偏置向量，公式中的$bias$。 | shape支持(N,)、(1, N)，N需与weight的N维一致。 | FLOAT16、FLOAT32、BFLOAT16 | ND | 1~2 |
| y | 必选输出 | 矩阵乘运算的计算结果，公式中的$y$。 | shape为(M, N)；quant_scale存在时数据类型为INT8，否则数据类型与x一致。 | FLOAT16、BFLOAT16、INT8 | ND | 2 |
| transpose_x | 必选属性 | 指示左矩阵x参与矩阵乘前是否转置。 | 默认值false，取值true时x逻辑转置后参与计算。 | Bool | - | - |
| transpose_weight | 必选属性 | 指示右矩阵weight参与矩阵乘前是否转置。 | 默认值false，取值true时weight逻辑转置后参与计算。 | Bool | - | - |
| antiquant_group_size | 必选属性 | MX、pergroup量化模式下对weight进行反量化计算的分组大小，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。 | 默认值0，表示不使用pergroup量化模式；取值范围为[0, K-1]且需为32的倍数；antiquant_scale数据类型为FLOAT8_E8M0时必须为32。 | Int | - | - |
| dtype | 必选属性 | 输出y的数据类型。 | 默认值-1，输入存在quant_scale时输出为INT8，否则输出与x数据类型一致；取值支持1（FLOAT16）、2（INT8）、27（BFLOAT16）。 | Int | - | - |
| inner_precise | 必选属性 | 计算模式。 | 默认值0，取值支持0（高精度模式）、1（高性能模式）。 | Int | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - x：不支持转置（transpose_x仅支持false），仅支持连续tensor。
  - weight：数据类型为INT32时作为INT4打包数据的载体；数据类型为FLOAT8_E4M3FN、HIFLOAT8时仅支持ND格式；数据格式为FRACTAL_NZ时transpose_weight仅支持false。
  - quant_scale、quant_offset：暂不支持。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：weight仅支持INT8、INT4、INT32数据类型；antiquant_scale仅支持FLOAT16、BFLOAT16、UINT64、INT64数据类型。
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas推理系列产品</term>：x仅支持FLOAT16数据类型，weight仅支持INT8数据类型，y仅支持FLOAT16数据类型，antiquant_scale仅支持FLOAT16数据类型。
<!-- end id9 -->

## 约束说明

- 确定性计算：
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：默认确定性实现。
  <!-- end id10 -->
  <!-- npu="A3,910b,310p" id11 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas推理系列产品</term>：默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
  <!-- end id11 -->
- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../../docs/zh/context/non_contiguous_tensor.md)仅支持转置场景。
- x与weight的Reduce维度K需大小相等。
- 各量化模式下输入输出的数据类型组合、shape取值关系及对齐要求等详细约束，参见[aclnnWeightQuantBatchMatmulV2](aclnnWeightQuantBatchMatmulV2.md)的约束说明。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR接口  | [test_geir_weight_quant_batch_matmul_v2](../examples/arch35/test_geir_weight_quant_batch_matmul_v2.cpp) | 通过[WeightQuantBatchMatmulV2算子IR](../op_graph/weight_quant_batch_matmul_v2_proto.h)构图并调用算子。 |
