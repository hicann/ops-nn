# QuantBatchMatmulV4

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
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：完成量化场景下的矩阵乘计算（反量化与矩阵乘融合计算），支持A8W8、A8W4、MxA8W4、A4W4等量化场景，并支持在乘法结果上累加偏置向量（bias）。支持T-T、T-C、K-T、K-C、K-G、G-B、B-B、MX、T-CG等[量化模式](../../../docs/zh/context/quant_mode_introduction.md)，不同量化模式对应的输入输出数据类型组合及约束参见[aclnnQuantMatmulV5](aclnnQuantMatmulV5.md)的约束说明。
- 计算公式（以K-C && K-T量化模式为例）：

  $$
  y = x1 @ x2 * x2Scale * x1Scale + bias
  $$

  其中：
  - $x1$：左矩阵，前0~4维为batch维度（可不存在），最后两维为$(M, K)$；当transpose_x1=true时，$x1$传入的最后两维为$(K, M)$，计算时逻辑转置为$(M, K)$。
  - $x2$：量化权重右矩阵，前0~4维为batch维度（可不存在），最后两维为$(K, N)$；当transpose_x2=true时，$x2$传入的最后两维为$(N, K)$，计算时逻辑转置为$(K, N)$。
  - $x1Scale$、$x2Scale$：量化参数的缩放因子，shape随量化模式不同而不同。
  - $bias$：维度为$(N)$、$(1, N)$或$(B, 1, N)$的偏置向量。
  - $y$：输出矩阵，前0~4维为batch维度（可不存在），最后两维为$(M, N)$。B为矩阵乘法的batch数，M为$x1$的行数和$y$的行数，N为$x2$的列数和$y$的列数，K为$x1$的列数和$x2$的行数。

## Ascend IR定义

Ascend IR定义所在头文件路径：[quant_batch_matmul_v4_proto.h](../op_graph/quant_batch_matmul_v4_proto.h)

```cpp
REG_OP(QuantBatchMatmulV4)
    .INPUT(x1, TensorType({DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_INT8, DT_INT4}))
    .INPUT(x2, TensorType({DT_FLOAT4_E2M1, DT_INT4, DT_INT8}))
    .OPTIONAL_INPUT(bias, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E8M0, DT_UINT64, DT_FLOAT32}))
    .OPTIONAL_INPUT(y_scale, TensorType({DT_UINT64}))
    .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT16, DT_BF16}))
    .OPTIONAL_INPUT(y_offset, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT32}))
    .OPTIONAL_INPUT(x2_table, TensorType({DT_INT8}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(compute_type, Int, -1)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, -1)
    .OP_END_FACTORY_REG(QuantBatchMatmulV4)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | 必选输入 | 矩阵乘运算的左矩阵，公式中的$x1$。 | transpose_x1=false时，最后两维为(M, K)；transpose_x1=true时，最后两维为(K, M)，计算时逻辑转置为(M, K)参与计算。 | FLOAT8_E5M2<sup>1</sup>、FLOAT8_E4M3FN<sup>1</sup>、INT8、INT4 | ND | 2~6 |
| x2 | 必选输入 | 矩阵乘运算的量化权重右矩阵，公式中的$x2$。 | transpose_x2=false时，最后两维为(K, N)；transpose_x2=true时，最后两维为(N, K)，计算时逻辑转置为(K, N)参与计算。FRACTAL_NZ格式下，transpose_x2=true时shape为(k1, n1, n0, k0)，n0=16，k0=32（x2为INT4时k0=64）；transpose_x2=false时shape为(n1, k1, k0, n0)，k0=16，n0=32（x2为INT4时n0=64）。 | FLOAT4_E2M1<sup>1</sup>、INT4、INT8 | ND、FRACTAL_NZ | 2~6 |
| bias | 可选输入 | 矩阵乘运算结果上累加的偏置向量，公式中的$bias$。 | N需与x2的N维一致。 | BFLOAT16、FLOAT16、FLOAT32 | ND | 1~3 |
| x1_scale | 可选输入 | x1量化参数的缩放因子，公式中的$x1Scale$。 | shape随量化模式不同而不同。 | FLOAT16<sup>1</sup>、BFLOAT16<sup>1</sup>、FLOAT8_E8M0<sup>1</sup>、FLOAT32 | ND | 1~7 |
| x2_scale | 可选输入 | x2量化参数的缩放因子，公式中的$x2Scale$。 | shape随量化模式不同而不同。 | FLOAT16<sup>1</sup>、BFLOAT16、FLOAT8_E8M0<sup>1</sup>、UINT64、FLOAT32 | ND | 1~7 |
| y_scale | 可选输入 | 输出y量化参数的缩放因子。 | T-CG量化模式下使用，shape为(1, N)。 | UINT64 | ND | 2 |
| x1_offset | 可选输入 | x1量化参数的偏置因子。 | 预留参数，当前暂不支持。 | FLOAT16、BFLOAT16 | ND | 1~7 |
| x2_offset | 可选输入 | x2量化参数的偏置因子。 | shape为(t,)，t=1或N，N与x2的N维一致。 | FLOAT16、BFLOAT16 | ND | 1~2 |
| y_offset | 可选输入 | 输出y量化参数的偏置因子。 | K-G量化模式下使用，shape为(N,)。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 1 |
| x2_table | 可选输入 | x2量化参数的查找表。 | 预留参数，当前暂不支持。 | INT8 | ND | 2 |
| y | 必选输出 | 矩阵乘运算的计算结果，公式中的$y$。 | batch维度为x1与x2广播后的结果，最后两维为(M, N)。 | FLOAT16、BFLOAT16 | ND | 2~6 |
| dtype | 必选属性 | 输出y的数据类型。 | 取值1（FLOAT16）、27（BFLOAT16）。 | Int | - | - |
| compute_type | 必选属性 | 计算类型。 | 默认值-1（auto），取值支持-1（auto）、4（f8f4）。 | Int | - | - |
| transpose_x1 | 必选属性 | 指示左矩阵x1参与矩阵乘前是否转置。 | 默认值false，取值true时x1最后两维逻辑转置后参与计算。 | Bool | - | - |
| transpose_x2 | 必选属性 | 指示右矩阵x2参与矩阵乘前是否转置。 | 默认值false，取值true时x2最后两维逻辑转置后参与计算。 | Bool | - | - |
| group_size | 必选属性 | 量化分组大小，由groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，占用低48位。 | 默认值-1，仅在MX、G-B、B-B、K-G、T-CG量化模式中生效，计算公式为group_size = groupSizeK \| groupSizeN << 16 \| groupSizeM << 32。 | Int | - | - |

<!-- npu="A3,910b" id7 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：上表数据类型列中的角标“1”代表该系列不支持的数据类型；不支持y_scale输入。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：x2_offset、y_offset暂不支持。
<!-- end id8 -->

## 约束说明

- 确定性计算：默认确定性实现。
- 空tensor：
  <!-- npu="A3,910b" id9 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：不支持空tensor。
  <!-- end id9 -->
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：全量化场景下支持空tensor，当x1为m=0或x2为n=0的空tensor时，输出为空tensor（x2为FRACTAL_NZ格式时仅支持x1为m=0的空tensor）；伪量化（A8W4）场景不支持空tensor。
  <!-- end id10 -->
- 支持连续tensor，[非连续tensor](../../../docs/zh/context/non_contiguous_tensor.md)仅支持转置场景。
- 各量化模式下输入输出的数据类型组合、shape取值关系及对齐要求等详细约束，参见[aclnnQuantMatmulV5](aclnnQuantMatmulV5.md)的约束说明。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GEIR接口  | [test_geir_quant_batch_matmul_v4](../examples/arch35/test_geir_quant_batch_matmul_v4.cpp) | 通过[QuantBatchMatmulV4算子IR](../op_graph/quant_batch_matmul_v4_proto.h)构图并调用算子。 |
