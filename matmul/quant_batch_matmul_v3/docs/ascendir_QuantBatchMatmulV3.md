# QuantBatchMatmulV3

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

- Kirin X90处理器系列产品：支持
- Kirin 9030处理器系列产品：支持

## 功能说明

- 算子功能：完成量化场景下的矩阵乘计算，支持对矩阵乘结果进行缩放、累加偏置和输出类型转换。支持T-T、T-C、K-T、K-C、G-B、B-B、MX等[量化模式](../../../docs/zh/context/quant_mode_introduction.md)，不同产品支持的模式及数据类型组合不同。
- 计算公式（以不沿K轴分组的量化模式为例）：
  - 无bias时：

    $$
    y = (op(x1) @ op(x2)) * scale * pertokenScale + offset
    $$

  - bias为INT32时，在缩放前累加：

    $$
    y = (op(x1) @ op(x2) + bias) * scale * pertokenScale + offset
    $$

  - bias为BFLOAT16、FLOAT16或FLOAT32时，在缩放后累加（此场景不支持offset）：

    $$
    y = (op(x1) @ op(x2)) * scale * pertokenScale + bias
    $$

  其中，$pertokenScale$对应输入pertoken_scale，未传入时按1处理；未传入offset时按0处理。offset仅在支持的INT8输出组合中使用，不能与pertoken_scale同时使用。实际输出还需按dtype进行类型转换，UINT64/INT64类型的scale存储编码后的量化参数，不能直接按整数值代入公式。

  INT32输出场景直接输出整数累加结果$y = op(x1) @ op(x2) + bias$（bias可省略），不执行上述缩放；scale仍是IR必选输入。

  G-B、B-B、MX模式沿K轴分组，每组矩阵乘结果乘以对应的左右缩放因子，再累加各组结果；不能将各组的scale统一移到完整矩阵乘之后。

  其中：
  - $op(x1)$、$op(x2)$表示按transpose_x1、transpose_x2属性进行逻辑转置后的矩阵，参与计算的最后两维分别为$(M, K)$、$(K, N)$。属性为false时保持输入不变，为true时交换对应输入的最后两维，输入shape分别为$(K, M)$或$(N, K)$。
  - $y$的最后两维为$(M, N)$，M、N分别为输出矩阵的行数、列数，K为矩阵乘的归约维度。输入和输出的前0 ~ 4维为可选的batch维度，输出batch为两个输入广播后的结果。

## Ascend IR定义

Ascend IR定义所在头文件路径：[quant_batch_matmul_v3_proto.h](../op_graph/quant_batch_matmul_v3_proto.h)

```cpp
REG_OP(QuantBatchMatmulV3)
    .INPUT(x1, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(x2, TensorType({DT_INT8, DT_INT4, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
    .INPUT(scale, TensorType({DT_UINT64, DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_INT32, DT_BF16, DT_FLOAT16, DT_FLOAT}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT8, DT_BF16, DT_INT32, DT_FLOAT}))
    .REQUIRED_ATTR(dtype, Int)
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(group_size, Int, 0)
    .OP_END_FACTORY_REG(QuantBatchMatmulV3)
```

## 参数说明

表中数据类型为IR声明的类型集合，不能任意组合；具体组合以产品及量化模式的约束为准。DT_FLOAT对应FLOAT32，DT_BF16对应BFLOAT16。维度列表示逻辑shape的维数；FRACTAL_NZ的存储shape另含分块轴，不能将存储shape作为ND逻辑shape传入。INT4、FLOAT4_E2M1的逻辑shape按元素个数计数，不是打包后的字节数。

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | 必选输入 | 矩阵乘运算的左矩阵，公式中的$op(x1)$。 | transpose_x1 = false时逻辑shape的最后两维为(M, K)，为true时为(K, M)。INT4仅支持二维且transpose_x1 = false。FRACTAL_NZ的存储shape为4 ~ 8维。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND、FRACTAL_NZ（受产品限制） | 2 ~ 6 |
| x2 | 必选输入 | 矩阵乘运算的右矩阵，公式中的$op(x2)$。 | transpose_x2 = false时逻辑shape的最后两维为(K, N)，为true时为(N, K)。INT4仅支持二维。FRACTAL_NZ的存储shape为4 ~ 8维，需使用对应产品支持的存储排布。 | INT8、INT4、HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1 | ND、FRACTAL_NZ（受产品限制） | 2 ~ 6 |
| scale | 必选输入 | 右矩阵的量化缩放参数，公式中的$scale$。 | pertensor/perchannel时shape为(1,)或(N,)；分组模式的shape见下表。UINT64/INT64为编码后的量化参数。 | UINT64、FLOAT32、INT64、BFLOAT16、FLOAT8_E8M0 | ND | 1 ~ 7 |
| offset | 可选输入 | 输出量化的偏置因子，公式中的$offset$。 | shape为(1,)或(N,)，仅支持相应的INT8输出组合；不传入时按0处理。 | FLOAT32 | ND | 1 |
| bias | 可选输入 | 矩阵乘累加的偏置，公式中的$bias$。 | shape为(N,)或(B, 1, N)，其中B为输出batch数。输出为二维、四维、五维、六维时仅支持一维bias；三维输出支持一维或三维bias。 | INT32、BFLOAT16、FLOAT16、FLOAT32 | ND | 1或3 |
| pertoken_scale | 可选输入 | 左矩阵的量化缩放参数，公式中的$pertokenScale$。 | 非分组模式下，INT8、INT4输入要求shape为(M,)；FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8输入支持(1,)或(M,)，还须满足下述WeightNZ限制。分组模式的shape见下表。未传入时按1处理，G-B、B-B、MX模式需传入。 | FLOAT32、FLOAT8_E8M0 | ND | 1 ~ 7 |
| y | 必选输出 | 矩阵乘运算的计算结果，公式中的$y$。 | batch维为x1与x2广播后的结果，逻辑shape的最后两维为(M, N)。数据类型由dtype指定；Atlas推理系列产品的pertoken场景使用FRACTAL_NZ。 | FLOAT16、INT8、BFLOAT16、INT32、FLOAT32 | ND、FRACTAL_NZ（受产品限制） | 2 ~ 6 |
| dtype | 必选属性 | 指定输出y的数据类型。 | 必须显式设置，无IR默认值。取值0（FLOAT32）、1（FLOAT16）、2（INT8）、3（INT32）、27（BFLOAT16），受产品及输入组合限制。 | Int | - | - |
| transpose_x1 | 可选属性 | 指示左矩阵x1参与矩阵乘前是否转置。 | 默认值false，为true时逻辑转置最后两维。 | Bool | - | - |
| transpose_x2 | 可选属性 | 指示右矩阵x2参与矩阵乘前是否转置。 | 默认值false，为true时逻辑转置最后两维。 | Bool | - | - |
| group_size | 可选属性 | 分组量化的分组大小。 | 默认值0。低16位为group_size_k，第16 ~ 31位为group_size_n，第32 ~ 47位为group_size_m，即group_size = group_size_k \| (group_size_n << 16) \| (group_size_m << 32)。 | Int | - | - |

### 分组量化shape

<!-- npu="950" id11 -->

<term>Ascend 950PR&950DT系列产品</term>的G-B、B-B、MX分组量化shape关系如下。batch_a、batch_b分别表示x1、x2的前0 ~ 4维，可省略；pertoken_scale的batch与batch_a一致，scale的batch与batch_b一致。batch_a与batch_b可以不同，但需满足广播规则。表中scale和pertoken_scale均为ND格式，x1、x2列表示逻辑shape；G-B、B-B以两个输入均不转置为例。MX表展示转置属性对应的shape关系，不表示所有数据类型和格式均支持全部组合。对于FLOAT4_E2M1的WeightNZ场景，按当前IR接口约定，仅适用x1不转置的前两行，且须满足下述WeightNZ约束；x1转置的后两行不适用于该场景。

| 量化模式 | x1 shape | x2 shape | pertoken_scale shape | scale shape | 最终[group_size_m, group_size_n, group_size_k] |
| -------- | -------- | -------- | -------------------- | ----------- | ------------------------------------------ |
| G-B | (batch_a, M, K) | (batch_b, K, N) | (batch_a, M, ⌈K/128⌉) | (batch_b, ⌈K/128⌉, ⌈N/128⌉) | [1, 128, 128] |
| B-B | (batch_a, M, K) | (batch_b, K, N) | (batch_a, ⌈M/128⌉, ⌈K/128⌉) | (batch_b, ⌈K/128⌉, ⌈N/128⌉) | [128, 128, 128] |
| MX | (batch_a, M, K) | (batch_b, K, N) | (batch_a, M, ⌈K/64⌉, 2) | (batch_b, ⌈K/64⌉, N, 2) | [1, 1, 32] |
| MX | (batch_a, M, K) | (batch_b, N, K) | (batch_a, M, ⌈K/64⌉, 2) | (batch_b, N, ⌈K/64⌉, 2) | [1, 1, 32] |
| MX | (batch_a, K, M) | (batch_b, K, N) | (batch_a, ⌈K/64⌉, M, 2) | (batch_b, ⌈K/64⌉, N, 2) | [1, 1, 32] |
| MX | (batch_a, K, M) | (batch_b, N, K) | (batch_a, ⌈K/64⌉, M, 2) | (batch_b, N, ⌈K/64⌉, 2) | [1, 1, 32] |

- G-B、B-B模式的scale与pertoken_scale类型为FLOAT32，MX模式为FLOAT8_E8M0。
- 分组大小中为0的分量由输入shape推导，推导时对应维度需满足整除要求，推导后必须满足上述组合；其他量化模式的group_size必须为0。
- 当分组轴不能被对应分组大小整除时，建议显式指定分组大小，避免使用0导致推导失败。例如MX使用`(int64_t{1} << 32) | (int64_t{1} << 16) | 32`。末轴2表示相邻两个K方向32元素分组的scale，不能将MX的group_size_k设置为64。

<!-- end id11 -->

### 各产品使用说明

<!-- npu="A3,910b" id7 -->

#### <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>

x1、x2仅支持INT8、INT4；scale不支持FLOAT8_E8M0；pertoken_scale仅支持FLOAT32；y不支持FLOAT32。x1为FRACTAL_NZ时要求x2为FRACTAL_NZ，且x1不能为INT4，y不能为INT8。x1、x2的逻辑shape最后一维均不能超过65535。

<!-- end id7 -->

<!-- npu="310p" id8 -->

#### <term>Atlas推理系列产品</term>

x1、x2仅支持INT8；scale支持UINT64、INT64，pertoken场景支持FLOAT32；bias支持INT32，pertoken场景额外支持FLOAT32；y支持FLOAT16、INT8，pertoken场景仅支持FLOAT16。非pertoken场景的x1、x2、y格式组合为ND、FRACTAL_NZ、ND；pertoken场景为FRACTAL_NZ、FRACTAL_NZ、FRACTAL_NZ。

<!-- end id8 -->

<!-- npu="950" id9 -->

#### <term>Ascend 950PR&950DT系列产品</term>

本文ND scale场景中，仅x2支持FRACTAL_NZ，其余输入输出使用ND。WeightNZ的x1与x2类型必须相同，且为INT8、HIFLOAT8、FLOAT8_E4M3FN或FLOAT4_E2M1，不支持INT4、FLOAT8_E5M2；K、N不能为1。

- HIFLOAT8、FLOAT8_E4M3FN的非MX场景：未传入pertoken_scale时，scale为UINT64/INT64；传入pertoken_scale时，scale与pertoken_scale均为FLOAT32，且transpose_x1必须为false。后一场景的非分组量化仅支持K-C、K-T，pertoken_scale的shape为(M,)，不支持M大于1时传入(1,)实现T-C、T-T。
- FLOAT8_E4M3FN、FLOAT4_E2M1的MX场景：scale与pertoken_scale均为FLOAT8_E8M0，二者均须传入，shape按上述MX表中适用的转置组合设置。
- FLOAT4_E2M1的WeightNZ场景：按当前[IR原型](../op_graph/quant_batch_matmul_v3_proto.h)约定，transpose_x1必须为false；transpose_x2 = false时N必须大于2，transpose_x2 = true时N必须大于1。同时需满足下述MX输入内轴为偶数、K大于2的约束。

<!-- end id9 -->

#### Kirin X90/Kirin 9030处理器系列产品

x1、x2仅支持INT8，x1仅支持ND，x2仅支持FRACTAL_NZ；scale仅支持UINT64、INT64；bias仅支持INT32；pertoken_scale仅支持FLOAT32；y仅支持FLOAT16、INT8。

## 约束说明

- x1与x2参与计算的K维大小必须相等，batch维需满足[broadcast关系](../../../docs/zh/context/broadcast_relationship.md)。
- G-B、B-B模式不支持bias。
- INT4输入仅支持二维，transpose_x1必须为false，输入的最后一维必须为偶数。
- MX模式下，FLOAT4_E2M1输入的内轴（传入逻辑shape的最后一维）必须为偶数，K必须大于2；x2为FRACTAL_NZ时还需满足上述WeightNZ约束。
- 空tensor：
  <!-- npu="950" id10 -->
  - <term>Ascend 950PR&950DT系列产品</term>：x2为ND时，支持x1的M = 0或x2的N = 0并输出空tensor；x2为FRACTAL_NZ时，支持x1的M = 0并输出空tensor。
  <!-- end id10 -->
  - 其他场景不支持维度大小为0的输入。
- 确定性计算：
  <!-- npu="A3,910b,310p,950" id12 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>、<term>Atlas推理系列产品</term>、<term>Ascend 950PR&950DT系列产品</term>：默认确定性实现。
  <!-- end id12 -->
- Batch一致性：
  <!-- npu="950" id13 -->
  - <term>Ascend 950PR&950DT系列产品</term>：B-B量化模式不支持[Batch一致性](../../../docs/zh/context/batch_consistency.md)，不能将默认确定性等同于不同batch组合下的结果一致。
  <!-- end id13 -->
- 支持连续tensor；输入转置由transpose_x1、transpose_x2属性指定，FRACTAL_NZ输入需满足对应的存储排布要求，不能仅修改format而不转换数据。
- 本文介绍scale为ND格式时的GEIR构图调用。ScaleNZ的ACLNN直调和ACLNN Graph调用要求，参见[aclnnQuantMatmulWeightNz](aclnnQuantMatmulWeightNz.md#scalenz约束)。
- GEIR构图需按IR要求准备输入类型，不能依赖ACLNN接口的预处理。例如，INT8输入、FLOAT16输出且不传入pertoken_scale时，scale使用编码后的UINT64/INT64，不能直接传入FLOAT32缩放值。
- 更多数据类型组合、shape及对齐约束，参见[aclnnQuantMatmulV4](aclnnQuantMatmulV4.md)、[aclnnQuantMatmulWeightNz](aclnnQuantMatmulWeightNz.md)及[aclnnQuantMatmulV5](../../quant_batch_matmul_v4/docs/aclnnQuantMatmulV5.md)的对应全量化场景。上述ACLNN接口还包含输入转换、接口层校验及其他算子的场景，不能将其全部类型、限制和调用能力直接用于本IR。

## 调用示例

| 调用方式 | 样例代码 | 说明 |
| -------- | -------- | ---- |
| GEIR接口 | [test_geir_quant_batch_matmul_v3](../examples/test_geir_quant_batch_matmul_v3.cpp) | 通过[QuantBatchMatmulV3算子IR](../op_graph/quant_batch_matmul_v3_proto.h)构图并调用算子。 |

示例使用INT8输入、FLOAT32的scale与pertoken_scale、FLOAT16输出，并在Host侧校验结果。运行环境为Atlas A2系列产品、Atlas A3系列产品或Ascend 950PR&950DT系列产品。

环境准备和构建方法参见[《算子调用指南》的GE图模式章节](../../../docs/zh/invocation/quick_op_invocation.md#ge图模式)。
