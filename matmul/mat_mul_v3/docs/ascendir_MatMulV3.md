# MatMulV3

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：完成两个二维矩阵的通用矩阵乘（GEMM）计算，支持通过属性分别控制左矩阵、右矩阵在参与计算前是否转置，并支持在乘法结果上按列累加偏置向量（bias），即"乘、加"一步完成。
- 计算公式：

  $$
  y = op(x1) @ op(x2) + bias
  $$

  其中：
  - $op(x1)$：参与矩阵乘的左矩阵，维度为$(M, K)$。当transpose_x1=false时，$op(x1)=x1$；当transpose_x1=true时，$op(x1)=x1^T$（x1传入为$(K, M)$，计算时逻辑转置为$(M, K)$）。
  - $op(x2)$：参与矩阵乘的右矩阵，维度为$(K, N)$。当transpose_x2=false时，$op(x2)=x2$；当transpose_x2=true时，$op(x2)=x2^T$（x2传入为$(N, K)$，计算时逻辑转置为$(K, N)$）。
  - $bias$：维度为$(N)$或$(1, N)$的偏置向量，逐元素累加到乘法结果对应列上。
  - $y$：输出矩阵，维度为$(M, N)$。

## Ascend IR定义

Ascend IR定义所在头文件路径：[mat_mul_v3_proto.h](../op_graph/mat_mul_v3_proto.h)

```cpp
REG_OP(MatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(transpose_x1, Bool, false)
    .ATTR(transpose_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(opImplMode, Int, 0x1)
    .OP_END_FACTORY_REG(MatMulV3)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | 必选输入 | 矩阵乘运算的左矩阵，公式中的$op(x1)$。 | transpose_x1=false时，shape为(M, K)；transpose_x1=true时，shape为(K, M)，计算时逻辑转置为(M, K)参与计算。 | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2 |
| x2 | 必选输入 | 矩阵乘运算的右矩阵，公式中的$op(x2)$。 | transpose_x2=false时，shape为(K, N)；transpose_x2=true时，shape为(N, K)，计算时逻辑转置为(K, N)参与计算。FRACTAL_NZ格式下为4维：不转置时排布为(n1, k1, k0, n0)，转置时排布为(k1, n1, n0, k0)，其中k0=16、n0=16。 | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2（ND）或4（NZ） |
| bias | 可选输入 | 矩阵乘运算结果上累加的偏置向量，公式中的$bias$。 | 长度需与x2的N维一致。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 1~2 |
| offset_w | 可选输入 | 权重量化偏移矩阵。 | 预留参数，当前暂不支持。 | INT8、INT4 | ND | 2 |
| y | 必选输出 | 矩阵乘运算的计算结果，公式中的$y$。 | - | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2 |
| transpose_x1 | 必选属性 | 指示左矩阵x1参与矩阵乘前是否转置。 | 默认值false；取值true时x1逻辑转置后参与计算。 | Bool | - | - |
| transpose_x2 | 必选属性 | 指示右矩阵x2参与矩阵乘前是否转置。 | 默认值false；取值true时x2逻辑转置后参与计算。 | Bool | - | - |
| offset_x | 必选属性 | 左矩阵x1的量化偏移。 | 预留参数，当前暂不支持。默认值0。 | Int | - | - |
| opImplMode | 必选属性 | 算子实现模式。 | 默认值0x1（default）。支持的取值：0x1（default，默认模式）、0x2（high_performance，高性能模式）、0x4（high_precision，高精度模式）、0x8（super_performance，超高性能模式）、0x10（support_of_bound_index）、0x20（enable_float_32_execution，FLOAT32执行）、0x40（enable_hi_float_32_execution，HF32执行）。 | Int | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：仅x2支持FRACTAL_NZ格式。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持FRACTAL_NZ格式，bias不支持BFLOAT16数据类型。
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas 推理系列产品</term>：仅支持FLOAT16数据类型，且x2仅支持FRACTAL_NZ格式。
<!-- end id9 -->

## 约束说明

- 确定性计算：默认确定性实现。
- 支持空tensor，空tensor场景下不支持bias。
- 支持连续tensor，[非连续tensor](../../../docs/zh/context/non_contiguous_tensor.md)仅支持转置场景。
- x1与x2需满足数据类型[互推导关系](../../../docs/zh/context/deduction_relationship.md)。
<!-- npu="A3,910b" id10 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持两个输入分别为BFLOAT16和FLOAT16、BFLOAT16和FLOAT32的类型推导。
<!-- end id10 -->

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_addmm](../examples/test_aclnn_addmm_aclnninplace_addmm.cpp) | 通过<br> - [aclnnAddmm&aclnnInplaceAddmm](aclnnAddmm&aclnnInplaceAddmm.md)<br> - [aclnnMatmul](aclnnMatmul.md)<br> - [aclnnMatmulWeightNz](aclnnMatmulWeightNz.md)<br> - [aclnnMm](aclnnMm.md)<br>- [aclnnAddmmWeightNz](aclnnAddmmWeightNz.md)<br>等方式调用MatMulV3算子。 |
