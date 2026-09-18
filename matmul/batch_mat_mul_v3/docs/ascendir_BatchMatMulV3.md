# BatchMatMulV3

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
- <term>Atlas 训练系列产品</term>：支持
<!-- end id6 -->

## 功能说明

- 算子功能：完成带batch维度的矩阵乘计算，输入支持2~6维，前若干维为batch维度（支持batch维广播），最后两维做矩阵乘计算，并支持在乘法结果上累加偏置向量（bias）。
- 计算公式：

  $$
  y = x1 @ x2 + bias
  $$

  其中：
  - $x1$：左矩阵，最后两维为$(M, K)$；当adj_x1=true时，$x1$传入的最后两维为$(K, M)$，计算时逻辑转置为$(M, K)$。
  - $x2$：右矩阵，最后两维为$(K, N)$；当adj_x2=true时，$x2$传入的最后两维为$(N, K)$，计算时逻辑转置为$(K, N)$。
  - $bias$：维度为$(N)$、$(1, N)$或$(B, 1, N)$的偏置向量，逐元素累加到乘法结果对应列上。
  - $y$：输出矩阵，最后两维为$(M, N)$，batch维度为$x1$与$x2$广播后的结果。例如$x1$、$x2$输入维度分别为$(B, M, K)$、$(B, K, N)$时，$y$维度为$(B, M, N)$。B为矩阵乘法的batch数，M为$x1$的行数和$y$的行数，N为$x2$的列数和$y$的列数，K为$x1$的列数和$x2$的行数。

## Ascend IR定义

Ascend IR定义所在头文件路径：[batch_mat_mul_v3_proto.h](../op_graph/batch_mat_mul_v3_proto.h)

```cpp
REG_OP(BatchMatMulV3)
    .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .ATTR(adj_x1, Bool, false)
    .ATTR(adj_x2, Bool, false)
    .ATTR(offset_x, Int, 0)
    .ATTR(enable_hf32, Bool, false)
    .OP_END_FACTORY_REG(BatchMatMulV3)
```

## 参数说明

| 参数名 | 输入/属性/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape） |
| ------ | -------------- | ---- | -------- | -------- | -------- | ------------- |
| x1 | 必选输入 | 矩阵乘运算的左矩阵，公式中的$x1$。 | adj_x1=false时，最后两维为(M, K)；adj_x1=true时，最后两维为(K, M)，计算时逻辑转置为(M, K)参与计算。 | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2~6 |
| x2 | 必选输入 | 矩阵乘运算的右矩阵，公式中的$x2$。 | adj_x2=false时，最后两维为(K, N)；adj_x2=true时，最后两维为(N, K)，计算时逻辑转置为(K, N)参与计算。Reduce维度K需与x1一致。 | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2~6 |
| bias | 可选输入 | 矩阵乘运算结果上累加的偏置向量，公式中的$bias$。 | N需与x2的N维一致。 | FLOAT16、BFLOAT16、FLOAT32 | ND | 1~3 |
| offset_w | 可选输入 | 权重量化偏移矩阵。 | 预留参数，当前暂不支持。 | INT8、INT4 | ND | 2 |
| y | 必选输出 | 矩阵乘运算的计算结果，公式中的$y$。 | batch维度为x1与x2广播后的结果。 | FLOAT16、BFLOAT16、FLOAT32 | ND、FRACTAL_NZ | 2~6 |
| adj_x1 | 必选属性 | 指示左矩阵x1参与矩阵乘前是否转置。 | 默认值false；取值true时x1最后两维逻辑转置后参与计算。 | Bool | - | - |
| adj_x2 | 必选属性 | 指示右矩阵x2参与矩阵乘前是否转置。 | 默认值false；取值true时x2最后两维逻辑转置后参与计算。 | Bool | - | - |
| offset_x | 必选属性 | 左矩阵x1的量化偏移。 | 预留参数，当前暂不支持。默认值0。 | Int | - | - |
| enable_hf32 | 必选属性 | 指示是否以HFLOAT32（高精度浮点32位）模式执行计算。 | 默认值false。 | Bool | - | - |

<!-- npu="950" id7 -->
- <term>Ascend 950PR/Ascend 950DT</term>：仅x2支持FRACTAL_NZ格式。
<!-- end id7 -->
<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：bias不支持BFLOAT16数据类型。
<!-- end id8 -->
<!-- npu="310p" id9 -->
- <term>Atlas 推理系列产品</term>：仅支持FLOAT16数据类型，且x2仅支持FRACTAL_NZ格式。
<!-- end id9 -->

## 约束说明

- 确定性计算：默认确定性实现。
- 支持空tensor，空tensor场景下不支持bias。
- 支持连续tensor，[非连续tensor](../../../docs/zh/context/non_contiguous_tensor.md)仅支持转置场景。
- x1与x2的batch维度需满足[broadcast关系](../../../docs/zh/context/broadcast_relationship.md)，Reduce维度K需大小相等。

## 调用示例

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_batchmatmul](../examples/test_aclnn_batchmatmul.cpp) | 通过<br>[aclnnAddbmm&aclnnInplaceAddbmm](aclnnAddbmm&aclnnInplaceAddbmm.md)<br>[aclnnBaddbmm&aclnnInplaceBaddbmm](aclnnBaddbmm&aclnnInplaceBaddbmm.md)<br>[aclnnBatchMatMul](aclnnBatchMatMul.md)<br>[aclnnBatchMatMulWeightNz](aclnnBatchMatMulWeightNz.md)<br>等方式调用BatchMatMulV3算子。 |
