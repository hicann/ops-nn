# fused_matmul

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：矩阵乘与通用向量计算融合，底层封装aclnnFusedMatmulV2。
- 计算公式：

  $$
  y = OP((x @ x2 + bias), x3)
  $$

  fused_op_type支持的计算如下：

  - ""：$y = x @ x2 + bias$。
  - "add"：$y = (x @ x2 + bias) + x3$。
  - "mul"：$y = (x @ x2 + bias) * x3$。
  - "gelu_erf"：$y = gelu\_erf(x @ x2)$。
  - "gelu_tanh"：$y = gelu\_tanh(x @ x2)$。
  - "relu"：$y = relu(x @ x2 + bias)$。
  - "16cast32"：$y = cast\_float32(x @ x2 + bias)$。

  当fused_op_type="add"，且alpha或beta不为1时：

  $$
  y = alpha * (x @ x2) + beta * x3
  $$

  该场景不支持bias，算子内部使用scale_add融合模式完成计算。

## 函数原型

```python
cann_ops_nn.fused_matmul(
    x,
    x2,
    *,
    bias=None,
    x3=None,
    alpha=None,
    beta=None,
    fused_op_type="",
) -> Tensor
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| x | Tensor | 必选 | 矩阵乘的第一个输入矩阵，最后两维为(M, K)。 | torch.float16、torch.bfloat16、torch.float32 | fused_op_type为""、"relu"时支持2-6维；为"add"、"mul"时支持2-3维；其他取值支持2维 |
| x2 | Tensor | 必选 | 矩阵乘的第二个输入矩阵，最后两维为(K, N)，数据类型与x一致，K轴长度与x一致。 | 与x一致 | fused_op_type为""、"relu"时支持2-6维；为"add"、"mul"时支持2-3维；其他取值支持2维 |
| bias | Tensor | 可选 | 矩阵乘的偏置。仅当fused_op_type为""、"16cast32"、"relu"、"add"或"mul"时生效。 | torch.float16、torch.bfloat16、torch.float32 | 1-2维 |
| x3 | Tensor | 可选 | 融合运算的输入矩阵。fused_op_type为"add"或"mul"时必须传入，其他场景必须为None。 | 与x一致 | 2-3维 |
| alpha | float | 可选 | 矩阵乘结果的缩放系数。传入None时取值为1.0，传入的数值在接口内部转换为FP32。 | - | - |
| beta | float | 可选 | x3的缩放系数。传入None时取值为1.0，传入的数值在接口内部转换为FP32。 | - | - |
| fused_op_type | str | 可选 | 融合模式。支持""、"16cast32"、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"relu"，默认值为""。"scale_add"仅供算子内部使用，不支持用户传入。 | - | - |

## 返回值说明

| 输出名 | 输出类型 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- |
| y | Tensor | 融合矩阵乘的输出。 | fused_op_type为"16cast32"时为torch.float32，其他场景与x一致 | (..., M, N) |

## 约束说明

- 该接口当前支持单算子模式调用。
- x、x2、bias和x3必须是NPU Tensor，可选Tensor可以传入None。
- x和x2的数据类型必须一致，shape必须满足矩阵乘关系；多维场景下x和x2的batch维度必须一致，不支持batch轴广播。
- 当x为torch.float16或torch.bfloat16时，bias的数据类型必须与x一致或为torch.float32；当x为torch.float32时，bias必须为torch.float32。
- 当fused_op_type为"add"或"mul"时，x3的数据类型必须与x一致。
- 当fused_op_type取值为"add"、"mul"时，在BMM（三维）场景下，x3支持2-3维；二维x3可按矩阵广播用于三维输出，三维x3的batch轴需要与y一致或为1。
- 当fused_op_type为"gelu_erf"或"gelu_tanh"时，不支持传入bias。
- 当alpha或beta不为1时，仅支持fused_op_type="add"的三维非转置场景，不支持bias和batch轴广播，x、x2、x3和y必须为相同的torch.float16或torch.bfloat16数据类型。
- fused_op_type的字符串长度不能超过100。
- torch_npu.npu.matmul.cube_math_type的优先级最高。该值不为None时，接口直接使用其设置的Cube计算模式，torch.npu.matmul.allow_hf32不生效；该值为None时，接口才根据torch.npu.matmul.allow_hf32选择USE_HF32或KEEP_DTYPE。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

  ```python
  import torch
  import torch_npu
  import cann_ops_nn

  x = torch.randn((32, 21, 9), dtype=torch.bfloat16).npu()
  x2 = torch.randn((32, 9, 16), dtype=torch.bfloat16).npu()
  x3 = torch.randn((32, 21, 16), dtype=torch.bfloat16).npu()

  y = cann_ops_nn.fused_matmul(
      x,
      x2,
      x3=x3,
      alpha=1.5,
      beta=0.5,
      fused_op_type="add",
  )
  print(y.shape, y.dtype)
  ```
