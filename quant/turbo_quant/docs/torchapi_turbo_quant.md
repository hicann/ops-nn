# turbo_quant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
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

- 接口功能：`turbo_quant`将MLA的KV latent按token归一化后量化为4bit codebook索引，并返回每个token的L2 norm。输入`latent`应已完成signed Hadamard旋转，但不能预先归一化。底层封装aclnnTurboQuant。

- 计算公式：

  1. 计算每个token的L2 norm：

    $$
    scale_i = norm_i = \sqrt{\sum_d latent_{i,d}^2 + 10^{-16}}
    $$

  2. 使用`norm_i`归一化：

    $$
    u_{i,d} = latent_{i,d} / scale_i
    $$

  3. 根据相邻codebook中心点的中点选择最近的4bit索引：

    $$
    nibble_{i,d} = \sum_{b=0}^{14} [u_{i,d} \ge (centroids_b + centroids_{b+1}) / 2]
    $$

  4. 相邻两个维度的索引打包到一个字节中，低nibble保存偶数维索引，高nibble保存奇数维索引：

    $$
    y_{i,k} = nibble_{i,2k} \mathbin{|} (nibble_{i,2k+1} \ll 4)
    $$

## 函数原型

```python
cann_ops_nn.turbo_quant(
    latent,
    centroids,
) -> (Tensor, Tensor)
```

也可以直接使用已注册的PyTorch算子：

```python
torch.ops.cann_ops_nn.turbo_quant(latent, centroids)
```

## 参数说明

| 参数名 | 参数类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| latent | Tensor | 必选 | 输入latent，对应公式中的`latent`。必须已完成signed Hadamard旋转，且未归一化。 | float32 | `[numTokens, 512]` |
| centroids | Tensor | 必选 | 按升序排列的16个Lloyd-Max codebook中心点，对应公式中的`centroids`。 | float32 | `[16]` |

## 返回值说明

| 输出名 | 输出类型 | 可选/必选 | 描述 | 数据类型 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| y | Tensor | 必选 | 两个4bit索引打包后的量化结果，对应公式中的`y`，低nibble在前。 | uint8 | `[numTokens, 256]` |
| scale | Tensor | 必选 | 每个token的L2 norm，对应公式中的`scale`。 | float16 | `[numTokens]` |

## 约束说明

- 输入和输出必须位于NPU，不支持非连续Tensor。
- 该接口支持训练、推理场景下使用。
- 该接口仅支持单算子模式调用，不支持TorchAir图模式调用。

## 确定性计算

默认支持确定性计算。

## 调用示例

- 单算子模式调用（eager）

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # latent需已完成signed Hadamard旋转，且未归一化
    latent = torch.randn((2, 512), dtype=torch.float32).npu()
    # centroids必须为升序排列的16个中心点
    centroids = torch.linspace(-1.0, 1.0, 16, dtype=torch.float32).npu()
    y, scale = cann_ops_nn.turbo_quant(latent, centroids)
    print("y:============", y.shape, y.cpu())
    print("scale:============", scale.shape, scale.cpu())
    ```
