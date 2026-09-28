# TurboQuant

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR&950DT系列产品</term> | × |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

TurboQuant将MLA的KV latent按token归一化后量化为4bit codebook索引，并单独输出每个token的L2 norm `scale`。输入`latent`应已完成signed Hadamard旋转，但不能预先归一化。

对第`i`个token，算子先计算：

$$
scale_i = norm_i = \sqrt{\sum_d latent_{i,d}^2 + 10^{-16}}
$$

归一化后的元素为：

$$
u_{i,d} = latent_{i,d} / scale_i
$$

根据相邻codebook中心点的中点选择索引：

$$
nibble_{i,d} = \sum_{b=0}^{14} [u_{i,d} \ge (centroids_b + centroids_{b+1}) / 2]
$$

相邻两个维度的索引打包到一个字节中，低nibble保存偶数维索引，高nibble保存奇数维索引：

$$
y_{i,k} = nibble_{i,2k} \mathbin{|} (nibble_{i,2k+1} \ll 4)
$$

## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 | 维度(shape) |
| --- | --- | --- | --- | --- | --- |
| latent | 输入 | 已完成signed Hadamard旋转、未归一化的KV latent | FLOAT32 | ND | `[numTokens, 512]` |
| centroids | 输入 | 按升序排列的16个Lloyd-Max codebook中心点 | FLOAT32 | ND | `[16]` |
| y | 输出 | 两个4bit索引打包后的量化结果，低nibble在前 | UINT8 | ND | `[numTokens, 256]` |
| scale | 输出 | 每个token的L2 norm | FLOAT16 | ND | `[numTokens]` |

## 约束说明

- 算子默认提供确定性计算。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_turbo_quant](./examples/test_aclnn_turbo_quant.cpp) | 通过[aclnnTurboQuant](./docs/aclnnTurboQuant.md)接口方式调用TurboQuant算子。 |
| PyTorch API | - | 通过[turbo_quant](./docs/torchapi_turbo_quant.md)接口调用TurboQuant算子。 |
