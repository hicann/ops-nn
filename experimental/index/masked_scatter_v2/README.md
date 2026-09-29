# MaskedScatterV2

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |


## 功能说明

- **算子功能**：根据布尔掩码 mask，将输入张量 x 中 mask 为 true 的位置，按扁平顺序替换为 updates 中的值。
- **计算公式**：给定输入张量 x、掩码张量 mask 与更新张量 updates，按一维扁平顺序展开，记 cnt(i) 为 mask 在位置 i 之前为 true 的元素个数：

  y_i = updates_{cnt(i)}（mask_i = true）；y_i = x_i（mask_i = false）

- **示例**：x=[1,2,3,4,5,6,7,8]，mask=[true,false,true,false,true,false,true,false]，updates=[10,20,30,40]
  → y=[10,2,20,4,30,6,40,8]。


## 参数说明

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| ------ | -------- | ---- | ------- | ------- |
| x | 输入 | 待更新张量 | FLOAT、FLOAT16、BFLOAT16、INT32 | ND |
| mask | 输入 | 布尔掩码 | BOOL | ND |
| updates | 输入 | 按 mask=true 位置顺序消耗的数据 | 与 x 相同 | ND |
| y | 输出 | 结果张量（shape = x 的 shape） | 与 x 相同 | ND |

## 约束说明

- mask 广播：mask 的 rank ≤ x 的 rank 且各维可广播（右对齐）；x 元素数 ≥ 2^31 时不支持（int32 偏移域）；
- 广播由 kernel 原生处理的准入条件：rank ≤ 8、mask 最内扩展维 stride = 1、x 元素数 ≥ 8192、mask 物理字节数 ≥ 32；
  其余广播 shape（如 dim-last 广播）需调用方先展开为与 x 同 shape（tiling 校验不通过时返回失败并提示）；
- x、updates、y 数据类型必须相同，且均需 contiguous；
- updates 按 mask = true 的扁平顺序依次消耗，长度必须 ≥ 广播展开后 mask 的 true 元素数。

## 调用说明

通过 aclnn 两段式接口 `aclnnMaskedScatterV2` 调用（由 opbuild autogen 生成，见
[docs/aclnnMaskedScatterV2.md](docs/aclnnMaskedScatterV2.md) 与
[examples/test_aclnn_masked_scatter_v2.cpp](examples/test_aclnn_masked_scatter_v2.cpp)）。
