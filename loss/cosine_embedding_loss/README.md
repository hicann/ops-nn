# CosineEmbeddingLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：计算两个输入张量的余弦嵌入损失。`x1` 和 `x2` 先按 ND 规则广播到共同 shape，固定按第 1 维（0-based `axis=1`）作为特征维做余弦相似度归约；删除该维后的 shape 再与 `target` 广播，得到逐元素 loss。`target` 中等于 1 的元素按相似样本处理，等于 -1 的元素按不相似样本处理，其他取值输出 0。
- 计算公式：

  $$
  cos_p = \frac{\sum_c x1_{p,c} x2_{p,c}}{\sqrt{\sum_c x1_{p,c}^2 + 1e-12}\sqrt{\sum_c x2_{p,c}^2 + 1e-12}}
  $$

  $$
  loss_p =
  \begin{cases}
	  1 - cos_p, & target_p = 1 \\
	  \max(0, cos_p - margin), & target_p = -1 \\
	  0, & \text{otherwise}
  \end{cases}
  $$

`reduction` 为 `none` 时输出逐元素 loss，输出 shape 为 `broadcast(remove_axis_1(broadcast(x1.shape, x2.shape)), target.shape)`；`sum` 时输出 loss 之和，`mean` 时输出 loss 均值，二者输出 shape 均为 `[1]`。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|---|---|---|---|---|
| x1 | 输入 | 第一个输入张量，与 `x2` 支持 ND 广播；广播后第 1 维为特征归约维。 | INT32、FLOAT16、FLOAT32 | ND |
| x2 | 输入 | 第二个输入张量，与 `x1` 支持 ND 广播，数据类型必须与 `x1` 相同。 | INT32、FLOAT16、FLOAT32 | ND |
| target | 输入 | 标签张量，与 `remove_axis_1(broadcast(x1.shape, x2.shape))` 支持 ND 广播；支持取值 1 或 -1，其他取值按 0 loss 处理。 | INT32、FLOAT16、FLOAT32 | ND |
| margin | 属性 | 不相似样本分支的间隔，默认 0.0。 | FLOAT | - |
| reduction | 属性 | 归约方式，支持 `none`、`sum`、`mean`，默认 `mean`。 | STRING | - |
| y | 输出 | 损失输出。`none` 输出逐元素结果，`sum`/`mean` 输出 shape `[1]`。 | FLOAT32 | ND |

## 约束说明

- `x1` 与 `x2` 必须可广播，且广播后 rank 至少为 2。
- `x1` 与 `x2` 的数据类型必须相同；`target` 可独立使用 INT32、FLOAT16 或 FLOAT32。
- `target` 必须可广播到 `x1/x2` 广播 shape 删除第 1 维后的 shape；也支持通过额外前导维扩展最终逐元素输出 shape。
- 输入 rank 最大支持 8；tiling 阶段要求运行时维度为正数。
- `reduction` 仅支持 `none`、`sum`、`mean`。
- 输入按 ND 格式处理。
- 当前 aclnn 接口不涉及，通过 GE/算子 IR 调用。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---|---|---|
| 图模式调用 | [test_geir_cosine_embedding_loss.cpp](examples/arch35/test_geir_cosine_embedding_loss.cpp) | 通过[算子IR](op_graph/cosine_embedding_loss_proto.h)构图方式调用 `CosineEmbeddingLoss` 算子。 |
