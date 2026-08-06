# EmbeddingDenseGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----: |
| Atlas 310P 推理系列产品 | √ |
## 功能说明

- 算子功能：`Embedding` 前向的反向算子，根据前向的索引 `indices` 将上游梯度 `grad` 累加（scatter-add）回词表梯度 `y` 上。当 `padding_idx` 有效时，该索引对应的行梯度被置零；当 `scale_grad_by_freq` 为 `true` 时，按各索引出现的频次对梯度做缩放。
- 计算公式：

  $$
  y[\text{indices}[i]] \mathrel{+}= grad[i]
  $$

  其中 `indices[i] == padding_idx` 的项不参与累加；若 `scale_grad_by_freq == true`，则 `y[w] = y[w] / freq[w]`。

- 示例：
  假设 `grad = [[1, 2], [3, 4]]`，`indices = [0, 2]`，`num_weights = 4`，`padding_idx = -1`，`scale_grad_by_freq = false`，则输出：

  $$
  y = \begin{bmatrix} 1 & 2 \\ 0 & 0 \\ 3 & 4 \\ 0 & 0 \end{bmatrix}
  $$

## 参数说明

<table>
<tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">EmbeddingDenseGrad</th></tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">grad</td><td align="center">-</td><td align="center">float16, float32</td><td align="center">ND</td></tr>
<tr><td align="center">indices</td><td align="center">-</td><td align="center">int32, int64</td><td align="center">ND</td></tr>
<tr><td rowspan="3" align="center">算子属性</td><td align="center">num_weights</td><td align="center">-</td><td align="center">int（必选）</td><td align="center">-</td></tr>
<tr><td align="center">padding_idx</td><td align="center">-</td><td align="center">int（默认 -1）</td><td align="center">-</td></tr>
<tr><td align="center">scale_grad_by_freq</td><td align="center">-</td><td align="center">bool（默认 false）</td><td align="center">-</td></tr>
<tr><td rowspan="2" align="center">算子输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[num_weights, embedding_dim]</td><td align="center">float16, float32</td><td align="center">ND</td></tr>
</table>

- 输出 `y` 的 shape 为 `[num_weights, grad.shape[-1]]`，dtype 与 `grad` 一致。

## 约束说明

- `indices` 的元素值必须在 `[0, num_weights)` 范围内，`padding_idx` 除外。
- `grad` 的第 0 维大小必须等于 `indices` 的元素个数。
- `padding_idx` 默认为 `-1`（不填充），有效范围 `[-1, num_weights)`。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| ---- | ---- | ---- |
| aclnn 调用 | [test_aclnn_embedding_dense_grad.cpp](./examples/test_aclnn_embedding_dense_grad.cpp) | 通过 [aclnnEmbeddingDenseGrad](./docs/aclnnEmbeddingDenseGrad.md) 文档中描述的 `aclnnEmbeddingDenseGrad` 两段式接口调用 EmbeddingDenseGrad 算子。 |

## 编译与部署

```bash
# 在 ops-nn 仓根目录
bash build.sh --pkg --experimental --soc=ascend310p --ops=embedding_dense_grad
bash build.sh --pkg --experimental --soc=ascend310p --ops=embedding_dense_grad

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 胡豪杰 | 华中科技大学 | EmbeddingDenseGrad | 2026/07/21 | EmbeddingDenseGrad 算子适配开源仓 |
