# SegmentSum

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term> Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：对输入tensor按分段索引求和。
- 计算公式：
    $$
    y[i] = \sum_{\substack{j\\ \text{segment\_ids}[j] = i}} x[j]
    $$
    其中，求和遍历所有满足 `segment_ids[j] == i` 的索引 `j`，将对应的 `x[j]` 累加到 `y[i]`。若某个段 `i` 没有对应的元素，则 `y[i] = 0`。
- 用例：

  输入tensor $x = \begin{bmatrix} [1 & 2] \\ [3 & 4] \\ [5 & 6] \\ [7 & 8] \end{bmatrix}$，
  分段索引tensor $segment\_ids = [0, 0, 1, 2]$，

  输出tensor $y = \begin{bmatrix} [4 & 6] \\ [5 & 6] \\ [7 & 8] \end{bmatrix}$
  - `segment_ids`必须按升序排序
  - `segment_ids`为分段索引，指示当前分段的值归属于哪个段
  - `segment_ids`值必须 >= 0
  - 输出shape为 `[max(segment_ids) + 1, x.shape[1:]]`

## 参数说明

| 参数名        | 输入/输出/属性 | 描述         | 数据类型                                                       | 数据格式 |
| ------------- | -------------- | ------------ | -------------------------------------------------------------- | -------- |
| x             | 输入           | 输入数据，即公式中的 `x`             | FLOAT32、FLOAT16、BFLOAT16、INT32、INT64、UINT32、UINT64       | ND       |
| segment_ids   | 输入           | 分段索引，即公式中的 `segment_ids`  | INT32、INT64                                                   | ND       |
| y             | 输出           | 输出值信息，即公式中的 `y`           | FLOAT32、FLOAT16、BFLOAT16、INT32、INT64、UINT32、UINT64       | ND       |

## 约束说明

**x**：

- 维度至少 1（rank >=1）。

**segment_ids**：

- 必须是 INT32 或 INT64 类型。
- 必须为 1D tensor，且 `segment_ids.shape[0] = x.shape[0]`。
- 值必须按升序排序，且 `segment_ids.value >= 0`。

**y**：

- 类型必须与 x 相同。
- 维度与 x 相同，shape 为 `[max(segment_ids) + 1, x.shape[1:]]`。

## 调用说明

| 调用方式   | 调用样例           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE图模式 | [test_geir_segment_sum](examples/test_geir_segment_sum.cpp) | 通过[算子IR](op_graph/segment_sum_proto.h)构图方式调用SegmentSum算子。 |
