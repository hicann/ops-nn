# aclnnMedian

## 支持的产品型号

- Atlas A2 训练系列产品 / Atlas A3 系列产品。

## 接口原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnMedianGetWorkspaceSize` 接口获取入参并根据计算流程计算所需 workspace 大小，再调用 `aclnnMedian` 接口执行计算。

- `aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor)`
- `aclnnStatus aclnnMedian(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## 功能描述

- **算子功能**：对标 `torch.median`，求中位数（下中位数）。
  - 当 `indicesOut == nullptr` 时为**全局中位数**：将 `self` 拍平为一维后取下中位数，仅输出 `valuesOut`。
  - 当 `indicesOut != nullptr` 时为**按维中位数**：沿 `dim` 维取下中位数，输出 `valuesOut` 及该中位数在该维上首个出现的下标 `indicesOut`。
- **计算公式**：设沿规约维排序后的非降序序列为 $s$、长度为 $k$，则

  $$
  values = s_{\lfloor (k-1)/2 \rfloor}
  $$

  即偶数长度取较小的中间值（与 NumPy 取均值不同）；`indices` 为该值在 `self` 沿规约维首个出现的位置。

## aclnnMedianGetWorkspaceSize 参数说明

<table>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th></tr></thead>
  <tbody>
    <tr><td>self</td><td>输入</td><td>待求中位数的输入张量。</td><td>支持非连续 Tensor。</td><td>FLOAT16、FLOAT、INT32、BF16、INT64、INT16、UINT8、INT8</td><td>ND</td></tr>
    <tr><td>dim</td><td>输入</td><td>规约维度。</td><td>取值范围 [-self.dimNum, self.dimNum)；全局中位数时该参数被忽略。</td><td>INT64</td><td>-</td></tr>
    <tr><td>keepDim</td><td>输入</td><td>是否保留规约维度。</td><td>为 true 时 valuesOut/indicesOut 在 dim 维长度为 1。</td><td>BOOL</td><td>-</td></tr>
    <tr><td>valuesOut</td><td>输出</td><td>中位数值。</td><td>dtype 须与 self 一致；shape 为 self 去掉 dim 维（keepDim 时该维为 1）。</td><td>同 self</td><td>ND</td></tr>
    <tr><td>indicesOut</td><td>输出</td><td>中位数下标；为 nullptr 时表示全局中位数。</td><td>shape 同 valuesOut。</td><td>INT32、INT64</td><td>ND</td></tr>
    <tr><td>workspaceSize</td><td>输出</td><td>返回需要在 Device 侧申请的 workspace 大小。</td><td>-</td><td>-</td><td>-</td></tr>
    <tr><td>executor</td><td>输出</td><td>返回 op 执行器。</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody>
</table>

## aclnnMedian 参数说明

<table>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在 Device 侧申请的 workspace 内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>workspace 大小，由 aclnnMedianGetWorkspaceSize 获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op 执行器，由 aclnnMedianGetWorkspaceSize 获取。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的 stream。</td></tr>
  </tbody>
</table>

## 返回值

返回 `aclnnStatus` 状态码，常见取值：

| 返回码 | 说明 |
| --- | --- |
| ACLNN_SUCCESS | 执行成功。 |
| ACLNN_ERR_PARAM_NULLPTR | self / valuesOut / workspaceSize / executor 为空指针。 |
| ACLNN_ERR_PARAM_INVALID | self 数据类型不在支持范围；valuesOut 与 self dtype 不一致；indicesOut dtype 非 INT32/INT64；self 为空张量（不支持空输入，行为对齐 PyTorch）。 |

## 约束说明

- `self` 与 `valuesOut` 数据类型须一致；`indicesOut` 仅支持 INT32 / INT64。
- 不支持空张量（`self` 元素个数为 0 时返回 ACLNN_ERR_PARAM_INVALID）。
- `dim` 维长度支持为 1（退化场景：values 即原值，indices 恒为 0）。

## 调用示例

详见 [test_aclnn_median.cpp](../examples/test_aclnn_median.cpp)。
