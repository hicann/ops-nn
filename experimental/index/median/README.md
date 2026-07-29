# Median

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Ascend 950PR/Ascend 950DT|×|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

- 算子功能：对标 `torch.median`，求中位数（下中位数）。支持两种形态：
  - **全局中位数**：将 `input` 全部元素视为一维后取下中位数，返回标量 `values`（不输出 indices）。
  - **按维中位数**：沿指定维度 `dim` 取下中位数，返回 `values` 及该中位数在原维度上首个出现的下标 `indices`。

- 计算说明：设沿规约维排序后的非降序序列为 `s`、长度为 `k`，则下中位数为 `s[(k-1)/2]`（`k` 为偶数时取较小者，与 NumPy 取均值的语义不同）；`indices` 为该中位数值在 `input` 沿规约维上首个出现的位置（对齐 `torch.median`）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>input</td>
      <td>输入</td>
      <td>待求中位数的输入张量。</td>
      <td>FLOAT16、FLOAT、INT32、BF16、INT64、INT16、UINT8、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>规约维度（按维中位数时使用）。</td>
      <td>int</td>
      <td>-</td>
    </tr>
    <tr>
      <td>keepdim</td>
      <td>属性</td>
      <td>是否保留规约维度（保留时该维长度为 1）。</td>
      <td>bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>values</td>
      <td>输出</td>
      <td>中位数值，dtype 与 input 相同。</td>
      <td>FLOAT16、FLOAT、INT32、BF16、INT64、INT16、UINT8、INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>中位数在规约维上的首个下标（仅按维中位数输出）。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- `dim` 规约维长度支持为 1（退化场景：values 即原值，indices 恒为 0）。
- 偶数长度取下中位数（排序后下标 `(k-1)/2`），与 NumPy 取均值不同。
- `indices` 取首个等值下标，对齐 `torch.median`。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/test_aclnn_median.cpp">test_aclnn_median</a></td>
    <td>参见<a href="../../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。接口说明见 <a href="./docs/aclnnMedian.md">aclnnMedian.md</a>。</td>
  </tr>
</tbody>
</table>

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| ---- | ---- | ---- | ---- | ---- |
| 开源社区贡献者 | 开源社区 | Median | 2026/06/30 | Median算子适配开源仓 |
