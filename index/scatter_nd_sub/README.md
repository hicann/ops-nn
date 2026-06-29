# ScatterNdSub

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                     |     √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>    |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>    |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                               |    ×     |
| <term>Atlas 训练系列产品</term>                               |    ×     |

## 功能说明

- 算子功能：ScatterNdSub根据indices指定的索引位置，将updates中的值从var中逐元素减去，并将结果写回var（in-place）。

- 计算公式：

$$
\text{output}[\text{indices}[i]] = \text{var}[\text{indices}[i]] - \text{updates}[i]
$$

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
      <td>var</td>
      <td>输入</td>
      <td>待操作的参数张量。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输入</td>
      <td>索引张量，指定需要减去updates的位置。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>输入</td>
      <td>要减去的值，shape与indices的前缀维度匹配。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输出</td>
      <td>更新后的参数张量，与输入var同shape同dtype。</td>
      <td>FLOAT16、FLOAT、INT32、INT8、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否使用锁保护变量更新（NPU上未实现锁机制），默认false。</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- var的rank必须大于等于indices的最后一维。
- updates的shape必须与indices.shape[:-1] + var.shape[indices.shape[-1]:]一致。
- indices值不支持越界。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :--- | :--- | :--- |
| 图模式调用 | [test_geir_scatter_nd_sub](./examples/test_geir_scatter_nd_sub.cpp) | 在host侧通过GE IR构图方式调用scatter_nd_sub算子。 |
