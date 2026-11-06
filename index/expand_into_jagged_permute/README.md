# ExpandIntoJaggedPermute

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √   |
|  <term>Atlas A3系列产品</term>  |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：将稀疏数据置换索引从表维度扩展到批次维度，适用于稀疏特征在不同rank中具有不同批次大小的情况。
- 计算公式：

$$
len = output_offsets[i+1] - output_offsets[i]
$$

$$
output_permute[output_offsets[i]:output_offsets[i+1]] = arange(input_offsets[permute[i]],input_offsets[permute[i]]+len)
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 100px">
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
      <td>permute</td>
      <td>输入</td>
      <td>表示表级别的置换索引。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_offsets</td>
      <td>输入</td>
      <td>表示表级别长度的互斥偏移量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_offsets</td>
      <td>输入</td>
      <td>表示表级别置换长度的互斥偏移量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_permute</td>
      <td>输出</td>
      <td>表示公式中的输出。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>output_size</td>
      <td>属性</td>
      <td><ul><li>输出结果的长度。</li></ul></td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

设P为permute的长度，input_offsets和output_offsets的长度均为P+1。
P=0时output_size必须为0；P>0时output_size必须大于等于P。

input_offsets和output_offsets必须从0开始，P>0时严格单调递增。
input_offsets的末值等于输入侧总元素数。
output_offsets的末值必须等于output_size。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_expand_into_jagged_permute.cpp](examples/test_aclnn_expand_into_jagged_permute.cpp) | 通过[aclnnExpandIntoJaggedPermute](docs/aclnnExpandIntoJaggedPermute.md)接口方式调用ExpandIntoJaggedPermute算子。 |
