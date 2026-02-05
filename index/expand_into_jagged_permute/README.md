# ExpandIntoJaggedPermute

##  产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×   |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：将稀疏数据置换索引从表维度扩展到批次维度，适用于稀疏特征在不同rank中具有不同批次大小的情况。
- 计算公式：

$$
len = outputOffset[i+1] - outputOffset[i]
$$

$$
outputPermuteOut[outputOffset[i]:outputOffset[i+1]] = arange(inputOffset[permute[i]],inputOffset[permute[i]]+len)
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
      <td>inputOffset</td>
      <td>输入</td>
      <td>表示表级别长度的互斥偏移量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputOffset</td>
      <td>输入</td>
      <td>表示表级别置换长度的互斥偏移量。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>outputPermute</td>
      <td>输出</td>
      <td>表示公式中的输出。</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>outputSize</td>
      <td>属性</td>
      <td><ul><li>输出结果的长度。</li></td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明
inputOffset、outputOffset的长度比permute多1。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_expand_into_jagged_permute.cpp](tests/ut/op_kernel/test_expand_into_jagged_permute.cpp) | 通过[aclnnExpandIntoJaggedPermute](docs/aclnnExpandIntoJaggedPermute.md)接口方式调用ExpandIntoJaggedPermute算子。 |

