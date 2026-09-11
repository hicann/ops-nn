# KthValue

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：返回输入张量x在指定维度dim上的第k个最小值及其在原输入中的索引。
- 计算流程：
  对沿dim维度取出的每个一维切片，设升序排列后的值为s，p[j]表示s[j]在该切片排序前的位置（从0开始计数），则：

  $$y = s[k - 1], \quad indices = p[k - 1]$$

  例如输入张量x=[[3, 1, 2], [6, 4, 5]]，k=2，dim=1，得到y=[[2], [5]]，indices=[[2], [2]]。其中2和5在各自原始行中的索引均为2。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待计算的输入张量。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>第k个最小值。数据类型与x相同，shape与x相同但dim维度的长度为1。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、UINT16、UINT32、UINT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>输出</td>
      <td>选中元素在输入张量dim维度上的索引，从0开始。shape与y相同。</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>k</td>
      <td>必选属性</td>
      <td>指定第k个最小值，取值范围为[1, x.shape[dim]]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>可选属性</td>
      <td><ul><li>指定计算的维度，取值范围为[-rank(x), rank(x) - 1]。</li><li>默认值为-1，表示最后一维。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
  </tbody>
</table>

## 约束说明

输入张量x不能为空，且维度数必须大于0。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_kthvalue](./examples/test_aclnn_kthvalue.cpp) | 通过[aclnnKthvalue](./docs/aclnnKthvalue.md)接口在指定维度上计算第k个最小值及索引。 |
