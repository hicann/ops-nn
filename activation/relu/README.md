# Relu

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：激活函数，返回与输入tensor shape相同的tensor，tensor中value大于等于0时，取该value，小于0，取0。
- 计算公式：
$$
relu(self) = \begin{cases} self, & self\gt 0 \\ 0, & self\le 0 \end{cases}
$$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1305px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 247px">
  <col style="width: 108px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>待进行Relu计算的入参。</td>
      <td>FLOAT、FLOAT16、INT8、INT32、INT64、BFLOAT16、UINT8</td>
      <td>ND</td>
  <tr>
      <td>y</td>
      <td>输出</td>
      <td>进行Relu计算后的输出</td>
      <td>FLOAT、FLOAT16、INT8、INT32、INT64、BFLOAT16、UINT8</td>
      <td>ND</td>
  </table>

## 约束说明

- 确定性计算：
  - aclnnRelu&aclnnInplaceRelu默认确定性实现。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_relu](./examples/test_aclnn_relu.cpp) | 通过[aclnnRelu](./docs/aclnnRelu.md)接口方式调用Relu算子。 |
| aclnn调用 | [test_aclnn_inplace_relu](./examples/test_aclnn_inplace_relu.cpp) | 通过[aclnnInplaceRelu](./docs/aclnnInplaceRelu.md)接口方式调用Relu算子。 |