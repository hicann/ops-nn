# Sigmoid

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：对输入Tensor完成Sigmoid运算。

- 计算公式：

$$
out = {\frac{1} {1+{e}^{-input}}}
$$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1393px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <td>待进行Sigmoid计算的入参，公式中的input。</td>
      <td>FLOAT、FLOAT16、DOUBLE、INT8、INT16、INT32、INT64、UINT8、BOOL、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算的出参。</td>
      <td>FLOAT、FLOAT16、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
      <td>ND</td>
  </tbody>
  </table>



## 约束说明

无

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn调用 | [test_aclnn_sigmoid](./examples/test_aclnn_sigmoid.cpp) | 通过[aclnnSigmoid](./docs/aclnnSigmoid.md)接口方式调用Sigmoid算子。 |
| aclnn调用 | [test_aclnn_inplace_sigmoid](./examples/test_aclnn_inplace_sigmoid.cpp) | 通过[aclnnInplaceSigmoid](./docs/aclnnInplaceSigmoid.md)接口方式调用Sigmoid算子。 |