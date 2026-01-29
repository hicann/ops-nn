# ReluGradV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/relu_grad_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品 </term>    |     √    |
| <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对应Relu操作的反向传播梯度。

- 计算公式：
      $$
      gradients = backprops *
      \begin{cases}
      1, \quad mask > 0\\
      0,  \quad mask \leq 0
      \end{cases}
      $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 820px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 190px">
  <col style="width: 260px">
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
      <td>gradients</td>
      <td>输入</td>
      <td>传递给对应Relu操作的反向传播梯度</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mask</td>
      <td>输入</td>
      <td>作为输入传递给对应ReluV2操作的特征</td>
      <td>UINT1</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>公式中的输出张量</td>
      <td>BFLOAT16、FLOAT16、FLOAT、INT8、INT32、UINT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 调用说明

| 调用方式   | 样例代码           | 说明                                            |
| ---------------- | --------------------------- |-----------------------------------------------|
| 图模式 | [test_geir_relu_grad_v2.cpp](./examples/test_geir_relu_grad_v2.cpp)  | 通过[算子IR](./op_graph/relu_grad_v2_proto.h)构图方式调用ReluGradV2算子。 |
