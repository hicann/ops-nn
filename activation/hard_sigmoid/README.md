# HardSigmoid

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：激活函数，对输入张量self逐元素进行HardSigmoid变换，输出与输入shape相同的张量。
- 计算公式：

  $$
  HardSigmoid(self)=\begin{cases}
  &1, &if(self\gt3) \\
  &0, &if(self\le-3) \\
  &\frac{self}{6} + \frac{1}{2}, &otherwise
  \end{cases}
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 800px"><colgroup>
  <col style="width: 110px">
  <col style="width: 130px">
  <col style="width: 300px">
  <col style="width: 180px">
  <col style="width: 80px">
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
      <td>self</td>
      <td>输入</td>
      <td>支持空Tensor。表示激活函数的输入，公式中的输入self。shape维度不超过8维。</td>
      <td>FLOAT、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>支持空Tensor。表示激活函数的输出，公式中的输出HardSigmoid(self)。数据类型与shape均与输入self一致。</td>
      <td>FLOAT、FLOAT16、INT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas 训练系列产品</term>：数据类型支持FLOAT、FLOAT16、INT32。

## 约束说明

无。

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_hard_sigmoid.cpp](examples/test_aclnn_hard_sigmoid.cpp) | 通过[aclnnHardsigmoid](docs/aclnnHardsigmoid&aclnnInplaceHardsigmoid.md)接口方式调用HardSigmoid算子。 |
| aclnn接口  | [test_aclnn_inplace_hard_sigmoid.cpp](examples/test_aclnn_inplace_hard_sigmoid.cpp) | 通过[aclnnInplaceHardsigmoid](docs/aclnnHardsigmoid&aclnnInplaceHardsigmoid.md)接口方式调用HardSigmoid算子。 |
