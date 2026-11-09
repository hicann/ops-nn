# HardShrinkGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>   |     √    |
|  <term>Atlas A2系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：[HardShrink](../hard_shrink/README.md)的反向算子，计算反向传播的梯度gradInput。

- 计算公式：

  $$
  HardshrinkBackward(x,grad)=
  \begin{cases}
  &0, &if(-\lambda \leq x \leq \lambda) \\
  &grad, &otherwise
  \end{cases}
  $$

  当x为NaN时，对应位置输出grad值；当grad为NaN且|x|>λ时，对应位置输出NaN。

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
      <td>gradients</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，作为本反向算子的输入，公式中的grad。非连续Tensor输入由框架自动转换为连续Tensor后执行。支持空Tensor（元素数为0）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>表示输入的Tensor，公式中的x。非连续Tensor输入由框架自动转换为连续Tensor后执行。支持空Tensor（元素数为0）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>反向传播梯度输出，公式中的HardshrinkBackward(x,grad)。非连续Tensor输出由框架自动转换为连续Tensor后执行。支持空Tensor；输入为空Tensor时，gradInput同为空Tensor，不执行计算。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
   </tr>
    <tr>
      <td>lambd</td>
      <td>属性</td>
      <td>公式中的λ，阈值参数，默认值0.5，取值无限制。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  - <term>Atlas训练系列产品</term>：数据类型支持FLOAT、FLOAT16。

## 约束说明

- GE图模式：gradients、features与backprops的shape必须一致；gradients、features与backprops的数据类型必须一致。

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_hard_shrink_backward.cpp](examples/test_aclnn_hard_shrink_backward.cpp) | 通过[aclnnHardshrinkBackward](docs/aclnnHardshrinkBackward.md)接口方式调用HardShrinkGrad算子。 |
| GE图模式 | - | 通过[算子IR](./op_graph/hard_shrink_grad_proto.h)构图方式调用HardShrinkGrad算子。 |
