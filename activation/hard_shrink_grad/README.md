# HardShrinkGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
| <term>Atlas 推理系列产品</term>    |     ×    |
| <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：[HardShrink](../hard_shrink/README.md)的反向算子，计算反向传播的梯度gradInput。

- 计算公式：

  $$
  HardshrinkBackward(x,grad)=
  \begin{cases}
  &grad, &if(x > \lambda) \\
  &grad, &if(x < -\lambda) \\
  &0, &otherwise
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
      <td>gradients</td>
      <td>输入</td>
      <td>反向传播过程中上一步输出的梯度，作为本反向算子的输入，公式中的grad。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>features</td>
      <td>输入</td>
      <td>表示输入的Tensor，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>输入</td>
      <td>公式中的λ，aclScalar类型。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>backprops</td>
      <td>输出</td>
      <td>反向传播梯度输出，公式中的HardshrinkBackward(x,grad)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
   </tr>
  </tbody>
  </table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_hard_shrink_backward.cpp](examples/test_aclnn_hard_shrink_backward.cpp) | 通过[aclnnHardshrinkBackward](docs/aclnnHardshrinkBackward.md)接口方式调用HardShrinkGrad算子。 |
