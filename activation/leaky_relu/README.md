# LeakyRelu

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明

- 算子功能：激活函数，用于解决Relu函数在输入小于0时输出为0的问题，避免神经元无法更新参数。
- 计算公式：

  $$
  out = max(0,self) + negativeSlope * min(0,self)
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1310px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 200px">
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
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>self</td>
      <td>输入</td>
      <td>待进行LeakyRelu激活函数的入参，公式中的self。</td>
      <td><ul><li>shape支持0到8维，shape需要与out一致。</li><li>支持空Tensor。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>negativeSlope</td>
      <td>输入</td>
      <td>表示self < 0时的斜率，公式中的negativeSlope。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>out</td>
      <td>输出</td>
      <td>待进行LeakyRelu激活函数的出参。</td>
      <td>out的数据类型需要是self可转换的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、DOUBLE</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas 训练系列产品</term>：FLOAT、FLOAT16、DOUBLE。

## 约束说明

- 确定性计算：
    - aclnnLeakyRelu&aclnnInplaceLeakyRelu默认确定性实现。

- negativeSlope使用整型类型作为属性输入，而输入self是FLOAT类型，那么如果negativeSlope大于2^24或小于-2^24可能存在精度损失。同理，如果输入self是FLOAT16类型，那么negativeSlope大于2^11或小于-2^11可能存在精度损失。


## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                                                                 |
| ---------------- |------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_leaky_relu.cpp](examples/test_aclnn_leaky_relu.cpp) | 通过[aclnnLeakyRelu](docs/aclnnLeakyRelu&aclnnInplaceLeakyRelu.md)接口方式调用LeakyRelu算子。 |