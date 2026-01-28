# Swish

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：Swish激活函数，对输入Tensor逐元素进行Swish函数运算并输出结果Tensor。

- 计算公式：

  $$
  s(x) = x*\sigma(\beta x)
  $$

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  其中$\sigma(x)$为sigmoid函数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 230px">
  <col style="width: 240px">
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
      <td>表示用于计算激活函数的张量，公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>self的shape和数据类型与out的一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>betaOptional</td>
      <td>输入</td>
      <td>表示可调节参数，用于控制Swish函数的形状和斜率的标量，公式中的β。</td>
      <td><ul><li>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</li><li>当betaOptional为空指针时，接口以1.0进行计算。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示Swish函数的输出，公式中的s(x)。</td>
      <td><ul><li>支持空Tensor。</li><li>out的shape和数据类型与self的一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

- <term>Atlas 推理系列产品</term>、<term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码                                                                         | 说明                                               |
| ---------------- |------------------------------------------------------------------------------|--------------------------------------------------|
| aclnn接口  | [test_aclnn_swish.cpp](examples/test_aclnn_swish.cpp) | 通过[aclnnSwish](docs/aclnnSwish.md)接口方式调用Swish算子。 |