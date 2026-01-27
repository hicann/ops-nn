# SiluGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 接口功能：[aclnnSilu](../../swish/docs/aclnnSilu.md)的反向传播，根据silu反向传播梯度与正向输出计算silu的梯度输入。
- 计算公式：

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  $$
  s(x) = x\sigma(x)
  $$

  $$
  s^\prime(x) = \sigma(x)(1+x-x\sigma(x))
  $$

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

其中$\sigma(x)$为sigmoid函数，$s(x)$为silu函数，$s^\prime(x)$为silu函数的导数。

## 参数说明

  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 280px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示输入梯度。公式中的gradOutput。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、self与gradInput的数据类型和shape一致。</li><li>gradOutput、self与gradInput的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示输入数据。公式中的x，且对应正向的输入参数。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、self与gradInput的数据类型和shape一致。</li><li>gradOutput、self与gradInput的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示对输入数据self求的梯度。公式中的gradInput。</td>
      <td><ul><li>gradOutput、self与gradInput的数据类型和shape一致。</li><li>gradOutput、self与gradInput的shape满足<a href="../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
   </tbody>
  </table>

- <term>Atlas 训练系列产品</term>：数据类型支持FLOAT16、FLOAT。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                                                |
| ---------------- | --------------------------- |-------------------------------------------------------------------|
| aclnn接口  | [test_aclnn_silu_grad.cpp](examples/test_aclnn_silu_grad.cpp) | 通过[aclnnSiluBackward](docs/aclnnSiluBackward.md)接口方式调用SiluGrad算子。 |