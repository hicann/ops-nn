# SwishGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term> |    √     |

## 功能说明

- 算子功能：Swish激活函数的反向传播，用于计算Swish激活函数的梯度。  
- 计算公式：
  
  - Swish函数公式

  $$
  s(x) = x*\sigma(\beta x)
  $$

  - Swish函数公式的导数实现

  $$

  s^\prime(x)= \beta s(x)+\sigma(\beta x)(1-\beta s(x))= \sigma(\beta x)*(1+\beta x(1-\sigma(\beta x)))

  $$    

  $$
  gradInput = gradOutput * s^\prime(x)
  $$

  $$
  \sigma(x) = {\frac{1} {1+{e}^{-x}}}
  $$

  其中$\sigma(x)$为Sigmoid函数，$s(x)$为Swish函数，$s^\prime(x)$为Swish函数的导数。

## 参数说明

 <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 230px">
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
      <td>表示Swish激活函数正向输出的梯度，公式中的gradOutput。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、self与gradInput的shape一致。</li><li>gradOutput、self与gradInput的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示用于计算激活函数的张量，公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、self与gradInput的shape一致。</li><li>gradOutput、self与gradInput的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>betaOptional</td>
      <td>输入</td>
      <td>表示可调节参数，用于控制Swish函数的形状和斜率的标量，公式中的β。</td>
      <td><ul><li>数据类型需要是可转换为FLOAT的数据类型（参见<a href="../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li><li>当betaOptional为空指针时，接口以1.0进行计算。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td><ul><li>backward计算的输出，为Swish正向输入的梯度值，即对输入进行求导后的结果。</li><li>公式中的gradInput。</li></ul></td>
      <td><ul><li>支持空Tensor。</li><li>gradOutput、self与gradInput的shape一致。</li><li>gradOutput、self与gradInput的数据类型一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
  </tbody>
  </table>

## 约束说明
无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_swish_grad](examples/test_aclnn_swish_grad.cpp) | 通过[aclnnSwishBackward](docs/aclnnSwishBackward.md)接口方式调用CtclossV2算子。 |