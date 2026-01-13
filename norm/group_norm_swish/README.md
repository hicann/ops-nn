# GroupNormSwish

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：计算输入x的组归一化结果out，均值meanOut，标准差的倒数rstdOut，以及Swish的输出。

- 计算公式：
  - **GroupNorm:**
    记 $E[x] = \bar{x}$代表$x$的均值，$Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$代表$x$的方差，则

    $$
    \left\{
    \begin{array} {rcl}
    yOut& &= \frac{x - E[x]}{\sqrt{Var[x] + eps}} * gamma + \beta \\
    meanOut& &= E[x]\\
    rstdOut& &= \frac{1}{\sqrt{Var[x] + eps}}\\
    \end{array}
    \right.
    $$

  - **Swish:**

    $$
    yOut = \frac{x}{1+e^{-scale * x}}
    $$

    当activateSwish为True时，会计算Swish， 此时Swish计算公式的x为GroupNorm公式得到的out。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>x</td>
      <td>输入</td>
      <td>计算公式中的x。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>组归一化中的 gamma 参数，`yOut`计算公式中的gamma。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>组归一化中的 beta 参数，`yOut`计算公式中的beta。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>numGroups</td>
      <td>属性</td>
      <td>输入$x$的第1维度分为group组。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dataFormatOptional</td>
      <td>属性</td>
      <td>当前版本只支持输入NCHW，Host侧的字符类型。</td>
      <td>char*</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps</td>
      <td>属性</td>
      <td>用于防止产生除0的偏移，`yOut`和`rstdOut`计算公式中的eps值。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateSwish</td>
      <td>属性</td>
      <td>表示是否支持Swish计算。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swishScale</td>
      <td>属性</td>
      <td>Swish计算时的scale值。</td>
      <td>DOUBLE</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>组归一化结果。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>x分组后的均值。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>x分组后的标准差的倒数。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_swish.cpp](examples/test_aclnn_group_norm_swish.cpp) | 通过[aclnnGroupNormSwishGrad](docs/aclnnGroupNormSwish.md)接口方式调用aclnnGroupNormSwish算子。 |