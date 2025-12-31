# EluGradV2

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 算子功能：激活函数的反向计算，输出ELU激活函数正向输入的梯度。

- 计算公式：

    - 当is_result是True时：

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast (x + \alpha \ast scale),  \quad x \leq 0
      \end{cases}
      $$

    - 当is_result是False时：

      $$
      gradInput = gradOutput *
      \begin{cases}
      scale, \quad x > 0\\
      inputScale \ast \alpha \ast scale \ast exp(x \ast inputScale), \quad x \leq 0
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
      <td>grads</td>
      <td>输入</td>
      <td>公式中的gradOutput</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activations</td>
      <td>输入</td>
      <td>公式中的x</td>
      <td>FLOAT,FLOAT16,BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示ELU激活函数的激活系数，公式中的alpha。如果is_result为true，alpha必须大于等于0。数据类型需要是可转换为FLOAT的数据类型。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>输入</td>
      <td>表示ELU激活函数的缩放系数，公式中的scale。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>输入</td>
      <td>表示ELU激活函数的输入的缩放系数，公式中的inputScale。数据类型需要是可转换为FLOAT的数据类型。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_result</td>
      <td>输入</td>
      <td>表示传给ELU反向计算的输入是否是ELU正向的输出。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>公式中的输出张量gradInput</td>
      <td>同grads</td>
      <td>ND</td>
    </tr>
  </tbody></table>



## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_elu_backward](examples/arch35/test_aclnn_elu_backward.cpp) | 通过[aclnnEluBackward](docs/aclnnEluBackward.md)接口方式调用算子。 |