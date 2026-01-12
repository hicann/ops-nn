# SoftmaxCrossEntropyWithLogits

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
| <term>Ascend 950PR/Ascend 950DT</term>       |    √     |

## 功能说明

- 接口功能：计算softmax和cross entropy的交叉熵损失，并给出对输入logits的反向梯度。
- 计算公式：
  
  $$
  \text{loss} = - \sum_i y_i \cdot \log(\text{softmax}(x_i))
  $$

$$
\text{backprop} = \text{softmax}(x_i) - y_i
$$

其中，$x_i$对应输入的features，$y_i$对应输入的labels。

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
      <th>输入/输出</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>features</td>
      <td>输入</td>
      <td>表示模型的输出logits，公式中x_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>labels</td>
      <td>输入</td>
      <td>表示输入的labels，公式中y_i。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>loss</td>
      <td>输出</td>
      <td>表示softmax和cross entropy的交叉熵损失，公式中loss。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
      <tr>
      <td>backprop</td>
      <td>输出</td>
      <td>表示输入logits的反向梯度，公式中backprop。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
  </table>

## 调用说明
  
  | 调用方式   | 样例代码           | 说明                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn接口 | [test_aclnn_softmax_cross_entropy_with_logits](examples/test_aclnn_softmax_cross_entropy_with_logits.cpp) | 通过[aclnnSoftmaxCrossEntropyWithLogits](docs/aclnnSoftmaxCrossEntropyWithLogits.md)接口方式调用SoftmaxCrossEntropyWithLogits算子。 |