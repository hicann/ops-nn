# DeepNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。
- 计算公式：

  $$
  DeepNorm(x_i^{\prime}) = (\frac{x_i^{\prime} - \bar{x^{\prime}}}{rstd}) * gamma + beta,
  $$
  $$
  \text { where } rstd = \sqrt{\frac{1}{n} \sum_{i=1}^n (x^{\prime}_i - \bar{x^{\prime}})^2 + eps} , \quad \operatorname{x^{\prime}_i} = alpha * x_i   + gx_i
  $$

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
      <td>输入数据，通常为神经网络的中间层输出，公式中的输入`x`。</td>
      <ul><li>输入数据，通常为神经网络的中间层输出，公式中的输入`x`。</li><li>shape支持2-8维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>输入</td>
      <td>输入数据的梯度，用于反向传播，公式中的输入`gx`。</td><ul><li>输入数据的梯度，用于反向传播，公式中的输入`gx`。</li><li>数据类型与输入`x`的保持一致。</li></ul>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>偏置参数，用于调整归一化后的输出，公式中的输入`beta`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>缩放参数，用于调整归一化后的输出，公式中的输入`gamma`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>可选属性</td>
      <td><ul><li>权重参数，用于调整输入x的权重，公式中的输入`alpha`。</li><li>默认值为0.3f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到分母中的值，以确保数值稳定，用于防止除0错误，对应公式中的eps。</li><li>默认值为1e-06f。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输出</td>
      <td>计算输出的均值，用于归一化操作，公式中的输出`mean`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输出</td>
      <td>计算输出的标准差，用于归一化操作，公式中的输出`rstd`。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>归一化后的输出数据。Device侧的aclTensor，公式中的输出`y`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>


## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_deep_norm](examples/test_aclnn_deep_norm.cpp) | 通过[aclnnDeepNorm](docs/aclnnDeepNorm.md)接口方式调用DeepNorm算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/deep_norm_proto.h)构图方式调用DeepNorm算子。         |

<!--[test_geir_deep_norm](examples/test_geir_deep_norm.cpp)-->