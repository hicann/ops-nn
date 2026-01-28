# GroupNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 算子功能：GroupNorm用于计算输入张量的组归一化结果，均值，标准差的倒数，该算子是对GroupNorm的反向计算。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。
- 计算公式：
  
  $$
  d\beta = \sum_{i=1}^n dy
  $$

  $$
  d\gamma = \sum_{i=1}^n (dy \cdot \hat{x})
  $$
  
  $$
  dx = mean \cdot rstd \cdot \gamma \begin{bmatrix}
  dy - \frac{1}{N}  (d\beta + \hat{x} \cdot d\gamma)
  \end{bmatrix}
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
      <td>dy</td>
      <td>输入</td>
      <td>反向计算的梯度tensor，对应公式中的`dy`。数据类型与`x`相同。`dy`支持2-8维（N, C, *），计算逻辑仅关注前两个维度（N和C），其余维度可合并为一个维度。</td><!--N\*C\*HxW在IR原型中没有这些参数，看看么破-->
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>正向计算的第二个输出，表示`x`分组后每个组的均值，对应公式中的`mean`。 必须是2D（N, num_groups）。</td><!--N\*group在IR原型中没有这些参数，看看么破-->
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>正向计算的第三个输出，表示`x`分组后每个组的标准差倒数，对应公式中的`rstd`。数据类型与`mean`相同。 必须是2D（N, num_groups）。</td><!--N\*group在IR原型中没有这些参数，看看么破-->
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>正向计算的首个输入，对应公式中的`x`。数据类型与`dy`相同。支持2-8维（N, C, *），计算逻辑仅关注前两个维度（N和C），其余维度可合并为一个维度。</td><!--N\*C\*HxW在IR原型中没有这些参数，看看么破-->
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示每个channel的缩放系数，对应公式中的`γ`，数据类型与`mean`相同。必须是1D。`gamma`的值需要与`x`的C轴值一致。</td><!--C在IR原型中没有这些参数，看看么破-->
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>num_groups</td>
      <td>属性</td>
      <td>表示将输入`dy`的C维度分为group组，group需大于0。</td><!--C在IR原型中没有这些参数，看看么破-->
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td><ul><li>指定输出的`dx`的数据格式。</li><li>默认值为NCHW。</li></ul></td><!--aclnn和IR原型对该参数的解释不清楚，以上是我自己补充的，待确认-->
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx_is_require</td>
      <td>可选属性</td>
      <td><ul><li>是否输出`dx`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dgamma_is_require</td>
      <td>可选属性</td>
      <td><ul><li>是否输出`dgamma`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dbeta_is_require</td>
      <td>可选属性</td>
      <td><ul><li>是否输出`dbeta`。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dx</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新输入`x`的梯度，对应公式中的`dx`。数据类型与`dy`相同，shape与`x`相同。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dgamma</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新缩放参数的梯度，对应公式中的`dγ`。数据类型与`mean`相同，shape与`gamma`相同。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dbeta</td>
      <td>输出</td>
      <td>计算输出的梯度，用于更新偏置参数的梯度，对应公式中的`dβ`。数据类型与`mean`相同，shape与`gamma`相同。</td>
      <td>FLOAT16、FLOAT32、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>：输入和输出参数的数据类型不支持BFLOAT16。

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_group_norm_grad](examples/test_aclnn_group_norm_grad.cpp) | 通过[aclnnGroupNormBackward](docs/aclnnGroupNormBackward.md)接口方式调用GroupNormGrad算子。 |
| 图模式 | -  | 通过[算子IR](op_graph/group_norm_grad_proto.h)构图方式调用GroupNormGrad算子。         |