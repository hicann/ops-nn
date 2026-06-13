# HardShrink

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：完成HardShrink激活函数计算，将输入张量中绝对值小于等于阈值lambd的元素置零，大于阈值的元素保持不变。

- 计算公式：

$$
\text{HardShrink}(x) = \begin{cases} x, & \text{if } x > \lambda \\ x, & \text{if } x < -\lambda \\ 0, & \text{otherwise} \end{cases}
$$

其中，x为输入张量，$\lambda$为阈值参数lambd。

## 参数说明

<table style="table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
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
      <td>self</td>
      <td>输入</td>
      <td>输入张量，对应公式中的x。支持0-8维，支持空Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>输入</td>
      <td>阈值参数，对应公式中的 λ，float类型标量，默认值0.5。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，与self同shape同dtype。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- self与out的数据类型必须一致，支持FLOAT、FLOAT16、BFLOAT16。
- self与out的shape必须一致，不涉及广播。
- self支持0-8维。
- self支持空Tensor（0元素），此时out也为空Tensor，不执行计算。
- lambd为float类型标量，取值无限制。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn调用</td>
    <td><a href="./examples/arch35/test_aclnn_hard_shrink.cpp">test_aclnn_hard_shrink</a></td>
    <td>通过aclnn两段式接口调用，详见 <a href="./docs/aclnnHardshrink.md">aclnnHardshrink接口文档</a>。</td>
  </tr>
</tbody>
</table>
