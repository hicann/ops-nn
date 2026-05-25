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

- 算子功能：完成 HardShrink 激活函数计算，将输入张量中绝对值小于等于阈值 lambd 的元素置零，大于阈值的元素保持不变。

- 计算公式：

$$
\text{HardShrink}(x) = \begin{cases} x, & \text{if } x > \lambda \\ x, & \text{if } x < -\lambda \\ 0, & \text{otherwise} \end{cases}
$$

其中，x 为输入张量，$\lambda$ 为阈值参数 lambd。


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
      <td>输入张量，对应公式中的 x。支持 0-8 维，支持空 Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lambd</td>
      <td>输入</td>
      <td>阈值参数，对应公式中的 λ，float 类型标量，默认值 0.5。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>输出张量，与 self 同 shape 同 dtype。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- self 与 out 的数据类型必须一致，支持 FLOAT、FLOAT16、BFLOAT16。
- self 与 out 的 shape 必须一致，不涉及广播。
- self 支持 0-8 维。
- self 支持空 Tensor（0 元素），此时 out 也为空 Tensor，不执行计算。
- lambd 为 float 类型标量，取值无限制。

## 调用说明

<table><thead>
  <tr>
    <th>调用方式</th>
    <th>调用样例</th>
    <th>说明</th>
  </tr></thead>
<tbody>
  <tr>
    <td>aclnn 调用</td>
    <td><a href="./examples/arch35/test_aclnn_hard_shrink.cpp">test_aclnn_hard_shrink</a></td>
    <td>通过 aclnn 两段式接口调用，详见 <a href="./docs/aclnnHardshrink.md">aclnnHardshrink 接口文档</a>。</td>
  </tr>
</tbody>
</table>
