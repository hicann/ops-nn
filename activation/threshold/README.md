# Threshold

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：对输入x进行阈值操作。当x中的elements大于threshold时，返回1；否则，返回0。

- 计算公式：

$$
y = (x > threshold) \; ? \; 1 : 0
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
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
      <td>x</td>
      <td>输入</td>
      <td>待进行threshold计算的入参，公式中的x。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>threshold计算后的出参，公式中的y。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>threshold</td>
      <td>属性</td>
      <td>阈值，x中大于此值的元素输出1，否则输出0。默认值为0.0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入输出为任意维度张量（ND格式）。
- 输入输出数据类型必须一致。
- 输出值为离散值（0.0或1.0），与Golden结果bit-exact一致。
- NaN输入输出0.0（NaN 与任意值比较恒为False）。
- +Inf输入输出1.0（+Inf > threshold 恒为True）。
- -Inf输入输出0.0（-Inf > threshold 恒为False）。

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
    <td><a href="./examples/test_aclnn_threshold.cpp">test_aclnn_threshold</a></td>
    <td rowspan="2">参见<a href="../../docs/zh/invocation/quick_op_invocation.md">算子调用</a>完成算子编译和验证。</td>
  </tr>
  <tr>
    <td>图模式调用</td>
    <td><a href="./examples/test_geir_threshold.cpp">test_geir_threshold</a></td>
  </tr>
</tbody>
</table>
