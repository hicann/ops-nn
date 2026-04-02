# Renorm

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：返回一个张量，其中输入张量self沿维度dim的每个子张量都经过归一化，使得子张量的p范数低于maxNorm值。

- 计算公式：

  $$
  output_i=\left\{
  \begin{aligned}
  input_i,\quad ||input_i||_p <= maxNorm \\
  \frac {input_i} {||input_i||_p} \cdot maxNorm,\quad ||input_i||_p>maxNorm
  \end{aligned}
  \right.
  $$

  其中：
  $i$为dim确定的某维度张量切片：

  $$
  ||input_i||_p = (\sum_{i=0}^{n}{input_i^p}^\frac{1}{p})
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
      <td>表示进行重归一化计算的输入，对应公式中的`input`。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>p</td>
      <td>属性</td>
      <td>表示范数，对应公式中的`p`。取值大于等于0。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td>表示指定求norm的维度方向。对应公式中的`i`。取值范围为：[-x的维度数量，x的维度数量-1]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxNorm</td>
      <td>属性</td>
      <td>表示最大允许的归一化值。对应公式中的`maxNorm`。取值大于等于0。如果运算时对应维度的`p`范数（由`p`值确定）大于`maxNorm`，则将该维度的值关于`p`范数归一化并乘上`maxNorm`。如果运算时对应维度的`p`范数（由`p`值确定）小于`maxNorm`，则该维度张量保持不变输出。</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示最终输出，对应公式中的`output`。数据类型、shape与入参`x`保持一致。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_renorm](examples/test_aclnn_renorm.cpp) | 通过[aclnnRenorm](docs/aclnnRenorm&aclnnInplaceRenorm.md)接口方式调用Renorm算子。 |
| aclnn接口  | [test_aclnn_inplace_renorm](examples/test_aclnn_inplace_renorm.cpp) | 通过[aclnnInplaceRenorm](docs/aclnnRenorm&aclnnInplaceRenorm.md)接口方式调用Renorm算子。 |
