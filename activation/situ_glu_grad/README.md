# SituGluGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 算子功能：SiTU门控线性单元（SiTU Gated Linear Unit）激活函数的反向梯度计算。根据上游梯度grad_y和前向输入x，计算输入x的梯度grad_x。

- 计算公式或计算流程：

  对给定的输入张量x（维度为[..., cut]，cut为偶数，令h = cut // 2）和上游梯度grad_y（维度为[..., h]）：

  1. 根据activate_left切分x为gate和up（各shape [..., h]）。

  2. 计算前向中间量：

     $$
     t = \tanh\left(\frac{gate}{\beta}\right), \quad s = \sigma(gate), \quad situ\_a = \beta \cdot t \cdot s
     $$

     当linear_beta > 0时：$up' = linear\_beta \cdot \tanh(up / linear\_beta)$，否则$up' = up$。

  3. 乘积法则：

     $$
     grad\_situ\_a = grad\_y \cdot up', \quad grad\_up' = grad\_y \cdot situ\_a
     $$

  4. gate梯度：

     $$
     grad\_gate = grad\_situ\_a \cdot s \cdot \left[ (1 - t^2) + \beta \cdot t \cdot (1 - s) \right]
     $$

  5. up梯度：

     - linear_beta > 0时：$grad\_up = grad\_up' \cdot (1 - \tanh^2(up / linear\_beta))$
     - linear_beta ≤ 0时：$grad\_up = grad\_up'$

  6. 根据activate_left将grad_gate、grad_up拼接为grad_x（shape与x一致）。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1576px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>grad_y</td>
      <td>输入</td>
      <td>反向传播上游梯度，dim维度大小为x的一半。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>前向输入，dim维度会被均分为gate与up两部分。数据类型需与grad_y一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>dim</td>
      <td>属性</td>
      <td><ul><li>切分维度，需与前向SituGlu一致。</li><li>取值范围为[-x.dim(), x.dim()-1]。</li><li>默认值为-1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>属性</td>
      <td><ul><li>SiTU门控部分的缩放系数，需与前向SituGlu保持一致。</li><li>默认值为1.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linear_beta</td>
      <td>属性</td>
      <td><ul><li>up路径线性tanh的缩放系数，需与前向SituGlu保持一致。</li><li>默认值为0.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td><ul><li>切分x时gate是否为前半部分，需与前向SituGlu保持一致。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>grad_x</td>
      <td>输出</td>
      <td>输入x的梯度，shape与x完全一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 支持非连续Tensor。
- beta、linear_beta、activate_left、dim需与前向SituGlu调用时的取值一致，否则梯度结果不正确。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | - | 通过[aclnnSituGluGrad](./docs/aclnnSituGluGrad.md)接口调用SituGluGrad算子。 |
| GE图模式 | -  | 通过[算子IR](./op_graph/situ_glu_grad_proto.h)构图方式调用SituGluGrad算子。         |
| PyTorch API |  -  | 通过[situ_glu_grad](./docs/torchapi_situ_glu_grad.md)接口调用SituGluGrad算子。         |
