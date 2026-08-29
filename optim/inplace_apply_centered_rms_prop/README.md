# InplaceApplyCenteredRMSProp

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：实现Centered RMSProp优化器的参数更新。在模型训练过程中，根据当前梯度、学习率和衰减系数，原地更新模型参数`var`、梯度一阶矩`mg`、梯度二阶矩`ms`和动量`mom`。
- 计算公式：

  $$
  \begin{aligned}
  mg_t &= \rho \cdot mg_{t-1} + (1 - \rho) \cdot g_t \\
  ms_t &= \rho \cdot ms_{t-1} + (1 - \rho) \cdot g_t^2 \\
  mom_t &= \mu \cdot mom_{t-1} + lr \cdot \frac{g_t}{\sqrt{ms_t - mg_t^2 + \epsilon}} \\
  var_t &= var_{t-1} - mom_t
  \end{aligned}
  $$

  其中，$var$表示模型参数，$mg$表示梯度一阶矩，$ms$表示梯度二阶矩，$mom$表示动量，$g_t$表示当前步梯度，$lr$表示学习率，$\rho$表示衰减系数，$\mu$表示动量系数，$\epsilon$表示数值稳定性常数。$\epsilon$位于平方根内部，与TensorFlow `ResourceApplyCenteredRMSProp`的计算方式一致。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
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
    <td>var</td>
    <td>输入</td>
    <td>待更新的模型参数，shape与mg、ms、mom和grad一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>输入</td>
    <td>梯度一阶矩的指数移动平均，shape与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>输入</td>
    <td>梯度二阶矩的指数移动平均，shape与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>输入</td>
    <td>动量，shape与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率。必须为标量或仅含一个元素的一维张量。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>输入</td>
    <td>衰减系数。必须为标量或仅含一个元素的一维张量。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数。必须为标量或仅含一个元素的一维张量。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>数值稳定性常数。必须为标量或仅含一个元素的一维张量。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前步梯度，shape与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>可选兼容属性，默认值为false，取值不影响计算结果。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的模型参数，与输入var共享存储。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>输出</td>
    <td>更新后的梯度一阶矩，与输入mg共享存储。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>输出</td>
    <td>更新后的梯度二阶矩，与输入ms共享存储。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>输出</td>
    <td>更新后的动量，与输入mom共享存储。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- var、mg、ms、mom和grad的shape必须完全一致。
- lr、rho、momentum和epsilon必须为标量或仅含一个元素的一维张量。
- 所有输入和输出的数据类型必须相同，支持FLOAT和FLOAT16。
- var、mg、ms和mom均为原地更新，输出与对应输入共享存储。
- <term>Ascend 950PR/Ascend 950DT</term>：
  - var、mg、ms、mom、grad及其对应输出仅支持ND格式，rank不超过8。
  - 支持空Tensor。
  - FLOAT16输入在内部使用FLOAT精度进行计算，结果转换为FLOAT16后输出。
- `use_locking`仅用于接口兼容，当前实现不提供额外的互斥锁。
- 本算子仅支持GE图模式调用，不提供公开的aclnn接口。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| 图模式 | [test_geir_inplace_apply_centered_rms_prop](./examples/arch35/test_geir_inplace_apply_centered_rms_prop.cpp) | 通过[算子IR](./op_graph/inplace_apply_centered_rms_prop_proto.h)构图方式调用InplaceApplyCenteredRMSProp算子。 |
