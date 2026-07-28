# InplaceApplyCenteredRMSProp

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：InplaceApplyCenteredRMSProp是Centered RMSProp优化器步进的in-place实现，对标TensorFlow `ResourceApplyCenteredRMSProp`，在模型训练的每步迭代中根据梯度、学习率和衰减率就地更新权重变量（var）、梯度移动平均（mg）、梯度平方移动平均（ms）和动量缓冲区（mom）。
- 计算公式：

$$
\begin{aligned}
\text{mg}_t &= \rho \cdot \text{mg}_{t-1} + (1 - \rho) \cdot \text{grad} \\
\text{ms}_t &= \rho \cdot \text{ms}_{t-1} + (1 - \rho) \cdot \text{grad}^2 \\
\text{mom}_t &= \text{momentum} \cdot \text{mom}_{t-1} + \frac{\text{lr} \cdot \text{grad}}{\sqrt{\text{ms}_t - \text{mg}_t^2 + \epsilon}} \\
\text{var}_t &= \text{var}_{t-1} - \text{mom}_t
\end{aligned}
$$

其中：var为权重，mg为梯度移动平均，ms为梯度平方移动平均，mom为动量缓冲区，grad为梯度，lr为学习率，rho为衰减率，momentum为动量系数，epsilon为数值稳定性常数。

epsilon加在sqrt内部（`sqrt(ms - mg² + epsilon)`），对标TensorFlow。PyTorch的epsilon在sqrt之外，不作为对标对象。

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
    <td>输入 / 输出(inplace)</td>
    <td>模型权重张量。Kernel内inplace更新，GE IR输出视图与输入var共享Device内存。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg</td>
    <td>输入(inplace更新)</td>
    <td>梯度移动平均。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms</td>
    <td>输入(inplace更新)</td>
    <td>梯度平方移动平均。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom</td>
    <td>输入(inplace更新)</td>
    <td>动量缓冲区。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率（0-d tensor）。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>输入</td>
    <td>衰减率（0-d tensor）。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数（0-d tensor）。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>数值稳定性常数（0-d tensor），加在sqrt内部。dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前步梯度。shape/dtype必须与var一致。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。当前实现不强制互斥锁，仅作语义占位。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的var Tensor，与输入var共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mg (output)</td>
    <td>输出</td>
    <td>更新后的mg Tensor，与输入mg共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ms (output)</td>
    <td>输出</td>
    <td>更新后的ms Tensor，与输入ms共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>mom (output)</td>
    <td>输出</td>
    <td>更新后的mom Tensor，与输入mom共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- **数据类型一致性**：var、mg、ms、mom、grad五个tensor的数据类型必须一致；lr、rho、momentum、epsilon四个scalar的数据类型必须与tensor一致。
- **Shape约束**：var、mg、ms、mom、grad的shape必须完全相同；lr、rho、momentum、epsilon必须为scalar（0-d tensor）；tensor维度范围0-8维。
- **In-place语义**：var、mg、ms、mom四个参数为就地更新，执行后原始数据被覆盖。
- **epsilon位置**：epsilon必须加在sqrt内部（`sqrt(ms - mg² + ε)`），禁止实现为 `sqrt(ms - mg²) + ε`。
- **标量精度**：lr/rho/momentum/epsilon统一在FP32域读取，禁止用FP16标量参与计算。
- **FP16计算**：FP16输入在kernel内部提升到FP32计算，结果cast回FP16（CAST_NONE输入 / CAST_RINT输出）。
- **大值溢出**：当 `ms - mg² + ε` 接近零时，`lr * grad / sqrt(denom)` 会产生大值溢出，kernel保持数学正确性（不截断、不饱和），输出可能为Inf/NaN。
- **空Tensor**：支持空Tensor（numel=0），kernel跳过计算。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| 图模式 | [test_geir_inplace_apply_centered_rms_prop](./examples/arch35/test_geir_inplace_apply_centered_rms_prop.cpp) | 通过[算子IR](./op_graph/inplace_apply_centered_rms_prop_proto.h)构图方式调用InplaceApplyCenteredRMSProp算子。 |
