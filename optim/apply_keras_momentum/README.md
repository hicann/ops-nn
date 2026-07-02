# ApplyKerasMomentum

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

- **算子功能**：执行Keras Momentum优化器的单步参数更新。根据动量系数momentum、学习率lr和梯度grad更新动量累积量accum，并按标准模式或Nesterov模式原地更新权重参数var（inplace 语义）。对标TensorFlow中`tf.raw_ops.ResourceApplyKerasMomentum`接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  accum_{new} &= momentum \cdot accum - lr \cdot grad
  \end{aligned}
  $$

  - 标准模式（use_nesterov = 0）：

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov模式（use_nesterov = 1）：

  $$
  var_{new} = var + momentum \cdot accum_{new} - lr \cdot grad
  $$

  其中`momentum`为动量系数，`lr`为学习率，`grad`为当前梯度，`accum`为动量累积量，`var`为待更新的权重参数。

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
    <td>输入 / 输出 (inplace)</td>
    <td>待更新的权重参数，对应公式中的var。Kernel内inplace更新，GE IR输出var_out与输入var共享Device内存。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入 (inplace 更新)</td>
    <td>动量累积量，对应公式中的accum。shape/dtype必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度Tensor，对应公式中的grad。shape/dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape={1}的1元素scalar Tensor，dtype为FLOAT。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数，对应公式中的momentum。shape={1}的1元素scalar Tensor，dtype为FLOAT。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>输入</td>
    <td>是否使用Nesterov动量，对应公式中的use_nesterov。shape={1}的1元素scalar Tensor，dtype为FLOAT（0.0=标准模式，1.0=Nesterov模式）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的var Tensor，与输入var共享Device内存（inplace）。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

> **注**：accum 为 in-place 更新（与 TensorFlow ResourceApplyKerasMomentum 一致），不作为显式输出返回，计算结果直接写回输入地址。

## 约束说明
- var、accum、grad必须具有相同的数据类型和形状。
- lr、momentum、use_nesterov为标量Tensor（shape={1}），数据类型为FLOAT。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_apply_keras_momentum](./examples/arch35/test_geir_apply_keras_momentum.cpp) | 通过 [算子IR](./op_graph/apply_keras_momentum_proto.h) 构图方式调用 ApplyKerasMomentum 算子。 |
