# InplaceApplyPowerSign

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

- **算子功能**：执行PowerSign优化器的单步参数更新。PowerSign将更新机制解耦为"方向"和"幅度"两个维度，利用梯度与一阶矩的符号一致性来自适应调整更新步长，对权重`var`和一阶矩`m`进行**原地**更新（inplace语义）。对标TensorFlow `tf.raw_ops.ApplyPowerSign` / `tf.raw_ops.ResourceApplyPowerSign`接口。

- **计算公式**：

  $$
  \begin{aligned}
  m_{t} &= \beta \cdot m_{t-1} + (1 - \beta) \cdot grad \\
  sign\_gm &= \text{sign}(m_{t}) \cdot \text{sign}(grad) \\
  var_{t} &= var_{t-1} - lr \cdot \exp(logbase \cdot sign\_decay \cdot sign\_gm) \cdot grad
  \end{aligned}
  $$

  其中`sign(x)`为符号函数（x > 0 返回 1，x < 0 返回 -1，x = 0 返回 0），`exp()`为自然指数函数。

## 参数说明

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 600px">
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
    <td>待更新的权重参数张量，对应公式中的var。与图输出端口var共享GM地址（inplace更新）。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输入</td>
    <td>一阶矩估计张量，对应公式中的m。shape/dtype必须与var一致；与图输出端口m共享GM地址（inplace更新）。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape必须为 [1] 的scalar Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>logbase</td>
    <td>输入</td>
    <td>对数基底，对应公式中的logbase。shape必须为 [1] 的scalar Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>sign_decay</td>
    <td>输入</td>
    <td>符号衰减因子，对应公式中的sign_decay。shape必须为 [1] 的scalar Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>beta</td>
    <td>输入</td>
    <td>一阶矩衰减率，对应公式中的β，取值范围 [0, 1)。shape必须为 [1] 的scalar Tensor。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度张量，对应公式中的grad。shape/dtype必须与var一致。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
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
    <td>var</td>
    <td>输出</td>
    <td>更新后的权重张量，与输入var共享Device内存（inplace）。shape/dtype与输入var完全相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>m</td>
    <td>输出</td>
    <td>更新后的一阶矩估计张量，与输入m共享Device内存（inplace）。shape/dtype与输入m完全相同。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- var、m、grad三个张量的shape必须一致。
- lr、logbase、sign_decay、beta必须为shape=[1]的scalar Tensor。
- 所有输入输出的DataType必须一致。
- FP16/BF16输入内部提升至FP32计算后cast回原类型。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_inplace_apply_power_sign](./examples/arch35/test_geir_inplace_apply_power_sign.cpp) | 通过[算子IR](./op_graph/inplace_apply_power_sign_proto.h)构图方式调用InplaceApplyPowerSign算子。 |
