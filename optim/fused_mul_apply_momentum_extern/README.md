# FusedMulApplyMomentumExtern

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- **算子功能**：CANN图模式融合训练算子，将梯度缩放（Mul）与带动量的参数更新（ApplyMomentum）融合为单次逐元素更新，并**原地**更新主权重`var`、动量缓冲`accum`及低精度权重副本`var_copy`（inplace语义）。由`momentum_lossscale_fusion_pass`融合PASS产生，对标TensorFlow `ApplyMomentum`的加号动量语义。`use_nesterov=True`时采用Nesterov动量。

- **计算公式**：

  $$
  \begin{aligned}
  grad          &= x1 \times x2 \\
  accum_{t}     &= accum \times momentum + grad
  \end{aligned}
  $$

  - 若use_nesterov = False（标准动量）：

    $$
    \Delta v = accum_{t} \times lr
    $$

  - 若use_nesterov = True（Nesterov动量）：

    $$
    \Delta v = grad \times lr + accum_{t} \times momentum \times lr
    $$

  - 参数更新：

    $$
    \begin{aligned}
    var_{t}       &= var - \Delta v \\
    var\_copy_{t} &= var\_copy - \text{Cast}(\Delta v) \\
    accum\_out    &= \text{Cast}(accum_{t})
    \end{aligned}
    $$

  其中`x2`为LossScale倒数缩放标量。主权重`var`恒为FLOAT（FP32）全程高精度更新；低精度副本`var_copy`独立同步更新（减去`Cast(Δv)`而非`Cast(var_t)`，避免二次精度损失）。低精度通路（FP16/BF16）内部升FP32计算，输出Cast回原低精度。

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
    <td>待更新的主权重参数（master weight），对应公式中的var。恒为FP32；Kernel内inplace更新，GE IR输出复用同名var与输入共享Device内存。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入 / 输出(inplace)</td>
    <td>动量缓冲，对应公式中的accum。跨调用持久，shape必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape={1}的1元素scalar Tensor，dtype必须与accum一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x1</td>
    <td>输入</td>
    <td>梯度分量，对应公式中的x1。shape/dtype必须与accum一致，与x2相乘得到梯度grad。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数，对应公式中的momentum。shape={1}的1元素scalar Tensor，dtype必须与accum一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x2</td>
    <td>输入</td>
    <td>梯度缩放标量（LossScale倒数），对应公式中的x2。shape={1}的1元素scalar Tensor，dtype必须与accum一致。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var_copy</td>
    <td>输入 / 输出(inplace)</td>
    <td>低精度权重副本（compute copy），对应公式中的var_copy。shape必须与var一致；Kernel内独立同步更新并写回输入GM地址。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>属性</td>
    <td>是否使用Nesterov动量。默认false（标准动量）。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。与TF语义对齐，当前实现不强制互斥锁，仅作语义占位，不影响数值结果。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的主权重，与输入var共享Device内存（inplace）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>var_copy (output)</td>
    <td>输出</td>
    <td>更新后的低精度权重副本，与输入var_copy共享Device内存（inplace）。</td>
    <td>FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum (output)</td>
    <td>输出</td>
    <td>更新后的动量缓冲，与输入accum共享Device内存（inplace）。</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

- **dtype通路约束**：`var`恒为FLOAT；`accum`/`lr`/`x1`/`momentum`/`x2`五者dtype必须一致，取值{FLOAT, FLOAT16, BFLOAT16}；`var_copy`恒为低精度（FP16/BF16），由accum通路推导（FP32通路→FP16，FP16通路→FP16，BF16通路→BF16）。

## 约束说明

- `var`、`accum`、`x1`、`var_copy`四张量shape必须严格一致（逐元素运算，张量间无broadcast）。
- `lr`、`momentum`、`x2`为shape={1}的标量，广播参与逐元素运算，不支持空Tensor。
- 仅支持Ascend 950PR/Ascend 950DT（DAV_3510 / arch35，RegBase路线）。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式 | [test_geir_fused_mul_apply_momentum_extern](./examples/arch35/test_geir_fused_mul_apply_momentum_extern.cpp) | 通过[算子IR](./op_graph/fused_mul_apply_momentum_extern_proto.h)构图方式调用FusedMulApplyMomentumExtern算子。本算子为CANN图模式融合训练算子，由`momentum_lossscale_fusion_pass`融合PASS产生。 |
