# ApplyAdadelta

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

- **算子功能**：执行 Adadelta 优化器的单步参数更新。根据当前梯度、梯度平方累积和更新量平方累积，计算参数更新量并**原地**更新权重参数（inplace 语义）。对标 TensorFlow `tf.raw_ops.ApplyAdadelta` 接口和 PyTorch `torch.optim.Adadelta` 的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  accum_{t} &= \rho \cdot accum_{t-1} + (1 - \rho) \cdot grad^2 \\
  update &= \frac{\sqrt{accum\_update_{t-1} + \epsilon}}{\sqrt{accum_{t} + \epsilon}} \cdot grad \\
  var_{t} &= var_{t-1} - lr \cdot update \\
  accum\_update_{t} &= \rho \cdot accum\_update_{t-1} + (1 - \rho) \cdot update^2
  \end{aligned}
  $$

  其中 `rho` 为衰减系数（取值范围 [0, 1)），`epsilon` 为数值稳定常数（必须 > 0），`lr` 为学习率。

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
    <td>待更新的权重参数，对应公式中的 var。Kernel 内 inplace 更新，GE IR 单输出视图与输入 var 共享 Device 内存。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入 (inplace 更新)</td>
    <td>梯度平方累积，对应公式中的 accum。shape/dtype 必须与 var 一致；Kernel 内显式写回输入 GM 地址。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum_update</td>
    <td>输入 (inplace 更新)</td>
    <td>更新量平方累积，对应公式中的 accum_update。shape/dtype 必须与 var 一致；Kernel 内显式写回输入 GM 地址。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的 lr。shape={1} 的 1 元素 scalar Tensor，dtype 必须与 var 一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>输入</td>
    <td>衰减系数，对应公式中的 rho。shape={1} 的 1 元素 scalar Tensor，取值范围 [0, 1)。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>数值稳定常数，对应公式中的 epsilon。shape={1} 的 1 元素 scalar Tensor，必须 > 0。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度 Tensor，对应公式中的 grad。shape/dtype 必须与 var 一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认 false。当前实现不强制互斥锁，仅作语义占位。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>输出</td>
    <td>更新后的 var Tensor，与输入 var 共享 Device 内存（inplace）。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明
无
## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_apply_adadelta](./examples/test_geir_apply_adadelta.cpp) | 通过 [算子IR](./op_graph/apply_adadelta_proto.h) 构图方式调用 ApplyAdadelta 算子。 |