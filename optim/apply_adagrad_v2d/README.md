# ApplyAdagradV2d

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

- **算子功能**：执行 Adagrad V2 优化器的单步参数更新。通过累积历史梯度的平方和来自适应调整每个参数的学习率，V2 相比 V1 在分母添加 epsilon 参数提供数值稳定性保护。Inplace 语义，var 和 accum 输出与输入共享 Device 内存。对标 CANNDEV 中 `ApplyAdagradV2D` 接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  accum\_new &= accum + grad^2 \\
  var\_new   &= var - lr \times \frac{grad}{\sqrt{accum\_new} + \epsilon}
  \end{aligned}
  $$

  其中`lr`为学习率，`epsilon`用于数值稳定性的小常数，`accum`更新受update_slots参数控制：当update_slots=false时，accum_new = accum（不更新累积和）。

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
    <td>待更新的参数张量，应来自Variable。对应公式中的var。Kernel内inplace更新，输出与输入var共享Device内存。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入 / 输出 (inplace)</td>
    <td>梯度平方累积和，应来自Variable。对应公式中的accum。shape/dtype 必须与var一致；Kernel内inplace更新，输出与输入accum共享Device内存。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape={1} 的 1 元素scalar Tensor。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度Tensor，对应公式中的grad。shape/dtype 必须与var一致。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>属性</td>
    <td>用于数值稳定性的小常数，对应公式中的epsilon。REQUIRED_ATTR，必填。</td>
    <td>FLOAT</td>
    <td>-</td>
  </tr>
  <tr>
    <td>update_slots</td>
    <td>属性</td>
    <td>是否更新accum。默认true。当设为false时，accum保持不变。</td>
    <td>BOOL</td>
    <td>-</td>
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
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum (output)</td>
    <td>输出</td>
    <td>更新后的accum Tensor，与输入accum共享Device内存（inplace）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- 输入张量var、accum、grad必须具有相同的形状和数据类型
- 支持空Tensor
- 支持的维度范围为0-8维

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_apply_adagrad_v2d](./examples/arch35/test_geir_apply_adagrad_v2d.cpp) | 通过 [算子IR](./op_graph/apply_adagrad_v2d_proto.h) 构图方式调用 ApplyAdagradV2d 算子。 |
