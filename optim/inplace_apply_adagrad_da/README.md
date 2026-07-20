# InplaceApplyAdagradDA

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

- **算子功能**：执行 Adagrad Dual Averaging 优化器的单步参数更新。结合 Adagrad 的自适应学习率机制与 Dual Averaging 的稀疏性诱导特性，根据当前梯度**原地**更新模型参数及梯度累加器（inplace 语义）。对标 TensorFlow 中 `tf.raw_ops.ApplyAdagradDA` 接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  g_{out}   &= g_{acc} + grad \\
  gg_{out}  &= gg_{acc} + grad^2 \\
  T         &= float(global\_step) \\
  tmp       &= sign(g_{out}) \cdot \max(|g_{out}| - l1 \cdot T,\ 0) \quad &\text{if } l1 \neq 0 \\
  tmp       &= g_{out} \quad &\text{if } l1 = 0 \\
  var_{out} &= \frac{-lr \cdot tmp}{l2 \cdot T \cdot lr + \sqrt{gg_{out}}}
  \end{aligned}
  $$

  其中`lr`为学习率，`l1`为 L1 正则化系数，`l2`为 L2 正则化系数，`global_step`为训练步数，`sign(\cdot)`为符号函数（NaN保留NaN）。

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
    <td>待更新的权重参数，对应公式中的var。Kernel内inplace 更新，GE IR单输出视图与输入var共享Device内存。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_accumulator</td>
    <td>输入 (inplace 更新)</td>
    <td>梯度累加器，对应公式中的g_acc。shape/dtype 必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_squared_accumulator</td>
    <td>输入 (inplace 更新)</td>
    <td>梯度平方累加器，对应公式中的gg_acc。shape/dtype 必须与var一致；Kernel内显式写回输入GM地址。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>当前梯度Tensor，对应公式中的grad。shape/dtype 必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，对应公式中的lr。shape={1} 的 1 元素scalar Tensor，dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l1</td>
    <td>输入</td>
    <td>L1 正则化系数，对应公式中的l1。shape={1} 的 1 元素scalar Tensor，dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>l2</td>
    <td>输入</td>
    <td>L2 正则化系数，对应公式中的l2。shape={1} 的 1 元素scalar Tensor，dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>global_step</td>
    <td>输入</td>
    <td>训练步数，对应公式中的global_step。shape={1} 的 1 元素scalar Tensor。float32 时为 INT32，float16 时为 INT64。</td>
    <td>INT32、INT64</td>
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
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_accumulator (output)</td>
    <td>输出</td>
    <td>更新后的梯度累加器 Tensor，与输入gradient_accumulator共享Device内存（inplace）。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_squared_accumulator (output)</td>
    <td>输出</td>
    <td>更新后的梯度平方累加器 Tensor，与输入gradient_squared_accumulator共享Device内存（inplace）。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- var/gradient_accumulator/gradient_squared_accumulator/grad/lr/l1/l2 的数据类型必须一致
- gradient_squared_accumulator 值应为非负（参与 sqrt 计算）
- lr/l2/global_step 值应为正数
- 支持 0~8 维输入，支持动态 Shape

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_inplace_apply_adagrad_da](./examples/arch35/test_geir_inplace_apply_adagrad_da.cpp) | 通过 [算子IR](./op_graph/inplace_apply_adagrad_da_proto.h) 构图方式调用 InplaceApplyAdagradDA 算子。 |
