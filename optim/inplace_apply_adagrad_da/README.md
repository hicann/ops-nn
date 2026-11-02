# InplaceApplyAdagradDA

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     ×    |
|  <term>Atlas A2系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>    |     ×    |
|  <term>Atlas训练系列产品</term>    |     ×    |

## 功能说明

- **算子功能**：执行 Adagrad Dual Averaging 优化器的单步参数更新。结合 Adagrad 的自适应学习率机制与 Dual Averaging 的稀疏性诱导特性，根据当前梯度**原地**更新模型参数及梯度累加器（inplace 语义）。对标 TensorFlow 中 `tf.raw_ops.ApplyAdagradDA` 接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  g_{out}   &= g_{acc} + grad \\
  gg_{out}  &= gg_{acc} + grad^2 \\
  T         &= float(global\_step) \\
  tmp       &= sign(g_{out}) \cdot \max(|g_{out}| - l1 \cdot T,\ 0) \quad &\text{if } l1 > 0 \\
  tmp       &= g_{out} \quad &\text{if } l1 \le 0 \\
  var_{out} &= \frac{-lr \cdot tmp}{l2 \cdot T \cdot lr + \sqrt{gg_{out}}}
  \end{aligned}
  $$

  其中`lr`为学习率，`l1`为 L1 正则化系数，`l2`为 L2 正则化系数，`global_step`为训练步数。仅当 `l1 > 0` 时执行稀疏截断；数值边界行为（NaN/Inf、值域、空 Tensor）见[约束说明](#约束说明)。

  > **注**：`var_out` 为按对偶平均法（Dual Averaging）从梯度累加器直接重算得到的**新参数值**，计算不依赖 `var` 输入（TF `ApplyAdagradDA` 与 CANN 内置 `ApplyAdagradDA`（含废弃别名 `ApplyAdagradDAD`）同此语义）。请勿将其理解为"待应用的更新量"——配合本算子的 inplace 写回，若下游再执行一次 `var = var - var_out` 将导致双重应用。

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
    <td>待更新的权重参数，对应公式中的var。Kernel内inplace 更新，GE IR单输出视图与输入var共享Device内存。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_accumulator</td>
    <td>输入</td>
    <td>梯度累加器，对应公式中的g_acc。shape/dtype 必须与var一致；Kernel内显式写回输入GM地址（inplace 更新）。</td>
    <td>FLOAT16、FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradient_squared_accumulator</td>
    <td>输入</td>
    <td>梯度平方累加器，对应公式中的gg_acc。shape/dtype 必须与var一致；Kernel内显式写回输入GM地址（inplace 更新）。</td>
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
    <td>训练步数，对应公式中的global_step。shape={1} 的 1 元素scalar Tensor。无论其他输入为 FLOAT 或 FLOAT16，均支持 INT32、INT64。</td>
    <td>INT32、INT64</td>
    <td>ND</td>
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
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否在更新时加锁。默认false。true 时忽略并正常计算（对齐 CANN TBE 内置实现与 TF——锁为并发保护语义，单算子执行无并发更新场景）。</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
</tbody></table>

## 约束说明

- var/gradient_accumulator/gradient_squared_accumulator/grad/lr/l1/l2 的数据类型必须一致
- 调用方需保证 gradient_squared_accumulator ≥ 0、lr > 0、l2 > 0、global_step > 0；算子内部不做运行时值域校验，违约值按公式计算（如负 gradient_squared_accumulator 经 sqrt 产生 NaN、lr=0 时 var_out 为 0），行为与 TensorFlow 原生实现一致，需由上游调用方规避
- 输入含 NaN/Inf 时，算子不做校验与拒绝，按计算公式处理：参与运算的张量输入（gradient_accumulator/gradient_squared_accumulator/grad）中某元素为 NaN/Inf 时，对应位置的输出元素为 NaN/Inf（逐元素传播）；标量输入（lr/l1/l2/global_step）为 NaN/Inf 时，经广播影响其参与计算的全部输出元素；var 输入的值不参与计算，其 NaN/Inf 不影响输出。传播行为与 TensorFlow 原生 ApplyAdagradDA 一致，需由上游调用方规避
- 支持 0-8 维 ND 输入
- 空 Tensor（含 0 长度维）为合法输入：输出透传空 shape，不执行计算
- 非连续 Tensor：算子注册框架自动连续化（AutoContiguous），非连续输入经框架转换后按连续 ND 布局计算；算子自身不提供原始 stride 的原生支持

## 错误行为说明

算子无独立命名错误码；对非法输入一律返回 ge::GRAPH_FAILED 并以 OP_LOGE 输出日志（各场景日志特征见下表）。调用方在框架层接到的错误码由 GE 包装产生、随框架版本而变，不属于本算子契约。

| 违约场景 | 拒绝阶段 | 日志特征 |
|---|---|---|
| 必选输入缺失 | GE 图构造参数检查（E80001）；Host 推理判空兜底 | required input[i] is missing; input name=... |
| var rank > 8 | Host InferShape / Tiling | var rank %zu exceeds maximum supported rank %zu |
| 张量输入 shape 与 var 不一致 | Host InferShape / Tiling | input[i] shape must exactly match var shape; ... |
| 标量输入 shape 非 [1] | Host InferShape / Tiling | input[i] must have scalar shape [1]; ... |
| dtype 域外或与 var 不一致 / global_step 非整型 | Host InferDataType / Tiling | unsupported dtype ... / dtype must match input var dtype ... |
| format 非 ND | Tiling | input[i] format ... is unsupported; legal format=ND |

注：动态 Shape 图（含 -1 未知维标记）在图构造期不做 shape 关系判定，由 Tiling 以绑定后的实际 shape 校验。

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| GE图模式 | [test_geir_inplace_apply_adagrad_da](./examples/arch35/test_geir_inplace_apply_adagrad_da.cpp) | 通过 [算子IR](./op_graph/inplace_apply_adagrad_da_proto.h) 构图方式调用 InplaceApplyAdagradDA 算子。 |
| TF前端接入 | — | TF 图中的 `ApplyAdagradDA`/`ResourceApplyAdagradDA` 经 [framework/apply_adagrad_da_tf_plugin.cpp](./framework/apply_adagrad_da_tf_plugin.cpp) 映射接入（承诺支持；plugin 注册名与算子注册名统一修复后开放端到端验证）。 |
