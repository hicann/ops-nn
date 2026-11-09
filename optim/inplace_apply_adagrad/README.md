# InplaceApplyAdagrad

<!-- codespell:ignore accum adagrad -->

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT系列产品</term>   |     √    |
|  <term>Atlas A3系列产品</term>  |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2推理产品</term>    |     ×    |
|  <term>Atlas推理系列产品</term>     |     ×    |
|  <term>Atlas训练系列产品</term>     |     ×    |

> 当前算子注册 `ascend950`、`ascend350`、`ascend960dt` 三个计算单元；其余产品不提供实现。

## 功能说明

- **算子功能**：执行Adagrad优化器的单步参数更新。根据当前梯度自适应调整每个参数的学习率，更新模型参数 `var` 与梯度平方累积量 `accum`。输出 `var`、`accum` 分别与同名输入共享Device内存，实现原地更新（inplace语义）。对标CANNDEV中 `ApplyAdagradD` 接口的计算语义。

- **计算公式**：

  $$
  \begin{aligned}
  accum_{new} &=
  \begin{cases}
  accum + grad^2, & \text{update\_slots} = true \\
  accum, & \text{update\_slots} = false
  \end{cases} \\
  var_{new} &= var - lr \times \frac{grad}{\sqrt{accum_{new}}}
  \end{aligned}
  $$

  其中 `lr` 为学习率，`update_slots` 用于控制是否更新 `accum`。

## 参数说明

<table style="table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 140px">
    <col style="width: 160px">
    <col style="width: 420px">
    <col style="width: 260px">
    <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>输入 / 输出（inplace）</td>
      <td>待更新的参数张量，应来自Variable。对应公式中的var。输出var与输入var共享Device内存。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>输入 / 输出（inplace）</td>
      <td>梯度平方累积量，应来自Variable。对应公式中的accum。shape/dtype必须与var一致；输出accum与输入accum共享Device内存。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>输入</td>
      <td>学习率。对应公式中的lr。支持rank-0标量或shape为 {1} 的单元素Tensor，dtype与var一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>当前梯度Tensor。对应公式中的grad。shape/dtype必须与var一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>update_slots</td>
      <td>属性</td>
      <td>是否更新accum，默认true。为false时accum保持不变，输出accum等于输入accum。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>属性</td>
      <td>是否在更新时加锁，默认false。当前实现不引入额外锁语义。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var（输出）</td>
      <td>输出</td>
      <td>更新后的var Tensor，与输入var共享Device内存（inplace）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum（输出）</td>
      <td>输出</td>
      <td>更新后的accum Tensor，与输入accum共享Device内存（inplace）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## 约束说明

- 输入张量 `var`、`accum`、`grad` 必须具有相同的shape、dtype和format；输出 `var`、`accum` 的shape/dtype与输入一致。
- `lr` 必须为rank-0标量或shape为 `{1}` 的单元素Tensor。
- 支持rank 0-8；支持空Tensor。
- FP16/BF16输入在FP32计算域完成运算后按目标dtype写回。
- `use_locking` 仅作为兼容属性接入，不改变计算结果。
- TensorFlow `ApplyAdagrad` / `ResourceApplyAdagrad` 在 Ascend950 映射到本算子，映射由 `framework/inplace_apply_adagrad_tf_plugin.cpp` 提供；其他产品回退到内置 `ApplyAdagrad`。
- 本算子不提供aclnn对外接口；调用方式为GE图模式和TensorFlow映射。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|:---------|:---------|:-----|
| GE图模式 | [test_geir_inplace_apply_adagrad.cpp](examples/test_geir_inplace_apply_adagrad.cpp) | 通过 [算子IR](op_graph/inplace_apply_adagrad_proto.h) 构图方式调用InplaceApplyAdagrad算子。 |
