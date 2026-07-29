# InplaceApplyMomentum

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

> 说明：InplaceApplyMomentum当前仅在<term>Ascend 950PR/Ascend 950DT</term>配置AICore信息库，其余产品暂未适配。

## 功能说明

- 算子功能：根据动量方案更新变量"var"。若设置use_nesterov=True，则使用Nesterov动量。

- 计算公式：

  $$
  accum = accum \times momentum + grad
  $$

  - 若use_nesterov = True：

    $$
    var = var - (grad \times lr + accum \times momentum \times lr)
    $$

  - 若use_nesterov = False：

    $$
    var = var - lr \times accum
    $$

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
    <td>待更新的参数张量，应来自Variable。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输入</td>
    <td>梯度累积值，应来自Variable。shape和dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，标量（0维或元素数为1）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>输入</td>
    <td>梯度张量。shape和dtype必须与var一致。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数，标量（0维或元素数为1）。</td>
    <td>FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_nesterov</td>
    <td>属性</td>
    <td><ul><li>是否使用Nesterov动量。</li><li>默认值为False。</li></ul></td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td><ul><li>是否使用锁机制保护更新操作。</li><li>默认值为False。</li></ul></td>
    <td>Bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的参数张量，与输入var共享Device内存。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>输出</td>
    <td>更新后的梯度累积值，与输入accum共享Device内存。</td>
    <td>FLOAT16、FLOAT、BFLOAT16</td>
    <td>ND</td>
  </tr>
</tbody></table>

## 约束说明

- var、accum、grad必须具有相同的数据类型和形状。
- lr、momentum为标量Tensor（0维或元素数为1），数据类型与var一致。
- FP16/BF16输入的中间计算统一提升至FP32执行，输出前还原至原始数据类型。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| 图模式调用 | [test_geir_inplace_apply_momentum](./examples/arch35/test_geir_inplace_apply_momentum.cpp) | 通过[算子IR](./op_graph/inplace_apply_momentum_proto.h)构图方式调用InplaceApplyMomentum算子。 |
| 图模式调用（动态Shape） | [test_geir_inplace_apply_momentum_dynamic](./examples/arch35/test_geir_inplace_apply_momentum_dynamic.cpp) | 使用动态Shape（`-2`）构图，验证多种实际Shape下的计算正确性。 |
