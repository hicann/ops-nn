# ApplyMomentum

##  产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：根据动量方案更新变量"var"。若设置use_nesterov=True，则使用Nesterov动量。

- 计算公式：

  $$
  accum = accum \times momentum + grad
  $$

  - 若 use_nesterov = True:

    $$
    var = var - grad \times lr + accum \times momentum \times lr
    $$

  - 若 use_nesterov = False:

    $$
    var = var - lr \times accum
    $$

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 100px">
<col style="width: 150px">
<col style="width: 280px">
<col style="width: 330px">
<col style="width: 120px">
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
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>accum</td>
    <td>输入</td>
    <td>梯度累积值，应来自Variable。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>grad</td>
    <td>输入</td>
    <td>梯度张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>momentum</td>
    <td>输入</td>
    <td>动量系数，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_nesterov</td>
    <td>属性</td>
    <td>是否使用Nesterov动量，默认为False。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否使用锁机制保护更新操作，默认为False。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的参数张量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

输入张量必须具有相同的形状和类型。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_apply_momentum](./examples/test_geir_apply_momentum.cpp)   | 通过[算子IR](./op_graph/apply_momentum_proto.h)构图方式调用ApplyMomentum算子。 |