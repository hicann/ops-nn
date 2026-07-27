# ActULQClampMaxGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：用于ULQ（Ultra-Low-bit Quantization，超低比特量化）量化感知训练中，计算clamp上界截断的反向梯度。该算子对输入张量执行全轴归约求和操作，输出一个标量梯度值。
- 计算公式：

  $$
  clamp\_max\_grad = \sum_{全轴}(y\_grad \times (x\_clamped\_loss + |clamp\_max\_mask|))
  $$

  其中：

  - $y\_grad$：来自后续层的梯度张量
  - $clamp\_max\_mask$：clamp上界掩码（布尔型或浮点型）
  - $x\_clamped\_loss$：经过clamp后的损失值张量
  - $clamp\_max\_grad$：输出的标量梯度（0维张量）

## 参数说明

<table style="table-layout: fixed; width: 1576px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 400px">
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
    <td>y_grad</td>
    <td>输入</td>
    <td>来自后续层的梯度张量，必选参数。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clamp_max_mask</td>
    <td>输入</td>
    <td>clamp上界掩码张量，与y_grad形状相同，必选参数。</td>
    <td>FLOAT16、FLOAT32、BOOL</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>x_clamped_loss</td>
    <td>输入</td>
    <td>经过clamp后的损失值张量，与y_grad形状相同，必选参数。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>clamp_max_grad</td>
    <td>输出</td>
    <td>输出的标量梯度（0维张量），数据类型与y_grad一致，必选参数。</td>
    <td>FLOAT16、FLOAT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

### 支持的数据类型组合

| y_grad | clamp_max_mask | x_clamped_loss | clamp_max_grad |
|--------|----------------|----------------|----------------|
| FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 |
| FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
| FLOAT16 | BOOL | FLOAT16 | FLOAT16 |
| FLOAT32 | BOOL | FLOAT32 | FLOAT32 |

## 约束说明

- 三个输入张量的shape必须完全一致，不支持广播。
- 当clamp_max_mask为BOOL类型时，y_grad和x_clamped_loss必须为相同的浮点类型（均为FLOAT16或均为FLOAT32）。
- 当clamp_max_mask为浮点类型时，三个输入必须为相同的浮点类型。
- 输入张量必须是连续存储。
- 不支持动态Format。
- 支持动态Shape和动态Rank。

## 调用说明

| 调用方式   | 调用样例            | 说明                          |
|-----------|--------------------|----------------------------------------------|
| GE图模式 | [test_geir_act_ulq_clamp_max_grad](examples/test_geir_act_ulq_clamp_max_grad.cpp) | 通过[算子IR](op_graph/act_ulq_clamp_max_grad_proto.h)构图方式调用ActULQClampMaxGrad算子。 |
