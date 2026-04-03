# ApplyFtrl

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

- 算子功能：根据Ftrl-proximal方案更新变量"var"。

- 计算公式：

  $$
  accum\_new = accum + grad \times grad
  $$

  $$
  linear = linear + grad - \frac{accum\_new^{-lr\_power} - accum^{-lr\_power}}{lr} \times var
  $$

  $$
  quadratic = \frac{accum\_new^{-lr\_power}}{lr} + 2 \times l2
  $$

  - 若 $|linear| > l1$:

    $$
    var = \frac{sign(linear) \times l1 - linear}{quadratic}
    $$

  - 否则:

    $$
    var = 0.0
    $$

  $$
  accum = accum\_new
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
    <td>linear</td>
    <td>输入</td>
    <td>校正项，应来自Variable。</td>
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
    <td>lr</td>
    <td>输入</td>
    <td>缩放因子，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>l1</td>
    <td>输入</td>
    <td>L1正则化系数，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>l2</td>
    <td>输入</td>
    <td>L2正则化系数，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>lr_power</td>
    <td>输入</td>
    <td>缩放因子的幂次，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否使用锁机制保护更新操作，默认为False。仅支持False。</td>
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
| 图模式调用 | [test_geir_apply_ftrl](./examples/test_geir_apply_ftrl.cpp)   | 通过[算子IR](./op_graph/apply_ftrl_proto.h)构图方式调用ApplyFtrl算子。 |