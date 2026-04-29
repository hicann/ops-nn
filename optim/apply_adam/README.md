# ApplyAdam

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：根据Adam算法更新变量"var"。

- 计算公式：

  $$
  lr = learning\_rate \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
  $$

  $$
  m = m + (1 - beta1) \times (grad - m)
  $$

  $$
  v = v + (1 - beta2) \times (grad \times grad - v)
  $$

  - 若 use_nesterov = True:

    $$
    var = var - lr \times \frac{m \times beta1 + (1 - beta1) \times grad}{\epsilon + \sqrt{v}}
    $$

  - 若 use_nesterov = False:

    $$
    var = var - lr \times \frac{m}{\epsilon + \sqrt{v}}
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
    <td>m</td>
    <td>输入</td>
    <td>一阶矩估计，应来自Variable。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>v</td>
    <td>输入</td>
    <td>二阶矩估计，应来自Variable。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta1_power</td>
    <td>输入</td>
    <td>beta1的幂次，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2_power</td>
    <td>输入</td>
    <td>beta2的幂次，标量。</td>
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
    <td>beta1</td>
    <td>输入</td>
    <td>一阶矩估计的衰减率，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2</td>
    <td>输入</td>
    <td>二阶矩估计的衰减率，标量。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>用于数值稳定性的小常数，标量。</td>
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
    <td>use_locking</td>
    <td>属性</td>
    <td>是否使用锁机制保护更新操作，默认为False。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_nesterov</td>
    <td>属性</td>
    <td>是否使用Nesterov加速梯度，默认为False。</td>
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

输入张量必须具有相同的形状。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式调用 | [test_geir_apply_adam](./examples/test_geir_apply_adam.cpp)   | 通过[算子IR](./op_graph/apply_adam_proto.h)构图方式调用ApplyAdam算子。 |
