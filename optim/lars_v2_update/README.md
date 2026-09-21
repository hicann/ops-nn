# LarsV2Update

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>     |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：LARS-V2 优化器梯度更新，根据权重的 L2 范数自适应调整梯度的信任系数，用于大 batch 训练场景。

- 计算公式：

  $$
  w\_norm = \sqrt{w\_square\_sum}
  $$

  $$
  g\_norm = \sqrt{g\_square\_sum}
  $$

  $$
  coeff = \frac{hyperpara \times w\_norm}{weight\_decay \times w\_norm + g\_norm + epsilon}
  $$

  - 若 use_clip = True:

    $$
    coeff = \max\left(0,\ \min\left(\frac{coeff}{learning\_rate},\ 1\right)\right)
    $$

  $$
  grad\_weight = w \times weight\_decay + g
  $$

  $$
  g\_new = grad\_weight \times coeff
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
    <td>w</td>
    <td>输入</td>
    <td>权重张量。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>g</td>
    <td>输入</td>
    <td>梯度张量，与w同型同形。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>w_square_sum</td>
    <td>输入</td>
    <td>权重平方和（由SquareSumAll预计算），标量。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>g_square_sum</td>
    <td>输入</td>
    <td>梯度平方和，标量。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>weight_decay</td>
    <td>输入</td>
    <td>权重衰减系数，标量。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>learning_rate</td>
    <td>输入</td>
    <td>学习率，标量。仅在use_clip=True时参与计算。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>hyperpara</td>
    <td>属性</td>
    <td>LARS信任系数（eta），默认0.001。</td>
    <td>Float</td>
    <td>-</td>
    </tr>
    <tr>
    <td>epsilon</td>
    <td>属性</td>
    <td>防除零小常数，默认0.00001。</td>
    <td>Float</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_clip</td>
    <td>属性</td>
    <td>是否将coeff裁剪到[0,1]，默认False。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>g_new</td>
    <td>输出</td>
    <td>更新后的梯度，与w同型同形。</td>
    <td>FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- 输入w和g必须具有相同的形状和数据类型（Tiling阶段校验dtype为DT_FLOAT且shape完全一致，见`op_host/arch35/lars_v2_update_tiling_arch35.cpp`）。
- 输出g_new与w同型同形同dtype（InferShape由CANN runtime广播推导，Tiling阶段兜底校验g_new与w同形）。
- w_square_sum、g_square_sum、weight_decay、learning_rate恒为FLOAT类型标量（元素数为1，广播到w的shape参与计算；Tiling阶段校验dtype与元素数）。
- 支持维度0~8维（0维标量按1维处理，Tiling阶段校验维数不超过8）。
- 支持空Tensor（元素数为0，如shape含0维，输出为空Tensor不报错，Kernel侧totalTiles==0短路跳过计算）。
- 输入数据中的NaN按IEEE 754语义传播（硬件向量指令对NaN的裁剪语义与框架参考实现可能存在差异，NaN输入场景需上层框架保证，算子侧不做NaN拦截）。

## 调用说明

| 调用方式 | 调用样例                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| 图模式调用 | [test_geir_lars_v2_update](./examples/test_geir_lars_v2_update.cpp)   | 通过[算子IR](./op_graph/lars_v2_update_proto.h)构图方式调用LarsV2Update算子。 |
