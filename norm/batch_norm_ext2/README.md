# BatchNormExt2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     √    |
|  <term>Atlas 推理系列产品</term>    |     √    |
|  <term>Atlas 训练系列产品</term>    |     √    |

## 功能说明

- 算子功能：对4D输入张量按通道维做批量归一化（Batch Normalization）。

- 计算公式：

  $$
  y = \frac{(x - E(x))}{\sqrt{Var(x) + ε}} * γ + β
  $$

  其中，训练模式下 `E(x)`、`Var(x)` 由当前批次在空间维度上统计得到；推理模式下 `E(x)`、`Var(x)` 取输入 `input_mean`、`input_variance`；`ε` 表示一个极小的浮点数，防止分母为0的情况。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>input_x</td>
      <td>输入</td>
      <td>
      <ul><li>进行批量归一化的输入张量，对应公式中的`x`。</li><li>一个4D张量，shape为(N, C, H, W)或(N, H, W, C)。</li></ul>
      </td>
      <td>FLOAT16、FLOAT32</td>
      <td>NCHW/NHWC</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>输入</td>
      <td>
      <ul><li>进行批量归一化的权重，对应公式中的`γ`。</li><li>一个1D张量，shape与输入input_x的通道维C相同。</li></ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_offset</td>
      <td>输入</td>
      <td><ul><li>进行批量归一化的偏置值，对应公式中的`β`。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mean</td>
      <td>可选输入</td>
      <td><ul><li>训练场景：须为空；推理场景：推理期间使用的均值，为必选输入，对应公式中的`E(x)`。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_variance</td>
      <td>可选输入</td>
      <td><ul><li>训练场景：须为空；推理场景：推理期间使用的方差，为必选输入，对应公式中的`Var(x)`。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>可选属性</td>
      <td><ul><li>添加到方差中的小值以避免除以零，对应公式中的`ε`。</li><li>默认值为1e-4f。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td><ul><li>指定输入input_x的数据格式，支持"NHWC"、"NCHW"。</li><li>默认值为"NHWC"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>可选属性</td>
      <td><ul><li>标记是否训练场景，true表示训练场景，false表示推理场景。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>输出</td>
      <td><ul><li>表示批量归一化后的输出结果，对应公式中的`y`。</li><li>数据类型、数据格式、shape与输入input_x保持一致。</li></ul></td>
      <td>FLOAT16、FLOAT32</td>
      <td>NCHW/NHWC</td>
    </tr>
    <tr>
      <td>output_mean</td>
      <td>输出</td>
      <td><ul><li>训练模式：当前批次的均值（有偏），推理模式：等于输入input_mean。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_variance</td>
      <td>输出</td>
      <td><ul><li>训练模式：当前批次的方差（无偏，贝塞尔校正），推理模式：等于输入input_variance。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_reserve_space_1</td>
      <td>输出</td>
      <td><ul><li>为梯度计算预留。训练模式：保存的均值（等于output_mean），推理模式：等于输入input_mean。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_reserve_space_2</td>
      <td>输出</td>
      <td><ul><li>为梯度计算预留。训练模式：保存的inv_var：1/sqrt(epsilon + variance)，用于反向梯度计算中重用，推理模式：等于输入input_variance。</li><li>一个1D张量，shape与入参input_scale保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入input_x仅支持4D张量，数据格式仅支持NCHW和NHWC。
- 训练模式下，输入input_mean、input_variance必须为空；推理模式下，输入input_mean、input_variance必须提供。
- 不支持空张量。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式调用 | [test_geir_batch_norm_ext2](./examples/test_geir_batch_norm_ext2.cpp)   | 通过[算子IR](./op_graph/batch_norm_ext2_proto.h)构图方式调用BatchNormExt2算子。 |
