# DeformableOffsets

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |

## 功能说明

- 算子功能：用于计算变形卷积（Deformable Convolution）输出的函数。通过引入偏移参数offsets，使得卷积核在输入特征图上的位置可以动态调整，从而适配不规则的集合变化。
- 计算公式：

  假定输入input的shape是[N, inH, inW, inC]，输出deformOut的shape为[N, outH\*K_H, outW\*K_W, inC]，根据已有属性计算outH、outW：

  $$
  outH = (inH + pads[0] + pads[1] - ((K\_H - 1) * dilations[1] + 1)) // strides[1] + 1
  $$

  $$
  outW = (inW + pads[2] + pads[3] - ((K\_W - 1) * dilations[2] + 1)) // strides[2] + 1
  $$

  标准卷积计算采样点下标：

  $$
  x = -pads[2] + ow*strides[2] + kw*dilations[2], ow的取值为(0, outW-1), kw的取值为(0, K\_W-1)
  $$

  $$
  y = -pads[0] + oh*strides[1] + kh*dilations[1], oh的取值为(0, outH-1), kh的取值为(0, K\_H-1)
  $$

  根据传入的`offsets`，进行变形卷积，计算偏移后的下标：

  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  使用双线性插值计算偏移后点的值：

  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$

  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\
  $$

  $$
  deformOut(x, y) = mask * (weight_{00} * input(x0, y0) + weight_{01} * input(x0, y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1, y1))
  $$

  其中，`input`表示输入张量（参数表中对应`x`），`offsets`表示偏移值，`deformOut`表示输出张量（参数表中对应`y`），`strides`指定高度和宽度方向上的步幅，`pads`指定高度和宽度方向上的填充，`ksize`指定卷积核大小（对应公式中的`K_H`、`K_W`），`dilations`指定每个维度上的膨胀因子，`offsetX`和`offsetY`表示`offsets`中的偏移分量，`mask`为`offsets`中对应卷积核位置的第三个分量（modulated=true时恒存在）。偏移后的采样点(x, y)及其插值四邻域点可能落在输入图像边界之外，kernel实际按“越界取0”处理。

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
      <td>x</td>
      <td>输入</td>
      <td>表示输入的原始数据。对应公式中的`input`。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>表示偏移值。对应公式中的`offsets`。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示变形卷积的输出。对应公式中的`deformOut`。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>表示指定卷积核在高度和宽度方向上的步幅，即(1, stridesH, stridesW, 1)。对应公式中的`strides`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>指定在输入的高度和宽度方向上添加的填充。对应公式中的`pads`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>指定卷积核的大小。对应公式中的`K_H`、`K_W`。size为2(K_H, K_W)，各元素均大于零。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>属性</td>
      <td>指定每个维度上的膨胀因子，即(1, dilationsH, dilationsW, 1)。对应公式中的`dilations`。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>指定输入x的数据格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>属性</td>
      <td>指定输入x在C轴上的分组数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>属性</td>
      <td>指定变形卷积的版本，表示offset中是否包含掩码。若为true，`offset`中包含掩码；若为false，则不包含。当前仅支持true。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_deformable_offsets](./examples/test_geir_deformable_offsets.cpp)   | 通过[算子IR](./op_graph/deformable_offsets_proto.h)构图方式调用DeformableOffsets算子。 |
