# DeformableOffsets

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |

## 功能说明

- 算子功能：用于计算变形卷积（Deformable Convolution）输出的函数。通过引入偏移参数offsets，使得卷积核在输入特征图上的位置可以动态调整，从而适配不规则的集合变化。

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
      <td>输入的坐标。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>输入</td>
      <td>偏移值。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>变形卷积的输出。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>指定卷积核在高度和宽度方向上的步幅。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>指定在输入的高度和宽度方向上添加的填充。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>指定卷积核的大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>属性</td>
      <td>指定每个维度上的膨胀因子。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>指定输入 x 的数据格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>属性</td>
      <td>指定输入 x 在 C 轴上的分组数。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>属性</td>
      <td>指定变形卷积的版本。</td>
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