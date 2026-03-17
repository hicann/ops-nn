# DeformableOffsetsGrad

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |

## 功能说明

- 算子功能：DeformableOffsets算子的目的是根据offsets（kernel采样点的偏移值）来收集用于卷积的特征采样点，并对其进行重组，方便Conv2d算子进行卷积计算。而DeformableOffsetsGrad即为这一过程的反向。

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
      <td>grad</td>
      <td>输入</td>
      <td>输入的梯度。</td>
      <td>DT_FLOAT16, DT_FLOAT, DT_BF16</td>
      <td>NHWC</td>
    </tr>
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
      <td>grad_x</td>
      <td>输出</td>
      <td>x 的梯度。</td>
      <td>DT_FLOAT16, DT_FLOAT</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>grad_offsets</td>
      <td>输出</td>
      <td>相对于 input_offsets 的梯度。</td>
      <td>DT_FLOAT16, DT_FLOAT</td>
      <td>NHWC</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>滑动窗口的步长，包含4个整数的列表。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>填充维度，包含4个整数的列表。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>卷积核的大小，包含2个整数列表。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilations</td>
      <td>属性</td>
      <td>每个维度的伸缩因子，包含4个整数的列表。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>属性</td>
      <td>输入 x 的数据格式。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformable_groups</td>
      <td>属性</td>
      <td> C 轴上的分组。</td>
      <td>INT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>属性</td>
      <td>可选 bool 指定 DeformableConv2D 的版本，true 表示 v2 ，false 表示 v1，只支持 " true "。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

无

## 调用说明

| 调用方式 | 样例代码                                                                   | 说明                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| 图模式 | [test_geir_deformable_offsets_grad](./examples/test_geir_deformable_offsets_grad.cpp)   | 通过[算子IR](./op_graph/deformable_offsets_grad_proto.h)构图方式调用DeformableOffsetsGrad算子。 |