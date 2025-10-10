# AvgPool3dGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|
|Atlas 200I/500 A2推理产品|×|

## 功能说明

- 算子功能：三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

- 计算公式：
$$
D_{in} = (D_{out} - 1) * {stride[0]} + kernel\_size[0] - 2 * padding[0]
$$
$$
H_{in} = (H_{out} - 1) * {stride[1]} + kernel\_size[1] - 2 * padding[1]
$$
$$
W_{in} = (W_{out} - 1) * {stride[2]} + kernel\_size[2] - 2 * padding[2]
$$

## 参数说明

<table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 250px">
  <col style="width: 200px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示输入的梯度。数据类型需要与`self`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示待进行avgpool3dgrad计算的入参。数据类型需要与`gradOutput`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>表示池化窗口大小。数值必须大于0。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>表示池化操作的步长。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>表示在输入的D、H、W方向上padding补0的层数。</td>
      <td>INT32、INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>表示正向平均池化过程中推导的输出的shape是否向上取整（True表示向上取整）。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>输入</td>
      <td>计算正向平均池化时是否包括padding填充的0（True表示包括填充的0）。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>输入</td>
      <td>表示取平均的除数。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>待进行avgpool3dgrad计算的出参。数据类型、数据格式需要与`gradOutput`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_avgpool2d_backward.cpp](examples/test_aclnn_avgpool2d_backward.cpp) | 通过[aclnnAvgPool2dBackward](docs/aclnnAvgPool2dBackward.md)接口方式调用AvgPool2dGrad算子。 |
| 图模式 | - | 通过[算子IR](op_graph/avg_pool3_d_grad_proto.h)构图方式调用AvgPool3dGrad算子。         |

<!-- [test_geir_avg_pool3_d_grad.cpp](examples/test_geir_avg_pool3_d_grad.cpp) -->
