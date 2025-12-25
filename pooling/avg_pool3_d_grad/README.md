# AvgPool3DGrad

##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 功能说明

- 算子功能：三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

- 计算公式：

$$
D_{out} = \left\lfloor \frac{D_{in} + 2 \times \text{pads}[0] - \text{ksize}[0]}{\text{strides}[0]} + 1 \right\rfloor
$$


$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{pads}[1] - \text{ksize}[1]}{\text{strides}[1]} + 1 \right\rfloor
$$


$$
W_{out} = \left\lfloor \frac{W_{in} + 2 \times \text{pads}[2] - \text{ksize}[2]}{\text{strides}[2]} + 1 \right\rfloor
$$

  若属性ceil_mode为 `True`，且满足条件 **$(D_{out} - 1) \times \text{stride}[0] \geq D_{in} + \text{padding}[0]$**，则会跳过最后一个窗口，这将导致维度 **$D_{out}$** 减少 1。此规则同样适用于维度 **$W_{out}$** 和 **$H_{out}$**。

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
      <td>orig_input_shape</td>
      <td>输入</td>
      <td><ul><li>表示正向的输入shape的值。</li><li>输入的值为[N，C，Din，Hin，Win]。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td><ul><li>表示反向输入的梯度。</li><li>输入shape为[N，C，Dout，Hout，Wout]或者[N，Dout，Hout，Wout，C]。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>表示池化窗口大小。数值必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>表示池化操作的步长。数值必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>表示在输入的D、H、W方向上pads补0的层数。数值必须大于等于0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>可选属性</td>
      <td><ul><li>表示正向平均池化过程中推导的输出的shape是否向上取整（True表示向上取整）。</li><li>默认值为false。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>count_include_pad</td>
      <td>可选属性</td>
      <td><ul><li>计算正向平均池化时是否包括pads填充的0（True表示包括填充的0）。</li><li>默认值为true。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisor_override</td>
      <td>可选属性</td>
      <td><ul><li>表示取平均的除数。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效。</li><li>默认值为0。</li></ul></td>
      <td>INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td><ul><li>指定输入`grads`数据格式。</li><li>取值必须为["NDHWC","NCDHW"]之一，默认值为"NDHWC"。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>待进行AvgPool3DGrad计算的出参。shape需要与`orig_input_shape`的值一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

无。


## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_avgpool3d_backward.cpp](examples/test_aclnn_avgpool3d_backward.cpp) | 通过[aclnnAvgPool3dBackward](docs/aclnnAvgPool3dBackward.md)接口方式调用AvgPool3DGrad算子。 |
