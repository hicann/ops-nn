# MaxPoolGrad

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|×|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|×|
|Atlas 200I/500 A2 推理产品|×|
|Atlas 推理系列产品|×|
|Atlas 训练系列产品|×|

## 功能说明

算子功能：正向最大池化（MaxPool）的反向传播。通过比较正向输入`x1`和正向输出`x2`定位每个池化窗口内最大值所在的坐标，将`grad`中的梯度回填到该坐标处，相同坐标处梯度累加。兼容TensorFlow的MaxPoolGrad算子。

**正向输出Shape推导公式**

正向MaxPool的输出Shape（即`x2`和`grad`的Shape）由输入Shape、ksize、strides和padding模式共同决定。以NHWC格式为例，输入`x1`的Shape为 $(N, H_{in}, W_{in}, C)$，正向输出Shape为 $(N, H_{out}, W_{out}, C)$，各维度计算公式如下：

**SAME模式：**

$$
H_{out} = \lceil \frac{H_{in}}{s_h} \rceil
$$

$$
W_{out} = \lceil \frac{W_{in}}{s_w} \rceil
$$

**VALID模式：**

$$
H_{out} = \lceil \frac{H_{in} - (k_h - 1)}{s_h} \rceil
$$

$$
W_{out} = \lceil \frac{W_{in} - (k_w - 1)}{s_w} \rceil
$$

**输出Shape：**

反向传播的输出`y`与输入`x1`的Shape完全一致：

$$
y.shape = x1.shape = (N, H_{in}, W_{in}, C)
$$

**参数说明：**
- $k_h, k_w$：ksize在H、W维度上的窗口大小
- $s_h, s_w$：strides在H、W维度上的步长

## 参数说明

<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 300px">
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
      <td>x1</td>
      <td>输入</td>
      <td>待进行MaxPoolGrad计算的入参，表示正向MaxPool的输入Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>待进行MaxPoolGrad计算的入参，表示正向MaxPool的输出Tensor。数据类型、数据格式需要与`x1`一致，shape需要与`grad`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>输入</td>
      <td>待进行MaxPoolGrad计算的入参，表示当前节点的梯度（正向输出的梯度）。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>属性</td>
      <td>表示最大池化的窗口大小。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>属性</td>
      <td>表示池化操作的步长。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>属性</td>
      <td>pad模式，支持SAME和VALID。SAME模式会在输入边缘填充-inf，使输出Shape为⌈输入Shape/步长⌉；VALID模式表示不填充。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>表示支持的数据格式，取值必须为["NHWC","NCHW"]之一，默认值为"NHWC"。</td>
      <td>-</td>
      <td>支持NHWC、NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行MaxPoolGrad计算的出参，表示输入的梯度。shape、数据类型、数据格式需要与`x1`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## 约束说明

- **ksize**：数组长度必须为4，且N和C维度对应的值必须为1，即 `data_format="NHWC"` 时为 `(1, k_h, k_w, 1)`，`data_format="NCHW"` 时为 `(1, 1, k_h, k_w)`，各元素必须为正整数。
- **strides**：数组长度必须为4，且N和C维度对应的值必须为1，即 `data_format="NHWC"` 时为 `(1, s_h, s_w, 1)`，`data_format="NCHW"` 时为 `(1, 1, s_h, s_w)`，各元素必须为正整数。
- **padding**：仅支持SAME和VALID。
- **data_format**：仅支持NHWC（默认）和NCHW。
- **x1、x2、grad**：均必须为4维Tensor，且三者的数据类型必须一致，仅支持FLOAT16、FLOAT、BFLOAT16；`x1`各维度的大小不能为0。
- **x2** 和 **grad** 的Shape必须一致，且与正向MaxPool的输出Shape匹配。
- **grad** 与 **x1** 的N、C维度大小必须一致。
- **y** 的Shape必须与 `x1` 的Shape一致。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| 图模式接口 | [test_geir_max_pool_grad](examples/test_geir_max_pool_grad.cpp) | 通过IR[MaxPoolGrad](./op_graph/max_pool_grad_proto.h)构图方式调用MaxPoolGrad算子。 |
