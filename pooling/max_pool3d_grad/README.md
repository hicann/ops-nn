# MaxPool3DGrad

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

算子功能：正向最大池化后反向传播，将梯度回填到每个窗口最大值的坐标处，相同坐标处累加。

**正向输出Shape推导公式**

正向MaxPool3D的输出Shape（即`orig_y`和`grads`的Shape）由输入Shape、ksize、strides、pads和padding模式共同决定。以NCDHW格式为例，输入`orig_x`的Shape为 $(N, C, D_{in}, H_{in}, W_{in})$，正向输出Shape为 $(N, C, D_{out}, H_{out}, W_{out})$，各维度计算公式如下（ceil_mode默认为False，向下取整）：

**CALCULATED模式：**

$$
D_{out} = \left\lfloor \frac{D_{in} + 2 \times pads[0] - (k_d - 1) - 1}{s_d} \right\rfloor + 1
$$

$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times pads[2] - (k_h - 1) - 1}{s_h} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2 \times pads[4] - (k_w - 1) - 1}{s_w} \right\rfloor + 1
$$

**VALID模式：**

等价于CALCULATED模式下 pads 全为0，即：

$$
D_{out} = \lfloor{\frac{D_{in} - {(k_d - 1) - 1}}{s_d}}\rfloor + 1
$$

$$
H_{out} = \lfloor{\frac{H_{in} - {(k_h - 1) - 1}}{s_h}}\rfloor + 1
$$

$$
W_{out} = \lfloor{\frac{W_{in} - {(k_w - 1) - 1}}{s_w}}\rfloor + 1
$$

**SAME模式：**

$$
D_{out} = \lfloor{\frac{D_{in} + {s_d} - 1}{s_d}}\rfloor
$$

$$
H_{out} = \lfloor{\frac{H_{in} + {s_h} - 1}{s_h}}\rfloor
$$

$$
W_{out} = \lfloor{\frac{W_{in} + {s_w} - 1}{s_w}}\rfloor
$$


**输出Shape：**

反向传播的输出`y`与输入`orig_x`的Shape完全一致：

$$
y.shape = orig\_x.shape = (N, C, D_{in}, H_{in}, W_{in})
$$

**参数说明：**
- $k_d, k_h, k_w$：ksize在D、H、W维度上的窗口大小
- $s_d, s_h, s_w$：strides在D、H、W维度上的步长

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
      <td>orig_x</td>
      <td>输入</td>
      <td>待进行MaxPool3DGrad计算的入参，表示正向的输入Tensor。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW，NDHWC</td>
    </tr>
    <tr>
      <td>orig_y</td>
      <td>输入</td>
      <td>表示正向输入中最大元素的索引位置。数据格式需要与`orig_x`一致，shape需要与`grads`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW，NDHWC</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>输入</td>
      <td>待进行MaxPool3DGrad计算的入参，表示当前节点的梯度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW，NDHWC</td>
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
      <td>可选属性</td>
      <td>pad模式，支持SAME、VALID和CALCULATED，填补的数据为-inf，默认SAME。</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>属性</td>
      <td>当pad模式为CALCULATED时生效，指定D/H/W三个维度前后方向的填充量，格式为[d_front, d_back, h_front, h_back, w_front, w_back]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>可选属性</td>
      <td>表示支持的数据格式。</td>
      <td>-</td>
      <td>支持NCDHW，NDHWC</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>待进行MaxPool3DGrad计算的出参。shape、数据格式需要与`orig_x`一致。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>NCDHW，NDHWC</td>
    </tr>
  </tbody></table>

## 约束说明

- **ksize**：数组长度必须为5，且N和C维度对应的值必须为1，即 `(1, 1, k_d, k_h, k_w)`。
- **strides**：数组长度必须为5，且N和C维度对应的值必须为1，即 `(1, 1, s_d, s_h, s_w)`。
- **pads**：仅在 `padding="CALCULATED"` 时生效，数组长度必须为6，格式为 `[d_front, d_back, h_front, h_back, w_front, w_back]`，各值 >= 0；前侧pad（pads[0]、pads[2]、pads[4]）需满足 <= 对应维度 kernel_size / 2。
- **orig_y** 和 **grads** 的Shape必须一致，且与正向MaxPool3D的输出Shape匹配。
- **y** 的Shape必须与 `orig_x` 的Shape一致。

## 调用说明

不涉及。
