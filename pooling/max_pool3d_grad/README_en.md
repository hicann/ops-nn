# MaxPool3DGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|×|
|Atlas A2 products|×|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

Operator function: Performs backpropagation after the forward max pooling, backfills the gradient at the coordinates of the maximum value of each window, and accumulates the gradients at the same coordinates.

**Formula for deriving the forward output shape**

The output shape of the forward MaxPool3D (that is, the shape of `orig_y` and `grads`) is jointly determined by the input shape, ksize, strides, pads, and the padding mode. Taking the NCDHW format as an example, the shape of the input `orig_x` is $(N, C, D_{in}, H_{in}, W_{in})$, and the forward output shape is $(N, C, D_{out}, H_{out}, W_{out})$. The formulas for computing the dimensions are as follows (ceil_mode defaults to False, rounding down):

**CALCULATED mode:**

$$
D_{out} = \left\lfloor \frac{D_{in} + 2 \times pads[0] - (k_d - 1) - 1}{s_d} \right\rfloor + 1
$$

$$
H_{out} = \left\lfloor \frac{H_{in} + 2 \times pads[2] - (k_h - 1) - 1}{s_h} \right\rfloor + 1
$$

$$
W_{out} = \left\lfloor \frac{W_{in} + 2 \times pads[4] - (k_w - 1) - 1}{s_w} \right\rfloor + 1
$$

**VALID mode:**

It is equivalent to the CALCULATED mode with all pads being 0, that is:

$$
D_{out} = \lfloor{\frac{D_{in} - {(k_d - 1) - 1}}{s_d}}\rfloor + 1
$$

$$
H_{out} = \lfloor{\frac{H_{in} - {(k_h - 1) - 1}}{s_h}}\rfloor + 1
$$

$$
W_{out} = \lfloor{\frac{W_{in} - {(k_w - 1) - 1}}{s_w}}\rfloor + 1
$$

**SAME mode:**

$$
D_{out} = \lfloor{\frac{D_{in} + {s_d} - 1}{s_d}}\rfloor
$$

$$
H_{out} = \lfloor{\frac{H_{in} + {s_h} - 1}{s_h}}\rfloor
$$

$$
W_{out} = \lfloor{\frac{W_{in} + {s_w} - 1}{s_w}}\rfloor
$$

**Output shape:**

The output `y` of the backpropagation has exactly the same shape as the input `orig_x`:

$$
y.shape = orig\_x.shape = (N, C, D_{in}, H_{in}, W_{in})
$$

**Parameter description:**

- $k_d, k_h, k_w$: window sizes of ksize in the D, H, and W dimensions
- $s_d, s_h, s_w$: strides in the D, H, and W dimensions

## Parameters

<table style="undefined;table-layout: fixed; width: 1300px"><colgroup>
  <col style="width: 150px">
  <col style="width: 150px">
  <col style="width: 500px">
  <col style="width: 300px">
  <col style="width: 200px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>orig_x</td>
      <td>Input</td>
      <td>Input parameter for the MaxPool3DGrad computation, representing the forward input tensor.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>orig_y</td>
      <td>Input</td>
      <td>Index positions of the maximum elements in the forward input. The data format must be the same as that of `orig_x`, and the shape must be the same as that of `grads`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td>Input parameter for the MaxPool3DGrad computation, representing the gradient of the current node.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Window size of the max pooling.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Stride of the pooling operation.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Optional attribute</td>
      <td>Pad mode. SAME, VALID, and CALCULATED are supported. The padded data is -inf, and the default value is SAME.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Takes effect when the pad mode is CALCULATED. It specifies the padding amounts at the front and back in the D/H/W dimensions, in the format of [d_front, d_back, h_front, h_back, w_front, w_back].</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Supported data formats.</td>
      <td>-</td>
      <td>NCDHW and NDHWC are supported</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the MaxPool3DGrad computation. The shape and data format must be the same as those of `orig_x`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC</td>
    </tr>
  </tbody></table>

## Constraints

- **ksize**: The array length must be 5, and the values corresponding to the N and C dimensions must be 1, that is, `(1, 1, k_d, k_h, k_w)`.
- **strides**: The array length must be 5, and the values corresponding to the N and C dimensions must be 1, that is, `(1, 1, s_d, s_h, s_w)`.
- **pads**: It takes effect only when `padding="CALCULATED"`. The array length must be 6, in the format of `[d_front, d_back, h_front, h_back, w_front, w_back]`, and each value must be >= 0. The front pads (pads[0], pads[2], and pads[4]) must satisfy <= the corresponding dimension kernel_size / 2.
- The shapes of **orig_y** and **grads** must be the same and must match the output shape of the forward MaxPool3D.
- The shape of **y** must be the same as the shape of `orig_x`.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode API | [test_geir_max_pool3d_grad](examples/test_geir_max_pool3d_grad.cpp) | Invoke the MaxPool3DGrad operator by building a graph through the IR [MaxPool3DGrad](./op_graph/max_pool3d_grad_proto.h). |
