# MaxPoolGrad

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

Operator function: backpropagation of forward max pooling (MaxPool). The operator compares the forward input `x1` with the forward output `x2` to locate the coordinates of the maximum value in each pooling window, backfills the gradient in `grad` to that coordinate, and accumulates gradients at the same coordinate. The operator is compatible with the TensorFlow MaxPoolGrad operator.

**Forward output shape derivation formula**

The output shape of the forward MaxPool (that is, the shapes of `x2` and `grad`) is jointly determined by the input shape, ksize, strides, and the padding mode. Taking the NHWC format as an example, the shape of the input `x1` is $(N, H_{in}, W_{in}, C)$, and the forward output shape is $(N, H_{out}, W_{out}, C)$. The computation formulas of each dimension are as follows:

**SAME mode:**

$$
H_{out} = \lceil \frac{H_{in}}{s_h} \rceil
$$

$$
W_{out} = \lceil \frac{W_{in}}{s_w} \rceil
$$

**VALID mode:**

$$
H_{out} = \lceil \frac{H_{in} - (k_h - 1)}{s_h} \rceil
$$

$$
W_{out} = \lceil \frac{W_{in} - (k_w - 1)}{s_w} \rceil
$$

**Output shape:**

The output `y` of the backpropagation has exactly the same shape as the input `x1`:

$$
y.shape = x1.shape = (N, H_{in}, W_{in}, C)
$$

**Parameter description:**

- $k_h, k_w$: window size of ksize on the H and W dimensions
- $s_h, s_w$: stride of strides on the H and W dimensions

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
      <td>x1</td>
      <td>Input</td>
      <td>Input parameter for the MaxPoolGrad computation, denoting the input tensor of the forward MaxPool.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input parameter for the MaxPoolGrad computation, denoting the output tensor of the forward MaxPool. The data type and data format must be consistent with `x1`, and the shape must be consistent with `grad`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Input parameter for the MaxPoolGrad computation, denoting the gradient of the current node (the gradient of the forward output).</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Denotes the window size of max pooling.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Denotes the stride of the pooling operation.</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>Pad mode, supporting SAME and VALID. In SAME mode, the operator pads -inf at the input edges so that the output shape is ⌈input shape/stride⌉; VALID mode means no padding.</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Denotes the supported data format. The value must be one of ["NHWC","NCHW"], and the default value is "NHWC".</td>
      <td>-</td>
      <td>Supports NHWC, NCHW</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output parameter of the MaxPoolGrad computation, denoting the gradient of the input. The shape, data type, and data format must be consistent with `x1`.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- **ksize**: the array length must be 4, and the values corresponding to the N and C dimensions must be 1, that is, `(1, k_h, k_w, 1)` when `data_format="NHWC"` and `(1, 1, k_h, k_w)` when `data_format="NCHW"`; each element must be a positive integer.
- **strides**: the array length must be 4, and the values corresponding to the N and C dimensions must be 1, that is, `(1, s_h, s_w, 1)` when `data_format="NHWC"` and `(1, 1, s_h, s_w)` when `data_format="NCHW"`; each element must be a positive integer.
- **padding**: only SAME and VALID are supported.
- **data_format**: only NHWC (default) and NCHW are supported.
- **x1, x2, grad**: all must be 4-dimensional tensors, and the data types of the three tensors must be the same; only FLOAT16, FLOAT, and BFLOAT16 are supported; no dimension of `x1` can be 0.
- The shapes of **x2** and **grad** must be consistent and match the output shape of the forward MaxPool.
- The N and C dimension sizes of **grad** and **x1** must be consistent.
- The shape of **y** must be consistent with the shape of `x1`.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode interface | [test_geir_max_pool_grad](examples/test_geir_max_pool_grad.cpp) | Calls the MaxPoolGrad operator by building a graph through the IR [MaxPoolGrad](./op_graph/max_pool_grad_proto.h). |
