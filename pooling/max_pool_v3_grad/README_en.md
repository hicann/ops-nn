# MaxPoolV3Grad

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    x     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

## Function

- Operator function: MaxPoolV3Grad is the backward propagation (gradient computation) operator of the MaxPoolV3 (max pooling) operator. It is used during neural network training to back-propagate the gradient of the loss function with respect to the pooling output to the pooling input. For each pooling output position, its gradient is propagated only to the input position that produced the maximum value in the forward computation (the argmax position). The gradients of the other input positions are 0.

- The argmax position is determined by recomputing the maximum of each pooling window from orig_input (it does not depend on orig_output). When multiple input positions within a window equal the maximum value, only the first matching position receives the gradient (first-wins policy, consistent with TensorFlow).

- Formula:
  Backward propagation returns the gradient to the argmax position. For each pooling output position $(ho, wo)$, find the argmax position $(h^*, w^*)$ in the corresponding window of orig_input:

  $$
  out\_grad[n, c, h^*, w^*] \mathrel{+}= grad[n, c, ho, wo]
  $$

  $$
  out\_grad[n, c, h, w] \mathrel{+}= 0 \quad \text{for } (h, w) \neq (h^*, w^*)
  $$

  The H/W dimensions of out_grad are consistent with orig_input. The H/W dimensions of orig_output are derived according to padding_mode as follows. Let the pooling window size of the H dimension be $k_h$ with stride $s_h$, and the pooling window size of the W dimension be $k_w$ with stride $s_w$:

  - When padding_mode is "SAME", the output sizes are:

    $$
    H_{out} = \lceil \frac{H_{in}}{s_h} \rceil, \quad W_{out} = \lceil \frac{W_{in}}{s_w} \rceil
    $$

    The total padding sizes are:

    $$
    pad_h = \max((H_{out} - 1) \times s_h + k_h - H_{in}, 0), \quad pad_w = \max((W_{out} - 1) \times s_w + k_w - W_{in}, 0)
    $$

    where $pad_{top} = \lfloor pad_h / 2 \rfloor$, $pad_{bottom} = pad_h - pad_{top}$, $pad_{left} = \lfloor pad_w / 2 \rfloor$, and $pad_{right} = pad_w - pad_{left}$.

  - When padding_mode is "VALID", there is no padding, and the output sizes are:

    $$
    pad_{top} = pad_{bottom} = pad_{left} = pad_{right} = 0
    $$

    $$
    H_{out} = \lfloor \frac{H_{in} - k_h}{s_h} \rfloor + 1, \quad W_{out} = \lfloor \frac{W_{in} - k_w}{s_w} \rfloor + 1
    $$

  - When padding_mode is "CALCULATED", $pad_{top} = pads[0]$, $pad_{bottom} = pads[1]$, $pad_{left} = pads[2]$, and $pad_{right} = pads[3]$:

    - When ceil_mode is false, the output sizes are:

      $$
      H_{out} = \lfloor \frac{H_{in} + pad_{top} + pad_{bottom} - k_h}{s_h} \rfloor + 1
      $$

      $$
      W_{out} = \lfloor \frac{W_{in} + pad_{left} + pad_{right} - k_w}{s_w} \rfloor + 1
      $$

    - When ceil_mode is true, the output sizes are:

      $$
      H_{out} = \lceil \frac{H_{in} + pad_{top} + pad_{bottom} - k_h}{s_h} \rceil + 1
      $$

      $$
      W_{out} = \lceil \frac{W_{in} + pad_{left} + pad_{right} - k_w}{s_w} \rceil + 1
      $$

      If the start position at the top-left corner of a sliding window falls in the bottom-side/right-side padding area or out of bounds (no valid value can be obtained), the result of that sliding window is discarded, and the shape of the corresponding spatial axis is reduced by 1:

      $$
      \begin{cases}
      H_{out} = H_{out} - 1 & \text{if } (H_{out} - 1) \times s_h \ge H_{in} + pad_{top} \\
      W_{out} = W_{out} - 1 & \text{if } (W_{out} - 1) \times s_w \ge W_{in} + pad_{left}
      \end{cases}
      $$

  - When global_pooling is true, the pooling window covers the entire H/W dimensions:

    $$
    k_h = H_{in}, \quad k_w = W_{in}, \quad pad_{top} = pad_{bottom} = pad_{left} = pad_{right} = 0, \quad H_{out} = W_{out} = 1
    $$

## Parameters

|Parameter name|Input/Output/Attribute|Description|Data type|Data format|
|-----|-----------|----|---------|------|
|orig_input|Input|Original input of forward MaxPoolV3, a 4D tensor. When data_format is "NCHW", the layout is (N, C, H, W); when data_format is "NHWC", the layout is (N, H, W, C).|FLOAT16, FLOAT, BFLOAT16, INT32, INT64, UINT8, INT16, INT8, UINT16|ND|
|orig_output|Input|Original output of forward MaxPoolV3, a 4D tensor. The shape is derived from orig_input and ksize/strides/padding, and the dtype is the same as orig_input.|FLOAT16, FLOAT, BFLOAT16, INT32, INT64, UINT8, INT16, INT8, UINT16|ND|
|grad|Input|Output gradient, a 4D tensor. Its shape and dtype are exactly the same as those of orig_output.|FLOAT16, FLOAT, BFLOAT16, INT32, INT64, UINT8, INT16, INT8, UINT16|ND|
|out_grad|Output|Input gradient, a 4D tensor. Its shape and dtype are the same as those of orig_input.|FLOAT16, FLOAT, BFLOAT16, INT32, INT64, UINT8, INT16, INT8, UINT16|ND|
|ksize|Attribute|Pooling window size, a list of length 4. The N and C dimensions must be 1, and the value range of the H/W dimensions is [1, 255]. It is ignored when global_pooling is true.|INT64|-|
|strides|Attribute|Pooling stride, a list of length 4. The N and C dimensions must be 1, and the value range of the H/W dimensions is [1, 63].|INT64|-|
|padding_mode|Attribute|Padding mode. Supports "CALCULATED", "SAME", and "VALID", with the default value "CALCULATED".|STRING|-|
|pads|Attribute|Padding size, a list of length 4 in the order [pad_top, pad_bottom, pad_left, pad_right]. It takes effect only when padding_mode is "CALCULATED". The value range of pad_top/pad_bottom is [0, ksize[H]), and the value range of pad_left/pad_right is [0, ksize[W]). The default value is [0, 0, 0, 0].|INT64|-|
|data_format|Attribute|Logical data format. Supports "NCHW" and "NHWC", with the default value "NCHW". It determines the index positions of the H, W, and C dimensions. Both input and output are in 4D ND format, and no 5HD format conversion is performed.|STRING|-|
|global_pooling|Attribute|Whether to perform global pooling, with the default value false. When it is true, ksize and pads are ignored, the pooling window covers the entire H/W dimensions, and the output H/W dimensions are 1.|BOOL|-|
|ceil_mode|Attribute|Whether to use the ceil mode to compute output sizes, with the default value false. It takes effect only when padding_mode is "CALCULATED".|BOOL|-|

## Constraints

- **Data format constraints:**
  - The input and output tensors are in 4D ND format and support two logical layouts, NCHW and NHWC, which are specified by the data_format attribute.
  - The data_format attribute (NCHW/NHWC) determines the index positions of the H, W, and C dimensions. The operator directly computes 4D addresses based on this format internally, and does not perform 5HD format conversion.
  - Non-contiguous tensors are automatically converted to contiguous ones through AutoContiguous before computation.

- **Data type constraints:**
  - The three inputs orig_input, orig_output, and grad must use the same dtype, and the output out_grad also uses the same dtype.
  - Nine dtypes are supported: FLOAT16, FLOAT, BFLOAT16, INT32, INT64, UINT8, INT16, INT8, and UINT16.
  - Cross-type computation is not supported.

- **Shape constraints:**
  - The shape of grad must be exactly the same as the shape of orig_output.
  - The N and C dimensions of orig_output must be consistent with those of orig_input (the N/C dimensions remain unchanged during pooling).
  - The shape of out_grad is exactly the same as the shape of orig_input.

- **ksize constraints:**
  - The length of the ksize array must be 4.
  - The ksize values of the N and C dimensions must be 1 (they do not participate in pooling). In NCHW format, ksize[0] and ksize[1] are 1; in NHWC format, ksize[0] and ksize[3] are 1.
  - The value range of the H/W dimension ksize is [1, 255]. An error is reported if the value is out of range.
  - When global_pooling is true, ksize is ignored, and the pooling window takes the input H/W dimension sizes.

- **strides constraints:**
  - The length of the strides array must be 4.
  - The strides values of the N and C dimensions must be 1. In NCHW format, strides[0] and strides[1] are 1; in NHWC format, strides[0] and strides[3] are 1.
  - The value range of the H/W dimension strides is [1, 63]. An error is reported if the value is out of range.

- **padding_mode constraints:**
  - Only "CALCULATED", "SAME", and "VALID" are supported. An error is reported for other values.
  - The pads attribute takes effect only when padding_mode is "CALCULATED". The length of pads must be 4, in the order [pad_top, pad_bottom, pad_left, pad_right], and each value must be greater than or equal to 0.
  - Each pads value must be smaller than the ksize of the corresponding spatial axis. That is, pad_top and pad_bottom are smaller than ksize[H], and pad_left and pad_right are smaller than ksize[W]. An error is reported if a value is out of range. This constraint is consistent with forward MaxPoolV3. When global_pooling is true, pads is ignored and this check is not performed.
  - ceil_mode takes effect only when padding_mode is "CALCULATED". In SAME and VALID modes, ceil_mode is ignored.

- **global_pooling constraints:**
  - global_pooling supports true and false. When it is true, the pooling window covers the entire H/W dimensions, the output H/W dimensions are 1, and ksize and pads are ignored.

- **Special value handling:**
  - NaN input: NaN propagation semantics are used, consistent with MaxPoolGrad. NaN participates in the maximum value comparison and always wins. When a NaN exists in a window, the gradient is propagated to the last NaN position in that window in row-major scan order. When all values in a window are NaN, the gradient is propagated by the same rule.
  - Inf input: Inf participates in comparison and gradient propagation normally as a maximum value.
  - +0.0 and -0.0 follow the IEEE 754 standard and compare as equal.
  - When multiple input positions have the same maximum value and none of them is NaN, only the first maximum value position receives the gradient (first-wins policy).
  - The padding area does not receive gradients.

- **Boundary condition handling:**
  - Empty tensor (a shape contains a dimension of 0): the shape of out_grad is consistent with orig_input, and the operator returns safely without illegal access. When orig_input is empty, out_grad is an empty tensor. When orig_input is not empty but the pooling output is empty (no gradient is propagated), out_grad is all 0.
  - Non-contiguous tensors: they are automatically converted to contiguous ones through AutoContiguous before computation.
  - When ceil_mode is true, some windows may exceed the input boundaries. The out-of-bounds parts do not receive gradients.

- **Deterministic guarantee:**
  - Each thread exclusively owns one out_grad element for gradient aggregation. There is no concurrent write to the same address, so determinism is guaranteed without atomic addition.
  - The argmax selection rule is unique: take the first maximum value for non-NaN values, and take the last NaN when the window contains NaN. The result is reproducible.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---------|---------|------|
| GE graph mode | [test_geir_max_pool_v3_grad](examples/test_geir_max_pool_v3_grad.cpp) | Calls the MaxPoolV3Grad operator by constructing a graph through [operator IR](op_graph/max_pool_v3_grad_proto.h). For details, see [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
