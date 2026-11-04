# BN3DTrainingReduceGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: BN3DTrainingReduceGrad is the elementwise closing operator of 3D Batch Normalization (BN3D, per-channel normalization on 5D tensors) training backpropagation. The operator broadcasts the 1D channel parameter tensors along the channel axis to 5D, and elementwise combines the gradient grads of the loss with respect to the BN forward output y, the diff_scale/diff_offset produced by the preceding reduction-stage BN3DTrainingUpdateGrad, and the forward statistics scale/batch_mean/batch_variance, to obtain the gradient dx of the loss with respect to the forward input x (output y). Both NCDHW and NDHWC layouts are supported (the channel axis is dim1 and dim4 respectively). The operator is commonly used as the last link of the BN3D training backward chain. This operator performs no cross-element reduction.

- Formula:

  $$
  y_{n,c,d,h,w} = \left( grads_{n,c,d,h,w} - \frac{diff\_scale_c \cdot (x_{n,c,d,h,w} - batch\_mean_c)}{num \cdot data\_sqrt_c} - \frac{diff\_offset_c}{num} \right) \cdot \frac{scale_c}{data\_sqrt_c}
  $$

  where:

  - num = N·D·H·W: the product of all dimensions except the channel count C (identical in both the NCDHW and NDHWC layouts);
  - data_sqrt_c = sqrt(batch_variance_c + epsilon): the square root after epsilon is added to the variance;
  - n, c, and (d, h, w) are the batch, channel, and spatial coordinates respectively, and the 1D parameter tensors are broadcast along the channel axis to 5D;
  - batch_variance uses the biased variance definition (E[x²]−E[x]²);
  - After the computation, the output y falls back to the dtype of grads (float16 falls back by cast, and bfloat16 falls back by round), with the same shape, format, and dtype as grads.

- float16/bfloat16 inputs are first upconverted to float32 inside the kernel to participate in all intermediate computations, and the results then fall back to the original dtype.

## Parameters

<table style="undefined;table-layout: fixed; width: 910px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 200px">
  <col style="width: 200px">
  <col style="width: 170px">
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
      <td>grads</td>
      <td>Input</td>
      <td>Gradient of the loss with respect to the BN forward output y, corresponding to grads in the formula. Its shape, data format, and data type must be exactly the same as those of x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>BN forward input, corresponding to x in the formula. Its shape, data format, and data type must be exactly the same as those of grads.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>Input</td>
      <td>Output of the preceding BN3DTrainingUpdateGrad and gradient of the loss with respect to scale, corresponding to diff_scale in the formula. The shape is (C,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>Input</td>
      <td>Output of the preceding BN3DTrainingUpdateGrad and gradient of the loss with respect to offset, corresponding to diff_offset in the formula. The shape is (C,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>BN forward scaling parameter γ, corresponding to scale in the formula. The shape is (C,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Input</td>
      <td>Batch mean of x (output of the preceding BN3DTrainingUpdate), corresponding to batch_mean in the formula. The shape is (C,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Input</td>
      <td>Batch variance of x (output of the preceding BN3DTrainingUpdate, biased definition), corresponding to batch_variance in the formula. The shape is (C,).</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Numerical stability constant added to batch_variance before the square root, corresponding to epsilon in the formula. It must be greater than 0.</li><li>The default value is 0.0001.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient dx of the loss with respect to the forward input x, corresponding to y in the formula. Its shape, data format, and data type are the same as those of grads.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>: The data format NDC1HWC0 is not supported.
- <term>Atlas A3 products</term>: The data format NDHWC is not supported.
- <term>Atlas A2 products</term>: The data format NDHWC is not supported.
- <term>Atlas 200I/500 A2 inference products</term>: The data type BFLOAT16 is not supported, and the data formats NCDHW and NDHWC are not supported.
- <term>Atlas inference products</term>: The data type BFLOAT16 is not supported, and the data format NDHWC is not supported.
- <term>Atlas training products</term>: The data type BFLOAT16 is not supported, and the data formats NCDHW and NDHWC are not supported.

## Constraints

- The shapes, data formats, and data types of grads, x, and y must be exactly the same.
- diff_scale, diff_offset, scale, batch_mean, and batch_variance must be 1D, with a length equal to the channel count C, and their data type must be FLOAT.
- batch_variance must use the biased variance definition (E[x²]−E[x]²).
- <term>Ascend 950PR&950DT products</term>:
  - grads, x, and y must all be 5D. The data formats NCDHW and NDHWC are supported (the channel axis is dim1 and dim4 respectively).
  - Empty tensors are not supported for grads, x, and y (an error is returned when any dimension is 0).
  - epsilon must be greater than 0.

## Invocation Description

| Invocation method | Invocation sample | Description |
| ---------------- | --------------- | ----------- |
| GE graph mode | [test_geir_bn3_d_training_reduce_grad](examples/arch35/test_geir_bn3_d_training_reduce_grad.cpp) | Calls the BN3DTrainingReduceGrad operator by building a graph through [operator IR](op_graph/bn3_d_training_reduce_grad_proto.h) |
