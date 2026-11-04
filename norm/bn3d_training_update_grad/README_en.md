# BN3DTrainingUpdateGrad

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

- Operator function: BN3DTrainingUpdateGrad is the per-channel parameter gradient reduction operator of the training backpropagation of 3D Batch Normalization (BN3D, which normalizes 5D tensors by channel). The operator reduces the gradient grads of the loss on the BN forward output y together with the forward input x, combined with the per-channel statistics batch_mean and batch_variance, along the batch and spatial dimensions (N, D, H, W), and obtains the gradient diff_scale of the loss on the scaling parameter gamma and the gradient diff_offset on the bias parameter beta. The channel axis is determined by the data format of grads (dim 1 for NCDHW/NCHW, and the last axis for NHWC). It is commonly used in the parameter gradient collection step of the BN3D training backward chain.

- float16/bfloat16 inputs are first upcast to float32 inside the kernel to participate in all intermediate operations and reductions, and the outputs diff_scale/diff_offset are always float32.

- Formula:

  $$
  x\_norm_{n,c,d,h,w} = \frac{x_{n,c,d,h,w} - batch\_mean_c}{\sqrt{batch\_variance_c + epsilon}}
  $$

  $$
  diff\_scale_c = \sum_{n,d,h,w} grads_{n,c,d,h,w} \cdot x\_norm_{n,c,d,h,w}
  $$

  $$
  diff\_offset_c = \sum_{n,d,h,w} grads_{n,c,d,h,w}
  $$

  Here:

  - c is the channel coordinate, and n and (d, h, w) are the batch and spatial coordinates respectively; per-channel statistics broadcast along the channel axis;
  - batch_variance uses the biased variance convention (E[x²]−E[x]²), and epsilon stays consistent with the forward BN3D;
  - diff_scale and diff_offset are always float32, with shape equal to the number of channels C.

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
      <td>Gradient of the loss on the BN forward output y, that is, grads in the formula. The shape, data format, and data type must be exactly the same as those of x.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>NCDHW, NCHW, NHWC</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>BN forward input, that is, x in the formula. The shape, data format, and data type must be exactly the same as those of grads.</td>
      <td>FLOAT16, FLOAT, BFLOAT16</td>
      <td>NCDHW, NCHW, NHWC</td>
    </tr>
    <tr>
      <td>batch_mean</td>
      <td>Input</td>
      <td>Per-channel mean, that is, batch_mean in the formula. The shape is the number of channels C.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>batch_variance</td>
      <td>Input</td>
      <td>Per-channel biased variance, that is, batch_variance in the formula. The shape is the number of channels C.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Attribute</td>
      <td>Optional attribute. A small constant added to the variance, with the default value 0.0001, and it must stay consistent with the forward BN3D.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>diff_scale</td>
      <td>Output</td>
      <td>Gradient of the loss on the scaling parameter gamma, that is, diff_scale in the formula. The shape is the number of channels C.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>diff_offset</td>
      <td>Output</td>
      <td>Gradient of the loss on the bias parameter beta, that is, diff_offset in the formula. The shape is the number of channels C.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The data types, shapes, and data formats of grads and x must be exactly the same.
- The channel dimension of batch_mean and batch_variance must equal the number of channels C of grads, and the non-channel dimensions are always 1.
- The data formats of grads/x support only the named formats NCDHW, NCHW, and NHWC, and do not support ND.

## Invocation Description

| Invocation method | Invocation interface | Description |
| -------- | -------- | ---- |
| Graph-mode invocation (GEIR) | Build the graph through `op::BN3DTrainingUpdateGrad`; see `examples/arch35/test_geir_bn3d_training_update_grad.cpp` | Invoked through Ascend Graph (GE IR), which compiles and executes the graph. |
