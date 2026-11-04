# INTrainingUpdateGrad

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | √ |
| <term>Atlas inference products</term> | √ |
| <term>Atlas training products</term> | √ |

## Function

- Operator function: the first stage of InstanceNorm training backpropagation (reduces the spatial dimensions and keeps per-(N, C) partial sums). It first normalizes `x` with the input `mean`/`variance`, then multiplies it with `dy` and sums along the spatial dimensions to obtain `res_gamma`; `dy` is summed directly along the spatial dimensions to obtain `res_beta`. **No reduction over N** (the reduction over N is completed by the separate downstream operator `INTrainingUpdateGradGammaBeta`).
- Formula (`ε` is the compile-time constant `1e-6`):

  $$
  x\_norm = (x - mean) \cdot \frac{1}{\sqrt{variance + \varepsilon}}
  $$

  $$
  res\_gamma = \sum_{spatial}(dy \cdot x\_norm), \quad res\_beta = \sum_{spatial}(dy)
  $$

## Parameters

<table style="table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>dy</td>
      <td>Input</td>
      <td>Denotes the gradient passed back, with complete spatial dimensions, corresponding to `dy` in the formula.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Denotes the input of the forward operator, with complete spatial dimensions, corresponding to `x` in the formula. Its data type matches the input `dy`.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>variance</td>
      <td>Input</td>
      <td>Denotes the variance of each instance, corresponding to `variance` in the formula. The spatial dimensions are 1.</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Denotes the mean of each instance, corresponding to `mean` in the formula. The spatial dimensions are 1.</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>res_gamma</td>
      <td>Output</td>
      <td>Denotes the partial sum of the gradient with respect to `gamma`, corresponding to `res_gamma` in the formula. The spatial dimensions are 1, and its shape matches the input `variance`.</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
    <tr>
      <td>res_beta</td>
      <td>Output</td>
      <td>Denotes the partial sum of the gradient with respect to `beta`, corresponding to `res_beta` in the formula. The spatial dimensions are 1, and its shape matches the input `variance`.</td>
      <td>FLOAT32</td>
      <td>NDC1HWC0</td>
    </tr>
  </tbody></table>

## Constraints

- Only graph-mode invocation is supported.
- The data format supports only NDC1HWC0 (6 dimensions), and all inputs and outputs are consistent.
- The shape of input `x` must be exactly the same as `dy`.
- The shapes of inputs `variance` and `mean` and outputs `res_gamma` and `res_beta` must be (N, 1, C1, 1, 1, C0), that is, determined by N, C1, and C0 of `dy`, with the reduction axes D/H/W folded to 1.
- The data types of inputs `dy` and `x` must be the same, both FLOAT16 or both FLOAT32. `variance`, `mean`, and the two outputs are always FLOAT32.
- When 0 exists in the reduction axes (D, H, W), all elements of `res_gamma` and `res_beta` are 0.
- The N, C1, and C0 dimensions must be greater than 0. 0 is not supported, and the operator refuses to execute.
- The C0 dimension must not be greater than 64 (the 64-lane limit of a single fp32 vector register). The operator refuses to execute when it is exceeded.
- The product of the dimensions must be within the uint32 range (D, H*W, C1, D*H*W, N*C1, H*W*C0, and N*C1*C0 must not exceed 2^32-1). The operator refuses to execute when it is exceeded.
- Dynamic shape/dynamic rank inputs: the output shape is passed through according to `variance`, and input validation is completed in the operator tiling stage.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| GE graph mode | [test_geir_in_training_update_grad](examples/test_geir_in_training_update_grad.cpp) | Calls the INTrainingUpdateGrad operator by building the graph through the [operator IR](op_graph/in_training_update_grad_proto.h). |
