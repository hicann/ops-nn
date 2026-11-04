# MultiAddRmsNormDynamicQuant

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: The RmsNorm operator is a normalization operation commonly used in large models. The DynamicQuant operator performs symmetric dynamic quantization on the input tensor. The MultiAddRmsNormDynamicQuant operator fuses the n Add operators before RmsNorm, the RmsNorm normalization output, and the DynamicQuant operator, reducing data move-in and move-out operations (n supports 0 to 4).
- Formula:

  $$
  x_{1}=x_{1a}+x_{1bOptional}+x_{1cOptional}+x_{1dOptional}+x_{1eOptional}  (x_{1} \text{ is a listOfTensor with a length of 1 to 5})
  $$

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - If neither smoothScale1Optional nor smoothScale2Optional has an input, the outputs y2 and scale2 have no practical meaning. The computation process is as follows:

  $$
   scale1=row\_max(abs(y))/127
  $$

  $$
   y1=round(y/scale1)
  $$

  - If only smoothScale1Optional has an input, the outputs y2 and scale2 have no practical meaning. The computation process is as follows:
  $$
    input = y\cdot smoothScale1Optional
  $$
  $$
   scale1=row\_max(abs(input))/127
  $$

  $$
   y1=round(input/scale1)
  $$

  - If both smoothScale1Optional and smoothScale2Optional have inputs, all five outputs of the operator are valid outputs. The computation process is as follows:
  $$
    input1 = y\cdot smoothScale1Optional
  $$
  $$
    input2 = y\cdot smoothScale2Optional
  $$
  $$
   scale1=row\_max(abs(input1))/127
  $$
  $$
   scale2=row\_max(abs(input2))/127
  $$
  $$
   y1=round(input1/scale1)
  $$
  $$
   y2=round(input2/scale2)
  $$

  where row\_max represents computing the maximum value of each row.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
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
      <td>x1</td>
      <td>Input</td>
      <td>List of source data tensors in the normalization process. It is a listOfTensor with a length of 1 to 5, corresponding to `x1` in the formula. The shape and data type of the tensors in the list must be the same, and the number of dimensions must not exceed 8. Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Source data tensor in the normalization process, corresponding to `x2` in the formula. The shape must be the same as that of each tensor in the `x1` list, and the data type must be the same as that of `x1`. Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight tensor in the normalization process, corresponding to `gamma` in the formula. The shape must be the same as the last dimension of `x1`, and the data type must be the same as that of `x1`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale1</td>
      <td>Optional input</td>
      <td>smoothScale tensor used to obtain y1 in the quantization process, corresponding to `smoothScale1Optional` in the formula. The shape must be the same as that of `gamma`, and the data type must be the same as that of `x1`.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>smooth_scale2</td>
      <td>Optional input</td>
      <td>smoothScale tensor used to obtain y2 in the quantization process, corresponding to `smoothScale2Optional` in the formula. The shape must be the same as that of `gamma`, and the data type must be the same as that of `x1`. Passing in only `smooth_scale2` without passing in `smooth_scale1` is not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td>Used to prevent division-by-zero errors, corresponding to `epsilon` in the formula. Default value: 1e-6. The value must be greater than zero.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y1` in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to `y2` in the formula. When `smooth_scale2` has no input, the shape of this output is fixed to [1] and the value has no practical meaning.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Sum of x after addition.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
        <tr>
      <td>y</td>
      <td>Output</td>
      <td>Normalization result of x.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale1</td>
      <td>Output</td>
      <td>Output of the first quantization path. The shape is the shape of the input `x1` after removing the last dimension ([1] when `x1` is 1-dimensional).</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scale2</td>
      <td>Output</td>
      <td>Output of the second quantization path. When `smooth_scale2` has no input, the shape of this output is fixed to [1] and the value has no practical meaning.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

  - <term>Ascend 950PR&950DT products</term>:

    The length of the last dimension (reduction axis) of the input parameter `x1` does not support 0.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph mode | [test_geir_multi_add_rms_norm_dynamic_quant](examples/test_geir_multi_add_rms_norm_dynamic_quant.cpp) | Invoke the MultiAddRmsNormDynamicQuant operator by building a graph through the [operator IR](op_graph/multi_add_rms_norm_dynamic_quant_proto.h). |
