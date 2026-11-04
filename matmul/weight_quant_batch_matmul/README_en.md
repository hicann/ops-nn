# WeightQuantBatchMatmul

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|×|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|×|
|Atlas training products|×|

## Function

- Operator function: performs a matrix multiplication computation in which one input is in the pseudo-quantization scenario, and can additionally perform quantization computation on the output.
- Formula:

  $$
  result = self @ ANTIQUANT(mat2) + bias
  $$

  $mat2$ in the formula is the input in the pseudo-quantization scenario, and its dequantization formula $ANTIQUANT(mat2)$ is

  $$
  ANTIQUANT(mat2) = (mat2 + antiquantOffset) * antiquantScale
  $$

  When the output needs to be quantized, the quantization formula is

  $$
  \begin{aligned}
  result &= QUANT(self @ ANTIQUANT(mat2) + bias) \\
  &= (self @ ANTIQUANT(mat2) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  When the output does not need to be quantized, the computation formula is

  $$
  result = self @ ANTIQUANT(mat2) + bias
  $$

## Parameters

<table class="tg"><thead>
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
      <td>Input self in the formula.</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input mat2 in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Input</td>
      <td>Input bias in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>antiquantScale</td>
      <td>Input</td>
      <td>Scale factor in the dequantization parameters, corresponding to antiquantScale in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>antiquantOffset</td>
      <td>Input</td>
      <td>Offset factor in the dequantization parameters, corresponding to antiquantOffset in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantScale</td>
      <td>Input</td>
      <td>Scale factor in the quantization parameters, corresponding to quantScale in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantOffset</td>
      <td>Input</td>
      <td>Offset factor in the quantization parameters, corresponding to quantOffset in the formula.</td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>result in the formula.</td>
      <td>FLOAT16,INT8</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface  | [test_aclnn_weight_quant_batch_mat_mul](examples/test_aclnn_weight_quant_batch_mat_mul.cpp) | Invokes the WeightQuantBatchMatmul operator through<br>[aclnnWeightQuantBatchMatmul](docs/aclnnWeightQuantBatchMatmul_en.md)<br>. |
