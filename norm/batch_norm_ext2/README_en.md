# BatchNormExt2

English | [简体中文](./README.md)

## Applicable Products

| Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: performs batch normalization (Batch Normalization) along the channel dimension for the 4D input tensor.

- Formula:

  $$
  y = \frac{(x - E(x))}{\sqrt{Var(x) + ε}} * γ + β
  $$

  Here, in training mode, `E(x)` and `Var(x)` are computed from the statistics of the current batch over the spatial dimensions. In inference mode, `E(x)` and `Var(x)` take the inputs `input_mean` and `input_variance`. `ε` denotes an extremely small floating-point number to prevent the denominator from being 0.

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
      <td>input_x</td>
      <td>Input</td>
      <td>
      <ul><li>The input tensor for batch normalization, corresponding to <code>x</code> in the formula.</li><li>A 4D tensor whose shape is (N, C, H, W) or (N, H, W, C). In the ND format, the position of the C axis is interpreted according to the data_format attribute (NCHW→dim1, NHWC→dim3).</li></ul>
      </td>
      <td>FLOAT16, FLOAT32</td>
      <td>NCHW, NHWC, ND, NC1HWC0</td>
    </tr>
    <tr>
      <td>input_scale</td>
      <td>Input</td>
      <td>
      <ul><li>The weight for batch normalization, corresponding to <code>γ</code> in the formula.</li><li>A 1D tensor whose shape is the same as the channel dimension C of the input input_x.</li></ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_offset</td>
      <td>Input</td>
      <td><ul><li>The bias value for batch normalization, corresponding to <code>β</code> in the formula.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_mean</td>
      <td>Optional input</td>
      <td><ul><li>Training scenario: it must be empty. Inference scenario: the mean used during inference. It is a required input, corresponding to <code>E(x)</code> in the formula.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>input_variance</td>
      <td>Optional input</td>
      <td><ul><li>Training scenario: it must be empty. Inference scenario: the variance used during inference. It is a required input, corresponding to <code>Var(x)</code> in the formula.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>The small value added to the variance to avoid division by zero, corresponding to <code>ε</code> in the formula.</li><li>The default value is 1e-4f.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies the data format of the input input_x. "NHWC" and "NCHW" are supported.</li><li>The default value is "NHWC".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>is_training</td>
      <td>Optional attribute</td>
      <td><ul><li>Indicates whether it is the training scenario. true indicates the training scenario, and false indicates the inference scenario.</li><li>The default value is true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output_y</td>
      <td>Output</td>
      <td><ul><li>Represents the output result after batch normalization, corresponding to <code>y</code> in the formula.</li><li>The data type, data format, and shape are consistent with the input input_x.</li></ul></td>
      <td>FLOAT16, FLOAT32</td>
      <td>NCHW, NHWC, ND, NC1HWC0</td>
    </tr>
    <tr>
      <td>output_mean</td>
      <td>Output</td>
      <td><ul><li>Training mode: the mean (biased) of the current batch. Inference mode: equal to the input input_mean.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_variance</td>
      <td>Output</td>
      <td><ul><li>Training mode: the variance (unbiased, Bessel correction) of the current batch. Inference mode: equal to the input input_variance.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_reserve_space_1</td>
      <td>Output</td>
      <td><ul><li>Reserved for gradient computation. Training mode: the saved mean (equal to output_mean). Inference mode: equal to the input input_mean.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output_reserve_space_2</td>
      <td>Output</td>
      <td><ul><li>Reserved for gradient computation. Training mode: the saved biased variance E[(x-mean)^2]. Inference mode: equal to the input input_variance.</li><li>A 1D tensor whose shape is consistent with the input parameter input_scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Ascend 950PR&950DT products</term>: the supported data formats are NCHW, NHWC, and ND.
- <term>Atlas A3 products</term>: the supported data formats are NC1HWC0 and ND.
- <term>Atlas A2 products</term>: the supported data formats are NC1HWC0 and ND.
- <term>Atlas 200I/500 A2 inference products</term>: the supported data formats are NC1HWC0 and ND.
- <term>Atlas inference products</term>: the supported data formats are NC1HWC0 and ND.
- <term>Atlas training products</term>: the supported data formats are NC1HWC0 and ND.

## Constraints

- The input input_x supports only 4D tensors, and the supported data formats are NCHW, NHWC, and ND (the position of the C axis of the ND input is interpreted according to the data_format attribute).
- When the input input_x is in a specific format (NCHW/NHWC), it must be consistent with the data_format attribute. The ND input is not subject to this restriction.
- In training mode, the inputs input_mean and input_variance must be empty. In inference mode, the inputs input_mean and input_variance must be provided.
- Empty tensors are not supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_batch_norm_ext2](./examples/arch35/test_geir_batch_norm_ext2.cpp)   | Invoke the BatchNormExt2 operator by constructing a graph through the [operator IR](./op_graph/batch_norm_ext2_proto.h). |
