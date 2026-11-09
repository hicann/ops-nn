# MaxPoolGradGradWithArgmax

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>                   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>   |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>                   |     ×    |
|  <term>Atlas inference products</term>                           |     ×    |
|  <term>Atlas training products</term>                           |     ×    |

## Function

- Operator function: This is the second-order gradient backpropagation operator of max pooling (MaxPool). It fixes the local index mode (include_batch_in_index=false) and gathers values from the `grad` tensor into the `output` tensor according to the `argmax` indices, using in-batch local indices.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
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
      <td>x</td>
      <td>Input</td>
      <td>Original pooling input tensor, with shape [N, H, W, C] and fixed NHWC format.</td>
        <td>DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>grad</td>
        <td>Input</td>
        <td>Gradient tensor of x, with the same shape and data type as x.</td>
         <td>DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16</td>
       <td>ND</td>
     </tr>
     <tr>
       <td>argmax</td>
      <td>Input</td>
      <td>Pooling maximum value index tensor. Its shape is the pooled output shape, and its values are in-batch local indices.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Second-order gradient output. Its shape is the same as argmax, and its data type is the same as x.</td>
        <td>DOUBLE, FLOAT32, FLOAT16, INT16, INT32, INT64, INT8, UINT16, UINT32, UINT64, UINT8, BFLOAT16</td>
       <td>ND</td>
     </tr>
     <tr>
       <td>ksize</td>
      <td>Attribute</td>
      <td>Pooling window size. It is a list of length 4. ksize[0]=1 and ksize[3]=1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Window movement stride. It is a list of length 4. strides[0]=1 and strides[3]=1.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute</td>
      <td>Padding algorithm. The value is "SAME" or "VALID".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input supports only 4 dimensions with fixed NHWC format.
- x.shape == grad.shape, and argmax.shape == y.shape.
- ksize[0] == 1 and strides[0] == 1 (the batch dimension is not pooled), and ksize[3] == 1 and strides[3] == 1 (the channel dimension is not pooled).
- argmax uses in-batch local indices (include_batch_in_index=false), with values in the range [0, H*W*C).
- Empty tensors are supported (when totalElements == 0, an empty output is returned directly).
- The data types of x, grad, and y must be the same.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_max_pool_grad_grad_with_argmax](examples/arch35/test_geir_max_pool_grad_grad_with_argmax.cpp) | Invokes the MaxPoolGradGradWithArgmax operator through the IR-based [MaxPoolGradGradWithArgmax](./op_graph/max_pool_grad_grad_with_argmax_proto.h) graph construction method. For details, see [Operator Invocation](../../docs/zh/invocation/quick_op_invocation.md). |
