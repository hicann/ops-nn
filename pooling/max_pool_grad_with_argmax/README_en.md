# MaxPoolGradWithArgmax

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>                   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>   |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>                   |     ×    |
|  <term>Atlas inference products</term>                           |     ×    |
|  <term>Atlas training products</term>                           |     ×    |

## Function

- Operator function: Backward gradient computation of the forward max pooling MaxPoolWithArgmax.

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
      <td>Input x, with the shape [NCHW] or [NHWC]</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient tensor, with the shape [NCHW] or [NHWC]</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Input</td>
      <td>Index output by the forward max pooling, with the same shape as grad</td>
      <td>INT32, INT64</td>
      <td>NCHW, NHWC</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute (required)</td>
      <td>Pooling window size, a list of length 4. In NCHW format, ksize[0]=1 and ksize[1]=1; in NHWC format, ksize[0]=1 and ksize[3]=1</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute (required)</td>
      <td>Window moving stride, a list of length 4. In NCHW format, strides[0]=1 and strides[1]=1; in NHWC format, strides[0]=1 and strides[3]=1</td>
      <td>ListInt</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Attribute (required)</td>
      <td>Padding algorithm, with the value "SAME" or "VALID"</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>include_batch_in_index</td>
      <td>Attribute (optional)</td>
      <td>Specifies whether to include the batch dimension when computing argmax indexes. Currently only false is supported. Default: false</td>
      <td>Bool</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute (optional)</td>
      <td>Data layout format, with the value "NHWC" or "NCHW". Default: "NHWC"</td>
      <td>String</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Output gradient, with the same shape as x</td>
      <td>FLOAT16, BFLOAT16, FLOAT</td>
      <td>NCHW, NHWC</td>
    </tr>
  </tbody></table>

## Constraints

- include_batch_in_index: currently only false is supported

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode interface | [test_geir_max_pool_grad_with_argmax](examples/arch35/test_geir_max_pool_grad_with_argmax.cpp) | Invokes the MaxPoolGradWithArgmax operator through graph construction with the IR [MaxPoolGradWithArgmax](./op_graph/max_pool_grad_with_argmax_proto.h). |
