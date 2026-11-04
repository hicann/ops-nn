# BucketizeV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | x |
| <term>Atlas A2 products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | x |
| <term>Atlas inference products</term> | x |
| <term>Atlas training products</term> | x |

## Function

- Operator function: it discretizes the values in a tensor according to the given boundaries, maps each value in the input tensor to the corresponding bucket interval according to the given boundaries, and returns the index of the interval the value belongs to.
  - Computation flow:
    For the input value $x$ and the boundary array $boundaries = [b₀, b₁, ..., bₙ₋₁]$, the output index $yi$ satisfies:

    **When right=false (the default, left-open and right-closed intervals):**
    If $x \le b_0$, then $yi = 0$
    If $b_{i-1} < x \le b_i$, then $yi = i$
    If $x > b_{n-1}$, then $yi = n$

    **When right=true (left-closed and right-open intervals):**
    If $x < b_0$, then $yi = 0$
    If $b_{i-1} \le x < b_i$, then $yi = i$
    If $x \ge b_{n-1}$, then $yi = n$

    Note: the semantics of this operator are the opposite of `numpy.digitize` (with `right=False`, numpy uses left-closed and right-open intervals).

    **Example description:**
    Assume $boundaries = [1, 3, 5]$. For the input value $x=3$:
    - When right=false, $x$ satisfies $1 < 3 \le 3$ and falls into the interval $(1, 3]$; the output index is 1
    - When right=true, $x$ satisfies $3 \le 3 < 5$ and falls into the interval $[3, 5)$; the output index is 2

    For the boundary value $x=1$:
    - When right=false, $x$ satisfies $x \le 1$ and falls into $(-\infty, 1]$; the output index is 0
    - When right=true, $x$ satisfies $1 \le 1 < 3$ and falls into the interval $[1, 3)$; the output index is 1

## Parameters

<table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 215px">
  <col style="width: 163px">
  <col style="width: 287px">
  <col style="width: 439px">
  <col style="width: 135px">
  </colgroup>
  <thead>
      <tr>
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Data type</th>
        <th>Data format</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>Input</td>
        <td>The input tensor, the x in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>boundaries</td>
        <td>Input</td>
        <td>The input tensor, denoting the given boundaries, the boundaries in the formula.</td>
        <td>FLOAT, FLOAT16, BFLOAT16, INT8, INT16, INT32, INT64, UINT8</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>y</td>
        <td>Output</td>
        <td>The result of the input, the y in the formula.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>right</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies whether the interval includes the right boundary.</li><li>Default value: false.</li></ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
      <tr>
        <td>out_int32</td>
        <td>Optional attribute</td>
        <td><ul><li>Specifies the data type of the output. true indicates that the dtype of output y is int32, and false indicates that the dtype of output y is int64.</li><li>Default value: false.</li></ul></td>
        <td>BOOL</td>
        <td>ND</td>
      </tr>
    </tbody></table>

## Constraints

The values in boundaries must be in ascending order and must not repeat.

## Invocation Description

| Invocation method | Invocation sample | Description |
| --- | --- | --- |
| aclnn invocation | [test_aclnn_bucketize](examples/arch35/test_aclnn_bucketize.cpp) | Calls the BucketizeV2 operator through the [aclnnBucketize](./docs/aclnnBucketize_en.md) API. |
| Graph-mode invocation | [test_geir_bucketize_v2](./examples/arch35/test_geir_bucketize_v2.cpp) | Calls the BucketizeV2 operator by building the graph through the [operator IR](./op_graph/bucketize_v2_proto.h). |
