# SegmentSum

English | [简体中文](./README.md)

## Applicable Products

| Product                                                        | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: sums the input tensor by segment indexes.
- Computation formula:
    $$
    y[i] = \sum_{\substack{j\\ \text{segment\_ids}[j] = i}} x[j]
    $$
    where the summation traverses all indexes `j` that satisfy `segment_ids[j] == i`, and accumulates the corresponding `x[j]` into `y[i]`. If a segment `i` has no corresponding elements, then `y[i] = 0`.
- Use case:

  Input tensor $x = \begin{bmatrix} [1 & 2] \\ [3 & 4] \\ [5 & 6] \\ [7 & 8] \end{bmatrix}$,
  segment index tensor $segment\_ids = [0, 0, 1, 2]$,

  Output tensor $y = \begin{bmatrix} [4 & 6] \\ [5 & 6] \\ [7 & 8] \end{bmatrix}$
  - `segment_ids` must be sorted in ascending order
  - `segment_ids` is the segment index, indicating which segment the value of the current segment belongs to
  - `segment_ids` values must be >= 0
  - The output shape is `[max(segment_ids) + 1, x.shape[1:]]`

## Parameters

| Parameter name        | Input/Output/Attribute | Description         | Data type                                                       | Data format |
| ------------- | -------------- | ------------ | -------------------------------------------------------------- | -------- |
| x             | Input           | Input data, that is, `x` in the formula             | FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, UINT32, UINT64       | ND       |
| segment_ids   | Input           | Segment index, that is, `segment_ids` in the formula  | INT32, INT64                                                   | ND       |
| y             | Output           | Output value information, that is, `y` in the formula           | FLOAT32, FLOAT16, BFLOAT16, INT32, INT64, UINT32, UINT64       | ND       |

## Constraints

**x**:

- At least 1 dimension (rank >=1).

**segment_ids**:

- Must be of the INT32 or INT64 type.
- Must be a 1D tensor, and `segment_ids.shape[0] = x.shape[0]`.
- Values must be sorted in ascending order, and `segment_ids.value >= 0`.

**y**:

- The type must be the same as x.
- The dimensions are the same as x, and the shape is `[max(segment_ids) + 1, x.shape[1:]]`.

## Invocation Description

| Invocation method   | Invocation sample           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_segment_sum](examples/test_geir_segment_sum.cpp) | Calls the SegmentSum operator by constructing a graph through the [operator IR](op_graph/segment_sum_proto.h). |
