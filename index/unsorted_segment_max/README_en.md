# UnsortedSegmentMax

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: computes the maximum value of the input tensor by segment.
- Formula:

$$
output[i]={\max}_{j...}data[j...]
$$
*max* returns the maximum value among the elements *j*..., where *segment_ids*[*j*...]==*i*.

- Use case:

  Input tensor $data = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [4&2&1] \end{bmatrix}$,
  segment index tensor $segment\_ ids = [0, 1, 1]$,
  $num\_ segments= 2$,

  Output tensor $output = \begin{bmatrix} [1&2&3] \\ [4&5&6] \end{bmatrix}$
  - `segment_ids` does not need to be sorted; it is the segment index, indicating which segment the value of the current segment is added to
  - `segment_ids` does not need to cover all values in the valid range
  - For `segment_ids`, if a given ID i is empty, the output is the minimum value of the data type
  - For `segment_ids`, if a given ID i is negative, that value is dropped and not added to the segment result
  - `num_segments` is a scalar and denotes the number of output segments; it should be greater than or equal to the actual number of segments

## Parameters

| Type | Name | Description | Supported data types | Data format |
| --- | --- | --- | --- | ---|
| Input|data|The input data|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64| ND |
| Input|segment_ids|The segment index|INT32, INT64| ND |
| Input|num_segments|The number of segments |INT32, INT64| ND |
| Output|value|The output value information|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64 | ND |

## Constraints

**data**:

  * It must be shape-compatible with segment_ids (data.dim_size(0) == segment_ids.NumElements()).
  * It has at least 1 dimension (rank >=1).

**segment_ids**:

  * It must be of the INT32 or INT64 type.
  * Its shape must be a prefix of data.shape (that is, segment_ids.shape is a subset of data.shape[:segment_ids.rank]).
  * Its values must be >= 0 and < num_segments (negative values are ignored).

**num_segments**:

  * It must be of the INT32 or INT64 type and > 0.
  * It should equal or exceed the maximum segment_id + 1.

## Invocation Description

  | Invocation method | Sample code | Description |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | Graph-mode invocation | [test_geir_unsorted_segment_max](examples/test_geir_unsorted_segment_max.cpp) | Calls the UnsortedSegmentMax operator by building the graph through the [operator IR](op_graph/unsorted_segment_max_proto.h). |
