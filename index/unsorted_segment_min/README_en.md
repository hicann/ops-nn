# UnsortedSegmentMin

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>    |    ×     |
| <term>Atlas A2 products</term> |    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Computes the minimum value of the input tensor by segments.
- Formula:
    $$
    output[i]={\min}_{j...}data[j...]
    $$
    min returns the minimum value among the elements *j*..., where segment_ids[*j*...]==*i*.
- Example:

  Input tensor $data = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [4&2&1] \end{bmatrix}$,
  segment index tensor $segment\_ids = [0, 1, 1]$,
  $num\_segments= 2$,

  Output tensor $output = \begin{bmatrix} [1&2&3] \\ [4&2&1] \end{bmatrix}$
  - `segment_ids` does not need to be sorted; it is the segment index, indicating which segment the value of the current segment belongs to
  - `segment_ids` does not need to cover all values in the entire valid range
  - If a given ID i in `segment_ids` is empty, the output is the maximum value of the data type
  - If a given ID i in `segment_ids` is negative, the value is deleted and does not participate in the minimum value computation
  - `num_segments` is a scalar, representing the number of segments of the output, and it should be greater than or equal to the actual number of segments

## Parameters

|Type| Name | Description |Supported data types | Data format |
| --- | --- | --- | --- | ---|
|Input|data|Input data|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64| ND |
|Input|segment_ids|Segment index|INT32, INT64| ND |
|Input|num_segments|Number of segments |INT32, INT64| ND |
|Output|value|Output value information|FLOAT32, INT32, INT64, BFLOAT16, FLOAT16, UINT32, UINT64 | ND |

## Constraints

**data**​:

- At least 1 dimension (rank >=1).

​**segment_ids**​:

- Must be of the INT32 or INT64 type.
- The shape must be a prefix of data.shape.
- The values must be >= 0 and < num_segments.

**num_segments**​:

- Must be of the INT32 or INT64 type, and > 0.
- It should be equal to or greater than the maximum segment_id + 1.

## Invocation Description

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | Graph-mode invocation | [test_geir_unsorted_segment_min](examples/test_geir_unsorted_segment_min.cpp) | Calls the UnsortedSegmentMin operator by constructing a graph through the [operator IR](op_graph/unsorted_segment_min_proto.h). |
