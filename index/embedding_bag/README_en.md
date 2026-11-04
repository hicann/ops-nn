# EmbeddingBag

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|×|
|Atlas inference products|√|
|Atlas training products|√|
|  <term>Kirin X90 processor series products</term> | √ |
|  <term>Kirin 9030 processor series products</term> | √ |

## Function

- Operator function: Obtains a group of values to be aggregated from weight according to indices, and then aggregates the obtained values with max, sum, or mean according to the offsets and the aggregation mode specified by mode. The other parameters provide finer control over the computation process.
  - The shape deduction is as follows:
    Assume:

    ```cpp
    The shape of weight is (numWeight, embeddingDim)
    The shape of indices is (indices)
    The shape of offsets is (offsets)
    ```

    - When mode is sum:

      ```cpp
      The shape of y is include_last_offset ? (offsets - 1, embeddingDim) : (offsets, embeddingDim)
      The shape of offset2bag is (indices,)
      The shape of bag_size is include_last_offset ? (offsets - 1) : (offsets,)
      The shape of max_indices is include_last_offset ? (offsets - 1) : (offsets,)
      ```

    - When mode is mean:

      ```cpp
      The shape of y is include_last_offset? (offsets - 1, embeddingDim) : (offsets, embeddingDim)
      The shape of offset2bag is (indices,)
      The shape of bag_size is include_last_offset ? (offsets - 1) : (offsets,)
      The shape of max_indices is include_last_offset ? (offsets - 1) : (offsets,)
      ```

    - When mode is max:

      ```cpp
      The shape of y is include_last_offset ? (offsets - 1, embeddingDim) : (offsets, embeddingDim)
      The shape of offset2bag is (indices,)
      The shape of bag_size is include_last_offset ? (offsets - 1) : (offsets,)
      The shape of max_indices is include_last_offset ? (offsets - 1, embeddingDim) : (offsets, embeddingDim)
      ```

## Parameters

  <table style="undefined;table-layout: fixed; width: 1250px"><colgroup>
    <col style="width: 50px">
    <col style="width: 160px">
    <col style="width: 350px">
    <col style="width: 200px">
    <col style="width: 80px">
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
      <td>weight</td>
      <td>Input</td>
      <td>Word embedding matrix, containing the embedding vectors of all words.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Tensor containing indices, specifying which words' embedding vectors to extract from weight.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offsets</td>
      <td>Optional Input</td>
      <td>Offset tensor used to split indices into multiple bags.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>per_sample_weights</td>
      <td>Optional Input</td>
      <td>Specifies the sample weights.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
     <tr>
      <td>y</td>
      <td>Output</td>
      <td>Result after aggregating the word embedding matrix.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>offset2bag</td>
      <td>Output</td>
      <td>Starting offsets of bags.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bag_size</td>
      <td>Output</td>
      <td>Size of each bag.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>max_indices</td>
      <td>Output</td>
      <td>When mode is max, the row where the maximum value of the word embedding vector is located.</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>Optional Attribute</td>
      <td><ul><li>Controls the aggregation mode.</li><li>Default value: "mean".</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_grad_by_freq</td>
      <td>Optional Attribute</td>
      <td><ul><li>Controls whether to scale gradients according to word frequency.</li><li>Default value: false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sparse</td>
      <td>Optional Attribute</td>
      <td><ul><li>Controls the sparse mode.</li><li>Default value: false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>include_last_offset</td>
      <td>Optional Attribute</td>
      <td><ul><li>Controls whether to include the last offset.</li><li>Default value: false.</li></ul></td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_idx</td>
      <td>Optional Attribute</td>
      <td><ul><li>Controls whether to include the last offset.</li><li>Default value: -1.</li></ul></td>
      <td>Int</td>
      <td>-</td>
    </tr>
  </tbody></table>

- Atlas training products: BFLOAT16 is not supported.
- Kirin X90/Kirin 9030 processor series products: BFLOAT16 is not supported.

## Constraints

None

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| aclnn invocation | [test_aclnn_embedding_bag](./examples/test_aclnn_embedding_bag.cpp) | Invokes the EmbeddingBag operator through the [aclnnEmbeddingBag](./docs/aclnnEmbeddingBag_en.md) interface. |
| Graph-mode invocation | [test_geir_embedding_bag](./examples/test_geir_embedding_bag.cpp)   | Builds the graph and invokes the EmbeddingBag operator through the [operator IR](./op_graph/embedding_bag_proto.h). |
