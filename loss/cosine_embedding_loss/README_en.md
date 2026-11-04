# CosineEmbeddingLoss

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Computes the cosine embedding loss of two input tensors. `x1` and `x2` are first broadcast to a common shape following the ND rules, and the cosine similarity reduction is always performed along dimension 1 (0-based `axis=1`) as the feature dimension; the shape after removing this dimension is then broadcast with `target` to obtain the element-wise loss. Elements equal to 1 in `target` are treated as similar samples, elements equal to -1 are treated as dissimilar samples, and other values output 0.
- Formula:

  $$
  cos_p = \frac{\sum_c x1_{p,c} x2_{p,c}}{\sqrt{\sum_c x1_{p,c}^2 + 1e-12}\sqrt{\sum_c x2_{p,c}^2 + 1e-12}}
  $$

  $$
  loss_p =
  \begin{cases}
 	  1 - cos_p, & target_p = 1 \\
 	  \max(0, cos_p - margin), & target_p = -1 \\
 	  0, & \text{otherwise}
  \end{cases}
  $$

When `reduction` is `none`, the element-wise loss is output, and the output shape is `broadcast(remove_axis_1(broadcast(x1.shape, x2.shape)), target.shape)`; when `sum` is used, the sum of the losses is output, and when `mean` is used, the mean of the losses is output. The output shape of both is `[1]`.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|---|---|---|---|---|
| x1 | Input | The first input tensor, which supports ND broadcasting with `x2`; after broadcasting, dimension 1 is the feature reduction dimension. | INT32, FLOAT16, FLOAT32 | ND |
| x2 | Input | The second input tensor, which supports ND broadcasting with `x1`; the data type must be the same as that of `x1`. | INT32, FLOAT16, FLOAT32 | ND |
| target | Input | The label tensor, which supports ND broadcasting with `remove_axis_1(broadcast(x1.shape, x2.shape))`; values 1 and -1 are supported, and other values are handled as 0 loss. | INT32, FLOAT16, FLOAT32 | ND |
| margin | Attribute | The margin of the dissimilar-sample branch; the default is 0.0. | FLOAT | - |
| reduction | Attribute | The reduction method, which supports `none`, `sum`, and `mean`; the default is `mean`. | STRING | - |
| y | Output | The loss output. `none` outputs the element-wise result, and `sum`/`mean` output shape `[1]`. | FLOAT32 | ND |

## Constraints

- `x1` and `x2` must be broadcastable, and the rank after broadcasting must be at least 2.
- The data types of `x1` and `x2` must be the same; `target` can independently use INT32, FLOAT16, or FLOAT32.
- `target` must be broadcastable to the shape of the `x1`/`x2` broadcast shape after removing dimension 1; extending the final element-wise output shape through additional leading dimensions is also supported.
- The input rank supports at most 8; the tiling stage requires runtime dimensions to be positive.
- `reduction` supports only `none`, `sum`, and `mean`.
- Inputs are processed in ND format.
- The current aclnn interface is not involved; call the operator through GE/operator IR.

## Invocation Description

| Invocation method | Invocation sample | Description |
|---|---|---|
| Graph-mode invocation | [test_geir_cosine_embedding_loss.cpp](examples/arch35/test_geir_cosine_embedding_loss.cpp) | Calls the `CosineEmbeddingLoss` operator by building a graph through the [operator IR](op_graph/cosine_embedding_loss_proto.h). |
