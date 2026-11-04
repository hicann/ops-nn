# BN3DTrainingReduce

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     √    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- Operator function: BN3DTrainingReduce is used in the reduction phase of the 3D Batch Normalization training forward pass, and it is used in pair with BN3DTrainingUpdate. The operator computes the sum and square sum of the input `x` per channel.
- Formula (taking 5D NCDHW as an example):

  $$
  sum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}
  $$

  $$
  squareSum_{c} = \sum_{n=0}^{N-1} \sum_{d=0}^{D-1} \sum_{h=0}^{H-1} \sum_{w=0}^{W-1} x_{(n,c,d,h,w)}^2
  $$

  where $x$ is the input feature map, and $sum$ and $squareSum$ are per-C statistics. `square_sum` is the sum after element-wise squaring, not $(\sum x)^2$.

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
      <td>x</td>
      <td>Input</td>
      <td>Input <code>x</code> in the formula. For the shape requirements, see <a href="#constraints">Constraints</a>.</td>
      <td>FLOAT16, FLOAT32, BFLOAT16</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
    <tr>
      <td>sum</td>
      <td>Output</td>
      <td><code>sum</code> in the formula. For the shape requirements, see <a href="#constraints">Constraints</a>.</td>
      <td>FLOAT32</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
    <tr>
      <td>square_sum</td>
      <td>Output</td>
      <td><code>squareSum</code> in the formula. Its shape and data format are the same as those of <code>sum</code>.</td>
      <td>FLOAT32</td>
      <td>NCDHW, NDHWC, NDC1HWC0</td>
    </tr>
  </tbody></table>

## Constraints

- The data formats in the parameter table refer to the original formats, and the specific storage format is selected by the graph compilation flow. The data format and the number of shape dimensions of the input `x` must simultaneously meet the requirements of the corresponding product; otherwise, an error is reported.
- Empty tensor: when the channel axis size is 0, the output is empty, and other axes are allowed to be 0. When the channel axis is not 0, the N axis and each spatial reduction axis must have a size greater than 0.
- Dynamic shape and dynamic rank are supported: during graph compilation, `-1` indicates an unknown dimension and `[-2]` indicates an unknown rank; during execution, revalidation is performed with the concrete shape, and a concrete shape must not contain negative dimensions. When the rank is unknown, the output shape is [-1] for the original formats NCDHW and NDHWC, and the output shape is [1,1,-1,1,1,-1] for the NDC1HWC0 inference branch. This inference rule does not expand the set of original formats supported by each product.
- When the input contains NaN or Inf, the values are propagated to the two outputs of the corresponding channel according to IEEE 754 semantics. When the squaring or accumulation of finite values overflows FLOAT32, the output Inf is allowed.
- <term>Ascend 950PR&950DT products</term>:
  - Input data types: FLOAT16, FLOAT32, and BFLOAT16 are supported.
  - Input formats and shapes: NCDHW has 2 to 5 dimensions, with the shape [N, C, ...]; NDHWC has 5 dimensions, with the shape [N, D, H, W, C].
  - Output formats and shapes: both channel vector outputs use the NCDHW format, with the shape [C].
  - Constructing a graph directly with NDC1HWC0 as the input original format is not supported.
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - Input data types: FLOAT16, FLOAT32, and BFLOAT16 are supported.
  - The NCDHW input has 2 to 5 dimensions, with the shape [N, C, ...]; the shapes of both outputs are [C].
  - The NDHWC input has 5 dimensions, with the shape [N, D, H, W, C]; the shapes of both outputs are [C].
  - The NDC1HWC0 input has 6 dimensions, with the shape [N, D, C1, H, W, C0]; the shapes of both outputs are [1, 1, C1, 1, 1, C0].
- <term>Atlas training products</term>, <term>Atlas inference products</term>, and <term>Atlas 200I/500 A2 inference products</term>:
  - Input data types: FLOAT16 and FLOAT32 are supported, and BFLOAT16 is not supported.

## Invocation Description

This operator is an internal graph-mode operator. It provides only the operator IR graph construction entry and does not provide a public aclnn API.

| Invocation method   | Invocation sample           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| Graph-mode invocation  | [test_geir_bn3d_training_reduce](examples/arch35/test_geir_bn3d_training_reduce.cpp) | Calls the BN3DTrainingReduce operator by constructing a graph through the [operator IR](op_graph/bn3d_training_reduce_proto.h). |
