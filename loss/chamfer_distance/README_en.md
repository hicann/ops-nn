# ChamferDistance

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                       |    √     |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                              |    ×     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Operator function: Computes the Chamfer Distance between two sets of 2D points. For each point in the first set, the operator finds the closest point in the second set and outputs the closest distance and the index of the closest point. It then symmetrically performs the same operation for the second set. The distance is the squared Euclidean distance, and no square root is taken.

- Formula:

  Assume that the point sets are stored separately by coordinate plane, $xyz[0]$ holds all x coordinates and $xyz[1]$ holds all y coordinates, $B$ is the number of batches, and $N$ is the number of points in each point set. Let the $i$-th point in the $b$-th batch be $p_{b,i}=(x_{b,i}, y_{b,i})$. Then:

  $$
  d(b, i, j) = (x^1_{b,i} - x^2_{b,j})^2 + (y^1_{b,i} - y^2_{b,j})^2
  $$

  $$
  \text{dist1}_{b,i} = \min_{j} d(b, i, j), \quad \text{idx1}_{b,i} = \arg\min_{j} d(b, i, j)
  $$

  $$
  \text{dist2}_{b,j} = \min_{i} d(b, i, j), \quad \text{idx2}_{b,j} = \arg\min_{i} d(b, i, j)
  $$

  Here:

  - Batches are independent of each other, and the number of points $N$ in the two point sets must be the same.
  - When tied minimum values exist, the smallest index is taken.

## Parameters

| Parameter name | Input/Output/Attribute | Description                                                         | Data type        | Data format |
| ------ | -------------- | ------------------------------------------------------------ | --------------- | -------- |
| xyz1   | Input           | First set of 2D points, with shape $(2, B, N)$: xyz1[0] holds all x coordinates, and xyz1[1] holds all y coordinates. | FLOAT16, FLOAT, BFLOAT16 | ND       |
| xyz2   | Input           | Second set of 2D points. Both the shape and data type are the same as those of xyz1.                 | FLOAT16, FLOAT, BFLOAT16 | ND       |
| dist1  | Output           | Minimum squared distance from each point in xyz1 to xyz2, with shape $(B, N)$. The data type is the same as that of xyz1. | FLOAT16, FLOAT, BFLOAT16 | ND       |
| dist2  | Output           | Minimum squared distance from each point in xyz2 to xyz1, with shape $(B, N)$. The data type is the same as that of xyz1. | FLOAT16, FLOAT, BFLOAT16 | ND       |
| idx1   | Output           | Index of the closest point corresponding to dist1. The value range is $[0, N)$, and the shape is $(B, N)$.   | INT32           | ND       |
| idx2   | Output           | Index of the closest point corresponding to dist2. The value range is $[0, N)$, and the shape is $(B, N)$.   | INT32           | ND       |

## Constraints

- The shape of xyz1 is fixed to 3 dimensions with the first dimension being 2. The first dimension contains the two coordinate planes x and y, and is not the coordinate dimension of points.
- The shape and data type of xyz2 must be the same as those of xyz1, that is, the two point sets have the same number of batches and the same number of points.
- The data types of dist1 and dist2 are the same as that of xyz1. The data types of idx1 and idx2 are fixed to INT32.
- The shape of the outputs is determined by the first and second dimensions of xyz1, that is, $(B, N)$.
- BFLOAT16 is supported only by <term>Ascend 950PR&950DT products</term>. The data types supported by other products are FLOAT16 and FLOAT.
- When $B$ or $N$ of xyz1 is 0 (both point sets are empty at the same time), dist1, dist2, idx1, and idx2 output the corresponding empty tensors, and the operator returns normally.
- When coordinate values contain inf or nan, they participate in comparison and propagation according to IEEE rules, and no interception is performed.

## Invocation Description

| Invocation method | Sample code | Description |
| -------- | -------- | ---- |
| Graph-mode invocation | [test_geir_chamfer_distance.cpp](examples/test_geir_chamfer_distance.cpp) | Calls the ChamferDistance operator through graph construction with the [operator IR](./op_graph/chamfer_distance_proto.h). |
