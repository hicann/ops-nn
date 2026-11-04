# ChamferDistanceGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Description: Backward operator of ChamferDistance. It computes the gradients of the inputs based on the contribution of the forward inputs to the output and the initial gradients.
- Formula:

  Assume that there are two point sets:  xyz1=[B,N,2], xyz2=[B,M,2]

  - The formula of the forward ChamferDistance operator is:

    $dist1_i=Min((x_{1_i}−x_2)^2+(y_{1_i}−y_2)^2), x_2, y_2∈xyz2$

    $dist2_i=Min((x_{2_i}-x_1)^2+(y_{2_i}-y_1)^2), x_1,y_1∈xyz1$

  - The backward operator computes the derivative of this formula. The computation formulas are:
    - The derivative of $dist1_i$ with respect to $x_{1_i}$ is $=2*grad\_dist1*(x_{1_i}-x_2)$

      Here, $x_{1_i}∈xyz1$, and $x_2$ is the x-coordinate of the point with the minimum distance taken from xyz2 based on the index value of idx1 output in the forward pass. The preceding is the single-point derivative formula. Because the positions updated by the single-point gradient are contiguous, parallel computation of multiple points is considered.

    - The derivative of $dist1_i$ with respect to $y_{1_i}$ is $=2*grad\_dist1*(y_{1_i}-y_2)$

      Here, $y_{1_i}∈xyz1$, and $y_2$ is the y-coordinate of the point with the minimum distance taken from xyz2 based on the index value of idx1 output in the forward pass. The preceding is the single-point derivative formula. Because the positions updated by the single-point gradient are contiguous, parallel computation of multiple points can also be considered.

    - The derivative of $dist1_i$ with respect to $x_2$ is $=-2*grad\_dist1*(x_1-x_{2_i})$

      Here, $x_{1_i}∈xyz1, x_2$ is the x-coordinate of the point with the minimum distance taken from xyz2 based on the index value of idx1 output in the forward pass. The preceding is the single-point derivative formula. Because the single-point gradient needs to be updated based on the index value corresponding to the minimum distance, this part cannot be computed in parallel and only supports single-point computation.

    - The derivative of $dist1_i$ with respect to $y_2$ is $=-2*grad\_dist1*(y_1-y_{2_i})$

      Here, $y_{1_i}∈xyz1, y_2$ is the y-coordinate of the point with the minimum distance taken from xyz2 based on the index value of idx1 output in the forward pass. The preceding is the single-point derivative formula. Because the single-point gradient needs to be updated based on the index value corresponding to the minimum value, this part also cannot be computed in parallel and only supports single-point computation.

  The derivatives of $dist2_i$ with respect to $x_{2_i}$, $x_1$, $y_{2_i}$, and $y_1$ are similar to the preceding process and are not repeated here.

  The final computation formulas are as follows, where i∈[0,n):

  $grad\_xyz1[2*i] = 2*grad\_dist1*(x_{1_i}-x_2) - 2*grad\_dist1*(x_1-x_{2_i})$

  $grad\_xyz1[2*i+1] = 2*grad\_dist1*(y_{1_i}-y_2) - 2*grad\_dist1*(y_1-y_{2_i})$

  $grad\_xyz2[2*i] =  2*grad\_dist2*(x_{1_i}-x_2) - 2*grad\_dist2*(x_1-x_{2_i})$

  $grad\_xyz2[2*i+1] = 2*grad\_dist2*(y_{1_i}-y_2) - 2*grad\_dist2*(y_1-y_{2_i})$

## Parameters

<table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
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
      <td>xyz1</td>
      <td>Input</td>
      <td>Input tensor of the forward operator.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>xyz2</td>
      <td>Input</td>
      <td>Input tensor of the forward operator.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx1</td>
      <td>Input</td>
      <td>Tensor output by the forward operator.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>idx2</td>
      <td>Input</td>
      <td>Tensor output by the forward operator.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_dist1</td>
      <td>Input</td>
      <td>Backward gradient of the forward output dist1, and also the initial gradient of the backward operator.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_dist2</td>
      <td>Input</td>
      <td>Backward gradient of the forward output dist2, and also the initial gradient of the backward operator.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_xyz1</td>
      <td>Output</td>
      <td>Gradient corresponding to the forward operator input xyz1 after the gradient update.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_xyz2</td>
      <td>Output</td>
      <td>Gradient corresponding to the forward operator input xyz2 after the gradient update.</td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

The shapes of xyz1 and xyz2 are (B, N, 2), and the shapes of grad_dist1, grad_dist2, idx1, and idx2 are (B, N). B and N in the shapes of all inputs must be consistent.

When it is invoked together with the forward `ChamferDistance`, the shapes of the forward `xyz1` and `xyz2` are `(2,B,N)`, while this backward interface requires `(B,N,2)`. Before invoking the backward interface, transpose the two forward inputs in the axis order `(1,2,0)` respectively and then pass them in.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_chamfer_distance_backward](./examples/test_aclnn_chamfer_distance_grad.cpp) | Invokes the ChamferDistanceGrad operator through the [aclnnChamferDistanceBackward](./docs/aclnnChamferDistanceBackward_en.md) interface.    |
