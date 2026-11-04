# SparseApplyFtrl

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: SparseApplyFtrl is the sparse update operator of the FTRL-Proximal (Follow The Regularized Leader) optimizer. It selects the corresponding rows of `var`, `accum`, and `linear` according to `indices`, and performs the FTRL-Proximal update using the gradient `grad`. This operator is widely used for sparse feature weight updates in scenarios such as recommendation systems and CTR estimation.

- Formula:

$$
\text{accum\_new} = \text{accum} + \text{grad}^2
$$

$$
\sigma = \frac{\text{accum}^{-\text{lr\_power}} - \text{accum\_new}^{-\text{lr\_power}}}{\text{lr}}
$$

$$
\text{linear\_new} = \text{linear} + \text{grad} + \sigma \cdot \text{var}
$$

$$
x = \text{clamp}(\text{linear\_new}, -l1, l1) - \text{linear\_new}
$$

$$
y = \frac{\text{accum\_new}^{-\text{lr\_power}}}{\text{lr}} + 2 \cdot l2
$$

$$
\text{var\_new} = \frac{x}{y}
$$

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
      <td>var</td>
      <td>Input</td>
      <td>Variable tensor to be updated, with ≥2 dimensions</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>Gradient square accumulator, with the same shape as var</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>Input</td>
      <td>Linear accumulator, with the same shape as var</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Gradient tensor, shape[1:] == var.shape[1:]</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Index vector, 1-D, shape[0] == grad.shape[0]</td>
      <td>INT32, INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate, scalar tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l1</td>
      <td>Input</td>
      <td>L1 regularization coefficient, scalar tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>l2</td>
      <td>Input</td>
      <td>L2 regularization coefficient, scalar tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr_power</td>
      <td>Input</td>
      <td>Power of the learning rate, scalar tensor</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>var</td>
      <td>Output</td>
      <td>Updated variable, with the same shape/dtype as the input var</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Output</td>
      <td>Updated accumulator, with the same shape/dtype as the input accum</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>linear</td>
      <td>Output</td>
      <td>Updated linear accumulator, with the same shape/dtype as the input linear</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Specifies whether to use a lock to protect the update operation (not used in the current implementation)</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

1. var/accum/linear must have the same shape and dtype (float32).
2. grad.shape[1:] must equal var.shape[1:], and grad.shape[0] must equal indices.shape[0].
3. indices must be a 1-D tensor.
4. lr/l1/l2/lr\_power are scalar tensors (0-D).
5. **lr is the learning rate and cannot be 0 when passed in as an input**.
6. **indices must be unique** (duplicate values are not allowed). Both TensorFlow and Ascend officially state that the behavior is undefined when indices contain duplicates.
7. **The value range of indices must satisfy 0 <= indices[i] < var.shape[0]**. The kernel does not perform a runtime out-of-bounds check; going out of bounds will cause out-of-bounds memory reads and writes.
8. **The input tensors must have a contiguous memory layout**. The kernel uses flat addressing, which assumes contiguous memory; `AutoContiguous()` takes effect only on the aclnn path, and on the geir/tf path the caller must guarantee that the inputs are contiguous.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a>-</a></td>
    <td>Invokes the SparseApplyFtrl operator by constructing the graph through the [operator IR] <a href="op_graph/sparse_apply_ftrl_proto.h">(..op_graph/sparse_apply_ftrl_proto.h)</a>.</td>
  </tr>
</tbody>
</table>
