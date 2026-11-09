# SparseApplyFtrl

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    √     |
| <term>Atlas training products</term>                               |    √     |

> Notes:
> - Ascend 950PR/950DT is provided by the Ascend C implementation in this repository (OpDef registers ascend950/ascend350; the build covers ascend950/ascend350). Atlas A3/A2 and Atlas inference/training products are provided by the canndev TBE implementation. Atlas 200I/500 A2 has no corresponding binary configuration and is not supported.
> - The current implementations use FLOAT for var/accum/linear/grad/lr/l1/l2/lr_power, INT32/INT64 for indices, and the ND data format.

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
      <td>Optional attribute. Default value: false. Specifies whether to use a lock to protect the update operation. The current implementation does not use this attribute and its value does not change the update behavior.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

1. var/accum/linear must have the same shape and dtype (FLOAT).
2. grad must have the same rank as var; grad.shape[0] must equal indices.shape[0], and grad.shape[1:] must equal var.shape[1:].
3. indices must be a 1-D tensor with dtype INT32 or INT64.
4. lr/l1/l2/lr\_power are scalar tensors in the form of a 0-D tensor or a 1-D tensor with shape [1]. Dynamic shapes (-1/-2) are handled as dynamic in InferShape; higher-rank forms such as `(1,1)` are rejected.
5. **The Ascend C implementation in this repository requires var/accum/linear to have at least 2 dimensions; the rank upper bound is not explicitly validated by this operator and is subject to framework/interface limits**. Implementations backed by canndev TBE (for example, Atlas A3/A2) accept a lower rank bound (1-D var is allowed). When using multiple products, prepare inputs according to the constraints of the corresponding implementation version.
6. **lr is the learning rate; the caller must ensure lr > 0**. l1 and l2 should be greater than or equal to 0, and lr\_power should be less than or equal to 0. Host/Tiling does not read device-side scalar values, so no runtime value-range validation is performed; invalid values lead to undefined results.
7. **indices must be unique** (duplicate values are not allowed). Both TensorFlow and Ascend officially state that the behavior is undefined when indices contain duplicates; the current implementation does not perform a runtime duplicate check.
8. **The value range of indices must satisfy 0 <= indices[i] < var.shape[0]**. The kernel does not perform a runtime out-of-bounds check; going out of bounds will cause out-of-bounds memory reads and writes.
9. **The input tensors must have a contiguous memory layout**. The kernel uses flat addressing, which assumes contiguous memory; `AutoContiguous()` takes effect only on the aclnn path, and on the geir/tf path the caller must guarantee that the inputs are contiguous.
10. **The outputs var/accum/linear are independent output buffers and do not share storage in-place with the inputs**. The caller should use the output tensors to obtain the updated results.
11. **Only the ND data format is supported**. The current binary_config provides only FLOAT + INT32/INT64 with ND. Positional formats such as NCHW/NHWC from historical versions are not supported in this version.

## Invocation Description

| Invocation method | Sample code | Description |
|:---|:---|:---|
| GE graph mode | [test_geir_sparse_apply_ftrl.cpp](./examples/test_geir_sparse_apply_ftrl.cpp) | Invokes the SparseApplyFtrl operator by constructing a graph with the [SparseApplyFtrl operator IR](./op_graph/sparse_apply_ftrl_proto.h). |

> The TensorFlow path currently registers only `SparseApplyFtrl`. `ResourceSparseApplyFtrl` is not registered in the plugin in this repository and is provided by the canndev implementation. For sample compilation and execution, see the CANN sample compilation and execution guide.
