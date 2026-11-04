# ApplyAdadelta

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     √    |
|  <term>Atlas training products</term>    |     √    |

## Function

- **Description**: Performs a single-step parameter update of the Adadelta optimizer. Based on the current gradient, the gradient square accumulation, and the update square accumulation, it computes the parameter update amount and updates the weight parameters **in place** (inplace semantics). It aligns with the computation semantics of the TensorFlow `tf.raw_ops.ApplyAdadelta` interface and PyTorch `torch.optim.Adadelta`.

- **Formula**:

  $$
  \begin{aligned}
  accum_{t} &= \rho \cdot accum_{t-1} + (1 - \rho) \cdot grad^2 \\
  update &= \frac{\sqrt{accum\_update_{t-1} + \epsilon}}{\sqrt{accum_{t} + \epsilon}} \cdot grad \\
  var_{t} &= var_{t-1} - lr \cdot update \\
  accum\_update_{t} &= \rho \cdot accum\_update_{t-1} + (1 - \rho) \cdot update^2
  \end{aligned}
  $$

  Here, `rho` is the decay coefficient (value range [0, 1)), `epsilon` is the numerical stability constant (must be > 0), and `lr` is the learning rate.

## Parameters

<table style="table-layout: fixed; width: 1500px"><colgroup>
<col style="width: 170px">
<col style="width: 170px">
<col style="width: 300px">
<col style="width: 200px">
<col style="width: 170px">
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
    <td>Input / Output (inplace)</td>
    <td>Weight parameter to be updated, corresponding to var in the formula. The update is performed inplace inside the Kernel, and the single-output view of the GE IR shares Device memory with the input var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum</td>
    <td>Input (inplace update)</td>
    <td>Gradient square accumulation, corresponding to accum in the formula. The shape/dtype must be consistent with those of var; the Kernel explicitly writes back to the input GM address.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>accum_update</td>
    <td>Input (inplace update)</td>
    <td>Update square accumulation, corresponding to accum_update in the formula. The shape/dtype must be consistent with those of var; the Kernel explicitly writes back to the input GM address.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>lr</td>
    <td>Input</td>
    <td>Learning rate, corresponding to lr in the formula. It is a 1-element scalar Tensor with shape={1}, and the dtype must be consistent with that of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rho</td>
    <td>Input</td>
    <td>Decay coefficient, corresponding to rho in the formula. It is a 1-element scalar Tensor with shape={1}, and the value range is [0, 1).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>epsilon</td>
    <td>Input</td>
    <td>Numerical stability constant, corresponding to epsilon in the formula. It is a 1-element scalar Tensor with shape={1}, and the value must be > 0.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>grad</td>
    <td>Input</td>
    <td>Current gradient Tensor, corresponding to grad in the formula. The shape/dtype must be consistent with those of var.</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>use_locking</td>
    <td>Attribute</td>
    <td>Whether to lock during the update. Default: false. The current implementation does not enforce a mutex lock; it serves only as a semantic placeholder.</td>
    <td>BOOL</td>
    <td>-</td>
  </tr>
  <tr>
    <td>var (output)</td>
    <td>Output</td>
    <td>The updated var Tensor, sharing Device memory with the input var (inplace).</td>
    <td>FLOAT16, FLOAT</td>
    <td>ND</td>
  </tr>
</tbody></table>

## Constraints

None.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph mode | [test_geir_apply_adadelta](./examples/test_geir_apply_adadelta.cpp) | Invokes the ApplyAdadelta operator by building a graph through the [operator IR](./op_graph/apply_adadelta_proto.h). |
