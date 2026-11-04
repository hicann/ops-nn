# InplaceApplyAdadelta

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: Performs a single-step parameter update of the Adadelta optimizer. Computes the parameter update amount based on the current gradient, the gradient square accumulation state accum, and the update amount square accumulation state accum_update. The three outputs var, accum, and accum_update share device memory (inplace) with the inputs of the same names respectively, implementing in-place updates.

- Formula:

  $$
      \begin{aligned}
      accum_{t} &= \rho \cdot accum_{t-1} + (1 - \rho) \cdot grad^2 \\
      update &= \frac{\sqrt{accum\_update_{t-1} + \epsilon}}{\sqrt{accum_{t} + \epsilon}} \cdot grad \\
      var_{t} &= var_{t-1} - lr \cdot update \\
      accum\_update_{t} &= \rho \cdot accum\_update_{t-1} + (1 - \rho) \cdot update^2
      \end{aligned}
  $$

  Here, `rho` is the decay coefficient (the value directly participates in the computation; the typical value is 0.9), `epsilon` is the numerical stability constant (usually a positive value), and `lr` is the learning rate.

## Parameters

<table style="table-layout: fixed; width: 1430px">
  <colgroup>
    <col style="width: 127px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>var</td>
      <td>Input</td>
      <td>Weight parameter, which corresponds to var in the formula.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum</td>
      <td>Input</td>
      <td>Gradient square accumulation state, which corresponds to accum in the formula. The shape/dtype must be the same as those of var.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update</td>
      <td>Input</td>
      <td>Update amount square accumulation state, which corresponds to accum_update in the formula. The shape/dtype must be the same as those of var.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>lr</td>
      <td>Input</td>
      <td>Learning rate, which corresponds to lr in the formula. Rank 0 scalar tensor (shape={}), and the dtype must be the same as that of var.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>rho</td>
      <td>Input</td>
      <td>Decay coefficient, which corresponds to rho in the formula. Rank 0 scalar tensor (shape={}), and the value directly participates in the computation.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Numerical stability constant, which corresponds to epsilon in the formula. Rank 0 scalar tensor (shape={}); usually a positive value, and a non-positive value is computed according to the formula.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad</td>
      <td>Input</td>
      <td>Current gradient tensor, which corresponds to grad in the formula. The shape/dtype must be the same as those of var.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>use_locking</td>
      <td>Attribute</td>
      <td>Kept only for API compatibility, to stay compatible with the implementation of the legacy A2 operator ApplyAdadeltaD. Default value: false. It does not participate in the computation flow or concurrency lock control.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>var(output)</td>
      <td>Output</td>
      <td>Updated weight parameter, which shares device memory (inplace) with the input var.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum(output)</td>
      <td>Output</td>
      <td>Updated gradient square accumulation state, which shares device memory (inplace) with the input accum.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>accum_update (output)</td>
      <td>Output</td>
      <td>Updated update amount square accumulation state, which shares device memory (inplace) with the input accum_update.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody>
</table>

## Constraints

- The shapes of var, accum, accum_update, and grad must be completely identical (ND format).
- The outputs var, accum, and accum_update share device memory (inplace) with the inputs of the same names respectively.
- The GE graph mode supports graph construction with static shapes, unknown dimensions (`-1`), and unknown ranks (`-2`). At runtime, the specific shapes of the four main tensors must be identical, and the three outputs use the specific shapes of the corresponding inputs.
- lr/rho/epsilon must be rank 0 scalar tensors (shape `{}`, physically stored as 1 element), and the dtype must be the same as that of var.
- Only FLOAT/FLOAT16 is supported. FP16 intermediate results are saved as half. The multiply-add operations in the accum and var updates use Fma and are rounded to half after the addition. Square root and reciprocal square root operations use FP32 for temporary storage, and the results are converted back to half.
- The FP32 path uses DAZ/FTZ numerical semantics: finite subnormal numbers in var, accum, accum_update, grad, lr, rho, and epsilon are converted to zeros with the same sign before participating in the computation. Finite subnormal results produced by each FP32 arithmetic step are likewise converted to zeros with the same sign before entering the next step. The judgment condition is `|x| < 1.17549435e-38`, and the smallest normal number that exactly equals this threshold is retained. NaN and Inf are not rewritten by the DAZ/FTZ processing, and subsequent results propagate according to the specific floating-point operations.
- Valid empty tensors with the same shape are supported. When the shapes of var, accum, accum_update, and grad are completely identical and contain zero-length dimensions (for example, `[0]` or `[0, 3]`), the operator returns successfully. The three outputs var, accum, and accum_update keep the shape, dtype, and inplace relationship of the corresponding inputs, and no element-wise computation is performed. lr, rho, and epsilon must still be rank 0 scalar tensors.

## Invocation Description

| Invocation method   | Sample code                                                | Description                            |
| :--------- | :------------------------------------------------------ | :------------------------------ |
| GE graph mode | [test_geir_inplace_apply_adadelta](./examples/test_geir_inplace_apply_adadelta.cpp) | Calls the InplaceApplyAdadelta operator through graph construction with the [operator IR](./op_graph/inplace_apply_adadelta_proto.h). This sample demonstrates a static shape. |
