# InplaceApplyKerasMomentum

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:-----|:-------:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term>| x |
| <term>Atlas A2 products</term> | x |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Performs a single-step parameter update of the Keras Momentum optimizer. It updates the momentum accumulation accum based on the momentum coefficient momentum, the learning rate lr, and the gradient grad, and updates the weight parameter var in-place (inplace semantics) in standard mode or Nesterov mode. This is the dual-output version: both var and accum are explicitly output and written back in-place. It matches the computation semantics of the `tf.raw_ops.ResourceApplyKerasMomentum` interface in TensorFlow.

- Formula:

  $$
  accum_{new} = momentum \cdot accum - lr \cdot grad
  $$

  - Standard mode (use_nesterov = False):

  $$
  var_{new} = var + accum_{new}
  $$

  - Nesterov mode (use_nesterov = True):

  $$
  var_{new} = var + momentum \cdot accum_{new} - lr \cdot grad
  $$

  Where `momentum` is the momentum coefficient, `lr` is the learning rate, `grad` is the current gradient, `accum` is the momentum accumulation, and `var` is the weight parameter to be updated. State write-back (inplace semantics): `accum ← accum_new` and `var ← var_new`.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:-------|:--------------|:-----|:--------|:--------|
| var | Input | Weight parameter to be updated, corresponding to var in the formula. It comes from Variable. | FLOAT16, FLOAT, BFLOAT16 | ND |
| accum | Input | Momentum accumulation, corresponding to accum in the formula. Its shape/dtype must be the same as those of var. | FLOAT16, FLOAT, BFLOAT16 | ND |
| lr | Input | Learning rate, corresponding to lr in the formula. It is a scalar tensor. | FLOAT | ND |
| grad | Input | Current gradient tensor, corresponding to grad in the formula. Its shape/dtype must be the same as those of var. | FLOAT16, FLOAT, BFLOAT16 | ND |
| momentum | Input | Momentum coefficient, corresponding to momentum in the formula. It is a scalar tensor. | FLOAT | ND |
| var | Output | Updated var tensor, sharing Device memory with the input var (inplace). | FLOAT16, FLOAT, BFLOAT16 | ND |
| accum | Output | Updated accum tensor, sharing Device memory with the input accum (inplace). | FLOAT16, FLOAT, BFLOAT16 | ND |
| use_locking | Attribute | Whether to lock. NPU single-stream execution ignores it; the attribute is kept for TF compatibility. The default is False. | Bool | - |
| use_nesterov | Attribute | Whether to use Nesterov momentum. When it is True, the var update formula switches to Nesterov mode. The default is False. | Bool | - |

## Constraints

- var, accum, and grad must have the same data type and shape, with a rank of 8 or less; when the rank exceeds 8 dimensions, tiling returns a failure.
- var, accum, and grad support only the contiguous ND layout, and broadcasting (broadcast) is not supported.
- lr and momentum must be scalar tensors or tensors with a shape of `{1}`, with the data type FLOAT; the GE graph mode does not support rank=0. Use `{1}` for construction.
- An empty tensor (with 0 elements) is valid: the operator does not update the data, and the output remains the same as the input.
- The outputs var and accum have the same shape as the input var (inplace semantics).
- Graph construction is supported through the `-1`/`-2` dynamic shape/rank flags: `-1` indicates a dynamic dimension, and `-2` indicates a dynamic rank. During graph construction, the Data/output descs use the dynamic flags, and the input tensors of `RunGraph` must carry the actual shapes (for example, `[4,4]`). The `-1`/`-2` flags cannot be directly used as feed shapes. After the dynamic dimensions are expanded, the preceding shape, dtype, and rank constraints must still be satisfied.
- Capacity boundary: the total number of elements of var/accum/grad is capped at `4,294,967,295` (`UINT32_MAX`); when it is exceeded, tiling returns a failure. `4G elements = 2^32 = 4,294,967,296` has already exceeded the cap and should be rejected.
- This operator does not declare an additional `4GiB` logical byte cap: `4GiB` corresponds to `2^30` elements for FP32 and `2^31` elements for FP16/BF16, all within the element count cap. Whether execution is possible also depends on the physical capacity such as device HBM.
- Tiling internally limits `blockNum` to `65535`; extra-large shapes that require more blocks are handled by increasing `blockFormer` per block, and shapes are not rejected due to the block count.
- Numerical semantics: FLOAT16/BFLOAT16 inputs are uniformly promoted to FP32 for computation; the exponent range of BFLOAT16 is the same as that of FP32, and intermediate quantities (such as `lr*grad`) may overflow to ±Inf in FP32. This is the declared intermediate precision semantics of this operator, not the FP64 mathematical truth.
- The var/accum results of FLOAT16 are saturated to `[-65504, 65504]` before being written back with CAST_ROUND, and ±65504 is obtained upon overflow instead of ±Inf; BFLOAT16 and FLOAT do not perform this saturation. The numerical behavior of NaN/Inf inputs is currently not guaranteed, and such inputs are not within the verification scope.

## Rejection Conditions and Error Codes

When tiling detects invalid inputs, it returns `GRAPH_FAILED` and reports the following predefined error codes:

| Rejection condition | Error code |
|:---------|:-------|
| The rank of var/accum/grad is greater than 8 | `EZ0012` |
| The shapes of var/accum/grad are inconsistent | `EZ0010` |
| lr/momentum is neither a scalar nor a tensor with a shape of `{1}` | `EZ0010` |
| The storage format of var is not ND | `EZ0017` |
| The dtype of var is not FLOAT16/FLOAT/BFLOAT16, or lr/momentum is not FLOAT | `EZ0019` |
| The total number of elements of var is less than 0 or greater than `4294967295` | `EZ0026` |

Supplementary notes:

- An empty tensor (with 0 elements) does not enter the rejection process; the operation returns success and does not update data.
- Non-contiguous ND inputs are not within the guarantee scope of this operator, and there is no corresponding runtime rejection check currently. The caller must ensure that the inputs are in the contiguous ND layout.

## Invocation Description

| Invocation method | Invocation sample | Description |
|:--------|:--------|:-----|
| GE graph mode | [test_geir_inplace_apply_keras_momentum](examples/arch35/test_geir_inplace_apply_keras_momentum.cpp) | Invokes the InplaceApplyKerasMomentum operator through the operator IR-based [graph construction](op_graph/inplace_apply_keras_momentum_proto.h) method. |
