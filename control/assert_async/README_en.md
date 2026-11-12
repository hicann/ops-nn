# AssertAsync

<!-- md-trans-meta sourceCommit=1c655dcd79f7aa4a4a10d66c09810576779a0c65 translatedAt=2026-11-03T12:54:39.448Z -->

English | [简体中文](./README.md)

## Applicable Products

| Product                                                      | Supported or Not |
| :----------------------------------------------------------- | :--------------: |
| <term>Ascend 950PR&950DT products</term>                     |        √         |
| <term>Atlas A3 products</term>                               |        ×         |
| <term>Atlas A2 products</term>                               |        ×         |
| <term>Atlas 200I/500 A2 inference products</term>            |        ×         |
| <term>Atlas inference products</term>                        |        ×         |
| <term>Atlas training products</term>                         |        ×         |

## Function

- Operator function: Asserts that the value of the input tensor `input_condition` (requires `numel == 1`) is non-zero. If the value is 0 (including bool `False`),
  the device prints the message specified by the attribute `assert_msg` and the operator execution fails; if the value is non-zero, the assertion passes directly. This operator has **no output tensor**,
  and its semantics align with PyTorch's `torch._assert_async` (including the `.msg` overload semantics).

- Decision criteria (bit-exact, no numerical computation):
  - Floating-point types (FLOAT16, FLOAT, DOUBLE, BFLOAT16) are determined by sign-bit masking: FLOAT uses `& 0x7FFFFFFF`, FLOAT16/BFLOAT16 use `& 0x7FFF`,
    and DOUBLE uses `& 0x7FFFFFFFFFFFFFFF`. Both `+0.0` and `-0.0` fail the assertion; `NaN`, `+inf`, and `-inf` pass.
  - Complex types (COMPLEX64, COMPLEX128): the assertion fails when the real and imaginary parts are both exactly 0 (including `-0.0` components); `NaN`/`±inf` components pass.
  - Integer and Boolean types are determined by all-zero bytes: `0` (including bool `False`) fails the assertion, and non-zero values (including `True`) pass.

- Failure path: In the kernel, the message is printed via `AscendC::printf`, and then `AscendC::Trap()` is called to interrupt the AI Core, causing the operator execution to fail;
  the failure status is propagated to the upper layer via stream synchronization (for example, `aclrtSynchronizeStream`). When the assertion passes, no message is printed and no output is produced.

## Parameters

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
|--------|--------------|------|---------|---------|
| input_condition | Input | Condition tensor to be asserted, which requires `numel == 1` (rank-0 or any shape that shrinks to a single element). The number of dimensions must not exceed 8; otherwise, ACLNN_ERR_PARAM_INVALID (161002) is returned. A value of 0 (including bool `False`) fails the assertion. | BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128 | ND |
| assert_msg | Optional attribute | Message string printed when the assertion fails, defaulting to `""`. The available length is 254 bytes (the length must be less than 255; the buffer is 256 bytes, including the trailing `'\0'`); when the limit is exceeded, ACLNN_ERR_PARAM_INVALID (161002) is returned. | String | - |

> This operator has no output tensor.

## Constraints

- This operator has no output tensor and involves no floating-point computation; the decision is bit-exact by behavior, so the floating-point tolerance (atol/rtol) precision standard for output tensors does not apply.
- Assertion failure occurs inside the device-side kernel; the operator failure must be observed through stream synchronization.
- This operator supports only the direct aclnn API invocation path and does not support dynamic graph (dynamic shape) scenarios.

## Invocation Description

| Invocation Mode | Invocation Example | Description |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_assert_async](./examples/arch35/test_aclnn_assert_async.cpp) | Invokes the AssertAsync operator through the [aclnnAssertAsync](./docs/aclnnAssertAsync_en.md) API. |
