# aclnnAssertAsync

<!-- md-trans-meta sourceCommit=1c655dcd79f7aa4a4a10d66c09810576779a0c65 translatedAt=2026-11-03T12:54:39.448Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- API function: Asserts that the value of the input tensor `inputCondition` (requires `numel == 1`) is non-zero. If the value is 0 (including bool `False`),
  the device prints the message specified by `assertMsg` and the operator execution fails; if the value is non-zero, the assertion passes directly. This API has **no output tensor**.
  Its semantics align with PyTorch's `torch._assert_async` (including the `.msg` overload semantics), and it can be used to asynchronously validate a condition value on the device.

- Decision rules (bit-exact, no floating-point computation):

  ```text
  The single element value of inputCondition is 0 (including bool False)      ->  Assertion fails: assertMsg is printed + operator failure is triggered
  The single element value of inputCondition is non-zero (including bool True) ->  Assertion passes: no output / no side effects
  ```

  - Floating-point types (FLOAT16, FLOAT, DOUBLE, BFLOAT16) are determined by sign-bit masking: FLOAT uses `& 0x7FFFFFFF`, FLOAT16/BFLOAT16 use `& 0x7FFF`,
    and DOUBLE uses `& 0x7FFFFFFFFFFFFFFF`; both `+0.0` and `-0.0` fail the assertion, while `NaN`, `+inf`, and `-inf` pass.
  - Complex types (COMPLEX64, COMPLEX128): the assertion fails when the real and imaginary parts are both exactly 0 (including `-0.0` components); `NaN`/`±inf` components pass.
  - Integer and Boolean types are determined by all-zero bytes: `0` (including bool `False`) fails the assertion, and non-zero values (including `True`) pass.

- Failure path: In the kernel, `"<assertMsg>\n"` is printed via `AscendC::printf`, and then `AscendC::Trap()` is called
  to interrupt the AI Core, causing the operator execution to fail; the failure status is propagated to the upper layer via stream synchronization.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAssertAsyncGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor containing the operator computation flow, and then call the "aclnnAssertAsync" API to execute the computation.

```Cpp
aclnnStatus aclnnAssertAsyncGetWorkspaceSize(
  const aclTensor* inputCondition,
  const char*      assertMsg,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnAssertAsync(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnAssertAsyncGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputCondition (aclTensor*)</td>
      <td>Input</td>
      <td>Condition tensor to be asserted, corresponding to the operator input input_condition. A value of 0 (including bool False) fails the assertion.</td>
      <td><ul><li>Empty tensors are not supported (numel must be 1).</li><li>numel must be 1 (rank-0 or any shape that shrinks to a single element); otherwise, ACLNN_ERR_PARAM_INVALID (161002) is returned.</li><li>The number of dimensions must not exceed 8; otherwise, ACLNN_ERR_PARAM_INVALID (161002) is returned.</li><li>The data type must be within the supported range; otherwise, ACLNN_ERR_PARAM_INVALID (161002) is returned.</li></ul></td>
      <td>BOOL, INT8, INT16, INT32, INT64, UINT8, FLOAT16, FLOAT, DOUBLE, BFLOAT16, COMPLEX64, COMPLEX128</td>
      <td>ND</td>
      <td>0-8 dimensions (numel == 1)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>assertMsg (char*)</td>
      <td>Input</td>
      <td>Message string printed when the assertion fails.</td>
      <td><ul><li>Can be nullptr, in which case it is treated as an empty string "".</li><li>It is a C string and must end with '\0'.</li><li>The available length limit is 254 bytes (the length must be less than 255; the buffer is 256 bytes, including the trailing '\0'); if the limit is exceeded, ACLNN_ERR_PARAM_INVALID (161002) is returned and the message is not truncated.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-stage API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>workspaceSize or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>inputCondition is a null pointer.</td>
    </tr>
    <tr>
      <td>The number of dimensions of inputCondition exceeds 8.</td>
    </tr>
    <tr>
      <td>The numel of inputCondition is not 1 (including empty tensors).</td>
    </tr>
    <tr>
      <td>The data type of inputCondition is not within the supported range.</td>
    </tr>
    <tr>
      <td>The length of assertMsg exceeds the upper limit (the length must be less than 255 bytes, and the maximum available length is 254 bytes).</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
      <td>561101</td>
      <td>Failed to create aclOpExecutor inside the API.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>The contiguous conversion inside the API (l0op::Contiguous) returned a null pointer, or the internal L0 launcher failed to add the operator (ADD_TO_LAUNCHER_LIST_AICORE).</td>
    </tr>
  </tbody></table>

## aclnnAssertAsync

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device. The current implementation does not require a workspace, so nullptr can be passed in.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-stage API aclnnAssertAsyncGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  If the value of `inputCondition` is 0 (including bool `False`), the assertion fails: `assertMsg` is printed on the device and the operator execution fails.
  The failure return must be observed through stream synchronization (for example, `aclrtSynchronizeStream`).

## Constraints

- Deterministic computation: aclnnAssertAsync defaults to a deterministic implementation.
- This API has no output tensor and involves no floating-point computation; the decision is bit-exact by behavior, so the floating-point tolerance (atol/rtol) precision standard for output tensors does not apply.
- Assertion failure occurs inside the device-side kernel; the operator failure must be observed through stream synchronization.
- This operator supports only the direct aclnn API invocation path and does not support dynamic graph (dynamic shape) scenarios.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

<term>Ascend 950PR&950DT products</term>:

```Cpp
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_assert_async.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<int64_t>(hostData.size()) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

static int RunAssertAsyncCase(bool value, const char* assertMsg)
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    void* conditionDeviceAddr = nullptr;
    aclTensor* condition = nullptr;
    ret = CreateAclTensor<uint8_t>({static_cast<uint8_t>(value ? 1 : 0)}, {1}, &conditionDeviceAddr, ACL_BOOL,
                                   &condition);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor failed. ERROR: %d\n", ret); return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnAssertAsyncGetWorkspaceSize(condition, assertMsg, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAssertAsyncGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > static_cast<uint64_t>(0)) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnAssertAsync(workspaceAddr, workspaceSize, executor, stream);
    LOG_PRINT("aclnnAssertAsync ret = %d (case value=%d)\n", ret, static_cast<int>(value));

    ret = aclrtSynchronizeStream(stream);
    LOG_PRINT("aclrtSynchronizeStream ret = %d (case value=%d)\n", ret, static_cast<int>(value));

    aclDestroyTensor(condition);
    aclrtFree(conditionDeviceAddr);
    if (workspaceSize > static_cast<uint64_t>(0)) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}

int main()
{
    LOG_PRINT("Testing aclnnAssertAsync with condition == true (expect pass).\n");
    RunAssertAsyncCase(true, "condition should be true");

    LOG_PRINT("Testing aclnnAssertAsync with condition == false (expect failure + printed message).\n");
    RunAssertAsyncCase(false, "condition is false");

    return 0;
}
```
