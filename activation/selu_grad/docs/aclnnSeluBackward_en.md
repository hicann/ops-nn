# aclnnSeluBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T13:28:31.630Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Performs backward propagation of [aclnnSelu](../../selu/docs/aclnnSelu&aclnnInplaceSelu.md).

- Formula:

  Computes the derivative of the activation function:

  $$
  \frac{\partial selu(x)}{\partial x}=\begin{cases} scale \cdot\alpha e^x,x<0 \\scale,x\geq 0\end{cases}
  $$

  Computes the derivative of the error with respect to the input:

  $$
  \frac{\partial E}{\partial x}=\frac{\partial E}{\partial y}\frac{\partial selu(x)}{\partial x}
  $$

  where $y$ is the output and $E$ is the loss function.
  $\alpha$=1.6732632423543772848170429916717

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnSeluBackwardGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the operator computation process, and then call the "aclnnSeluBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnSeluBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* result,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSeluBackward(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSeluBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 330px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
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
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient of the Selu computation output, which is ∂E/∂y in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape must be consistent with those of result and gradInput.</li></ul></td>
      <td>FLOAT, FLOAT16, INT32, INT8, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>result (aclTensor*)</td>
      <td>Input</td>
      <td>Forward output of the Selu computation, which is y in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape must be the same as those of gradOutput and gradInput.</li></ul></td>
      <td>FLOAT, FLOAT16, INT32, INT8, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>gradInput (aclTensor*)</td>
      <td>Output</td>
      <td>Gradient of the input in the Selu computation, that is, ∂E/∂X in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type and shape must be the same as those of gradOutput and result.</li></ul></td>
      <td>FLOAT, FLOAT16, INT32, INT8, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation process of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

   <!-- npu="910,310p" id7 -->
   - <term>Atlas inference products</term> and <term>Atlas training products</term>: FLOAT, FLOAT16, INT32, and INT8 data types are supported.
   <!-- end id7 -->

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>gradOutput, result, gradInput, workspaceSize, or executor is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of gradOutput, result, and gradInput are not within the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of gradOutput, result, and gradInput are inconsistent.</td>
    </tr>
    <tr>
      <td>The data types of gradOutput, result, and gradInput do not comply with the data type deduction rules.</td>
    </tr>
    <tr>
      <td>The ranks of gradOutput, result, or gradInput are outside 1-8. A rank-0 scalar is not an empty tensor.</td>
    </tr>
  </tbody></table>

## aclnnSeluBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>Pointer to the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnSeluBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnSeluBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnn_selu_backward.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}

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
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> shape = {4, 2};
    void* gradDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* gradients = nullptr;
    aclTensor* result = nullptr;
    aclTensor* y = nullptr;

    // SELU constant
    const float SCALE = 1.0507009873554804f;
    const float ALPHA = 1.6732632423543772f;
    const float SCALE_ALPHA_PRODUCT = SCALE * ALPHA;

    // Construct inputs: gradients = all 1s, result = [-2, -1, 0, 1, 2, 3, -0.5, 0.5].
    std::vector<float> gradHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> outHostData = {-2, -1, 0, 1, 2, 3, -0.5, 0.5};
    std::vector<float> yHostData(8, 0);

    ret = CreateAclTensor(gradHostData, shape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &gradients);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outHostData, shape, &outDeviceAddr, aclDataType::ACL_FLOAT, &result);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, shape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnSeluBackwardGetWorkspaceSize(gradients, result, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnSeluBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSeluBackward failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(float), yDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\n=== SeluGrad Results ===\n");
    LOG_PRINT("result >= 0: y = SCALE * gradients = %.6f * grad\n", SCALE);
    LOG_PRINT("result <  0: y = grad * (result + SCALE_ALPHA) = grad * (out + %.6f)\n\n", SCALE_ALPHA_PRODUCT);

    int pass = 0, fail = 0;
    for (int64_t i = 0; i < size; i++) {
        float expected;
        if (outHostData[i] < 0) {
            expected = gradHostData[i] * (outHostData[i] + SCALE_ALPHA_PRODUCT);
        } else {
            expected = SCALE * gradHostData[i];
        }
        bool ok = std::fabs(resultData[i] - expected) < 0.01f;
        if (ok)
            pass++;
        else
            fail++;
        LOG_PRINT("  [%ld] out=%.2f grad=%.2f => NPU=%.6f  expected=%.6f  %s\n", i, outHostData[i], gradHostData[i],
                  resultData[i], expected, ok ? "PASS" : "FAIL");
    }
    LOG_PRINT("\nTotal: %d PASS, %d FAIL\n", pass, fail);

    aclDestroyTensor(gradients);
    aclDestroyTensor(result);
    aclDestroyTensor(y);
    aclrtFree(gradDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(yDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return fail > 0 ? 1 : 0;
}
```
