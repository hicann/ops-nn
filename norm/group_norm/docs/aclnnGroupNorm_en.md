# aclnnGroupNorm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:05:38.673Z -->

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

- Description: Computes the group normalization result out, mean meanOut, and reciprocal of standard deviation rstdOut for the input self.
- Formula:

  Let $x=self$, where $E[x] = \bar{x}$ denotes the mean of $x$, and $Var[x] = \frac{1}{n} * \sum_{i=1}^n(x_i - E[x])^2$ denotes the population variance of $x$. Then

  $$
  out = \frac{x - E[x]}{\sqrt{Var[x] + eps}} * \gamma + \beta
  $$

  $$
  meanOut = E[x]
  $$

  $$
  rstdOut = \frac{1}{\sqrt{Var[x] + eps}}
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the `aclnnGroupNormGetWorkspaceSize` API to obtain the workspace size required for the calculation and the executor that contains the operator computation process, and then call the `aclnnGroupNorm` API to execute the calculation.

```Cpp
aclnnStatus aclnnGroupNormGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *gamma,
  const aclTensor *beta,
  int64_t          N,
  int64_t          C,
  int64_t          HxW,
  int64_t          group,
  double           eps,
  aclTensor       *out,
  aclTensor       *meanOut,
  aclTensor       *rstdOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnGroupNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupNormGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the normalization calculation, corresponding to `x` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*)</td>
      <td>Input</td>
      <td>Scaling factor (weight) for the normalization calculation, corresponding to `γ` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Same data type as self.</li><li>Same number of elements as C.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*)</td>
      <td>Input</td>
      <td>Offset for the normalization calculation, corresponding to `β` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Same data type as self.</li><li>Same number of elements as C.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>N (int64_t)</td>
      <td>Input</td>
      <td>Represents the spatial size of input self dimension-wise in N.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>C (int64_t)</td>
      <td>Input</td>
      <td>Represents the dimension-wise spatial size of input self on the C dimension.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>HxW (int64_t)</td>
      <td>Input</td>
      <td>Indicates the spatial size of input self excluding the N and C dimensions.</td>
      <td><ul><li>1 if there are no other dimensions.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group (int64_t)</td>
      <td>Input</td>
      <td>Indicates that the C dimension of input self is divided into group groups.</td>
      <td><ul><li>group must be greater than 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>eps (double)</td>
      <td>Input</td>
      <td>The `eps` value in the formula for computing out and rstdOut, added to the denominator for numerical stability.</td>
      <td><ul><li>Must be greater than 0. For precision, eps must be greater than or equal to 1e-5.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the result of the normalization calculation, corresponding to `out` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>The data type and shape are the same as those of self.</li><li>The data format is the same as that of self.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>meanOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the mean after normalization, corresponding to `meanOut` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Data type is the same as self.</li><li>Shape is (N, group).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut (aclTensor*)</td>
      <td>Output</td>
      <td>Represents the reciprocal of the standard deviation after normalization, corresponding to `rstdOut` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type is the same as that of self.</li><li>The shape is (N, group).</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that the user needs to request on the Device.</td>
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

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:

    The data types of the `self`, `gamma`, `beta`, `out`, `meanOut`, and `rstdOut` parameters do not support BFLOAT16.
  <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>A required input, output, or required attribute is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="15">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="15">161002</td>
      <td>The data type of self, gamma, beta, out, meanOut, or rstdOut is not within the supported range.</td>
    </tr>
    <tr>
      <td>The data type of gamma, beta, out, meanOut, or rstdOut is different from that of self.</td>
    </tr>
    <tr>
      <td>The dimension of self is less than 2 or greater than 8.</td>
    </tr>
    <tr>
      <td>group is not greater than 0.</td>
    </tr>
    <tr>
      <td>C is not divisible by group.</td>
    </tr>
    <tr>
      <td>eps is not greater than 0.</td>
    </tr>
    <tr>
      <td>The shape of out is different from that of x.</td>
    </tr>
    <tr>
      <td>The shape of meanOut and rstdOut is not (N, group).</td>
    </tr>
    <tr>
      <td>The dimension-wise spatial size of self in N is not equal to N.</td>
    </tr>
    <tr>
      <td>The dimension-wise spatial size of self in C is not equal to C.</td>
    </tr>
    <tr>
      <td>The product of the sizes of the third and subsequent dimensions of self is not equal to HxW (HxW must be 1 when self is a two-dimensional tensor).</td>
    </tr>
    <tr>
      <td>The number of elements in self is not equal to N*C*HxW.</td>
    </tr>
    <tr>
      <td>gamma is not a 1-dimensional tensor and the number of its elements is not equal to C.</td>
    </tr>
    <tr>
      <td>beta is not a 1-dimensional tensor and the number of its elements is not equal to C.</td>
    </tr>
    <tr>
      <td>The formats of self and out are different.</td>
    </tr>
  </tbody></table>

## aclnnGroupNorm

- **Parameters**

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
      <td>Workspace memory address requested on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the Device, obtained by the first-phase API aclnnGroupNormGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnGroupNorm defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm.h"

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
    // Boilerplate for resource initialization.
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API documentation.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. You need to customize the construction based on the API.
    std::vector<int64_t> selfShape = {2, 3, 4};
    std::vector<int64_t> gammaShape = {3};
    std::vector<int64_t> betaShape = {3};
    std::vector<int64_t> outShape = {2, 3, 4};
    std::vector<int64_t> meanOutShape = {2, 1};
    std::vector<int64_t> rstdOutShape = {2, 1};
    void* selfDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanOutDeviceAddr = nullptr;
    void* rstdOutDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* out = nullptr;
    aclTensor* meanOut = nullptr;
    aclTensor* rstdOut = nullptr;
    std::vector<float> selfHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                       13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> gammaHostData = {2.0, 2, 2};
    std::vector<float> betaHostData = {2.0, 2, 2};
    std::vector<float> outHostData = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                                      13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    std::vector<float> meanOutHostData = {2.0, 2};
    std::vector<float> rstdOutHostData = {2.0, 2};
    int64_t N = 2;
    int64_t C = 3;
    int64_t HxW = 4;
    int64_t group = 1;
    double eps = 1e-5;
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create beta aclTensor.
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create meanOut aclTensor.
    ret = CreateAclTensor(meanOutHostData, meanOutShape, &meanOutDeviceAddr, aclDataType::ACL_FLOAT, &meanOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create rstdOut aclTensor.
    ret = CreateAclTensor(rstdOutHostData, rstdOutShape, &rstdOutDeviceAddr, aclDataType::ACL_FLOAT, &rstdOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, and replace it with the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API aclnnGroupNorm.
    ret = aclnnGroupNormGetWorkspaceSize(
        self, gamma, beta, N, C, HxW, group, eps, out, meanOut, rstdOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Request device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API aclnnGroupNorm.
    ret = aclnnGroupNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNorm failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output values, and copy the result from the device side to the host side. Modify the code based on the definition of the specific API.
    auto size = GetShapeSize(outShape);
    std::vector<float> outResultData(size, 0);
    ret = aclrtMemcpy(
        outResultData.data(), outResultData.size() * sizeof(outResultData[0]), outDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("outResultData[%ld] is: %f\n", i, outResultData[i]);
    }

    size = GetShapeSize(meanOutShape);
    std::vector<float> meanResultData(size, 0);
    ret = aclrtMemcpy(
        meanResultData.data(), meanResultData.size() * sizeof(meanResultData[0]), meanOutDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("meanResultData[%ld] is: %f\n", i, meanResultData[i]);
    }

    size = GetShapeSize(rstdOutShape);
    std::vector<float> rstdResultData(size, 0);
    ret = aclrtMemcpy(
        rstdResultData.data(), rstdResultData.size() * sizeof(rstdResultData[0]), rstdOutDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("rstdResultData[%ld] is: %f\n", i, rstdResultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify as required by the specific API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(out);
    aclDestroyTensor(meanOut);
    aclDestroyTensor(rstdOut);

    // 7. Release device resources. Modify as required by the specific API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(meanOutDeviceAddr);
    aclrtFree(rstdOutDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
