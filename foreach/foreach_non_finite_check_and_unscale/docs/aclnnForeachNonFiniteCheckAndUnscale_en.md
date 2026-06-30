# aclnnForeachNonFiniteCheckAndUnscale

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/foreach/foreach_non_finite_check_and_unscale)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Iterates over all tensors in **scaledGrads** to check for Inf or NaN. Sets **foundInf** to **1.0** if any Inf or NaN is detected; otherwise, **foundInf** remains unchanged. Then unscales all tensors in **scaledGrads** by multiplying with **invScale**.

- Formula:

  $$
  foundInf = \begin{cases}1.0, & if any Inf or NaN exists in scaledGrads,\\
    foundInf, & otherwise.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnForeachNonFiniteCheckAndUnscale** is called to perform computation.

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(
  const aclTensorList *scaledGrads,
  const aclTensor     *foundInf,
  const aclTensor     *invScale,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

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
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>scaledGrads</td>
      <td>Input/Output</td>
      <td>Input and output tensor list for the unscaling computation, corresponding to `scaledGrads` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>All tensors within this parameter must share the same data type. </li><li>The tensor list contains a maximum of 256 tensors.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>    
    <tr>
      <td>foundInf</td>
      <td>Input/Output</td>
      <td>Tensor used to indicate whether Inf or NaN exists in **scaledGrads**, corresponding to `foundInf` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The tensor contains only one element.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invScale</td>
      <td>Input</td>
      <td>Tensor for the unscaling computation, corresponding to `invScale` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The tensor contains only one element.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0–8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace required to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed scaledGrads, foundInf, or invScale is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The data type of scaledGrads, foundInf, or invScale is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>The length of scaledGrads exceeds the limit.</td>
    </tr>
    <tr>
      <td>The data types of tensors in scaledGrads do not match.</td>
    </tr>
    <tr>
      <td>foundInf or invScale does not contain exactly one element.</td>
    </tr>
  </tbody></table>

## aclnnForeachNonFiniteCheckAndUnscale

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnForeachNonFiniteCheckAndUnscale** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_non_finite_check_and_unscale.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Boilerplate) Initialize resources.
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
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> foundInfShape = {1};
  std::vector<int64_t> invScaleShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* foundInfDeviceAddr = nullptr;
  void* invScaleDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* foundInf = nullptr;
  aclTensor* invScale = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> foundInfHostData = {-1.0f};
  std::vector<float> invScaleHostData = {3.1f};
  // Create an input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a foundInf aclTensor.
  ret = CreateAclTensor(foundInfHostData, foundInfShape, &foundInfDeviceAddr, aclDataType::ACL_FLOAT, &foundInf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an invScale aclTensor.
  ret = CreateAclTensor(invScaleHostData, invScaleShape, &invScaleDeviceAddr, aclDataType::ACL_FLOAT, &invScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<aclTensor*> tempInput{input1, input2};
  aclTensorList* tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachNonFiniteCheckAndUnscale.
  ret = aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(tensorListInput, foundInf, invScale, &workspaceSize,
                                                             &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachNonFiniteCheckAndUnscale.
  ret = aclnnForeachNonFiniteCheckAndUnscale(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscale failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(selfShape1);
  std::vector<float> self1Data(size, 0);
  ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                    size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self1 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self1 result[%ld] is: %f\n", i, self1Data[i]);
  }

  size = GetShapeSize(selfShape2);
  std::vector<float> self2Data(size, 0);
  ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                    size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self2 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self2 result[%ld] is: %f\n", i, self2Data[i]);
  }

  size = GetShapeSize(foundInfShape);
  std::vector<float> foundInfData(size, 0);
  ret = aclrtMemcpy(foundInfData.data(), foundInfData.size() * sizeof(foundInfData[0]), foundInfDeviceAddr,
                    size * sizeof(foundInfData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy foundInf result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("foundInf result is: %f\n", foundInfData[0]);

  // 6. Destroy aclTensorList and aclTensor. Modify the code based on the API definition.
  aclDestroyTensorList(tensorListInput);
  aclDestroyTensor(foundInf);
  aclDestroyTensor(invScale);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(foundInfDeviceAddr);
  aclrtFree(invScaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
