# aclnnForeachSubList

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-21T02:52:59.516Z -->

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
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Performs element-wise subtraction operation on two input tensor lists, with the subtraction coefficient adjustable via the alpha parameter.

- Formula:

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}]\\
  x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = x1_i-{x2_i}*alpha (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnForeachSubListGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the computation flow of the operator, and then call the "aclnnForeachSubList" API to execute the computation.

```Cpp
aclnnStatus aclnnForeachSubListGetWorkspaceSize(
  const aclTensorList     *x1,
  const aclTensorList     *x2,
  const aclTensor         *alpha,
  aclTensorList           *out,
  uint64_t                *workspaceSize,
  aclOpExecutor          **executor)
```

```Cpp
aclnnStatus aclnnForeachSubList(
  void               *workspace,
  uint64_t           workspaceSize,
  aclOpExecutor      *executor,
  aclrtStream         stream)
```

## aclnnForeachSubListGetWorkspaceSize

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
      <td>x1 (aclTensorList*)</td>
      <td>Input</td>
      <td>Tensor list of the minuend in the subtraction operation, corresponding to `x1` in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Data types of all Tensors in this parameter must be the same.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensorList*)</td>
      <td>Input</td>
      <td>Tensor list of the subtrahend in the subtraction operation, corresponding to `x2` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The data type, data format, and shape are the same as those of input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha (aclTensor*)</td>
      <td>Input</td>
      <td>Specifies the coefficient of the subtrahend in the subtraction operation, corresponding to `alpha` in the formula.</td>
      <td><ul><li>Does not support empty tensors.</li><li>Contains one element.</li><li>The data type has a specific correspondence with the data type of input `x1`:<ul><li>When the data type of `x1` is FLOAT32, FLOAT16, or INT32, the data type is the same as that of `x1`.</li><li>When the data type of `x1` is INT16, INT8, or UINT8, the data type supports INT32.</li><li>When the data type of `x1` is BFLOAT16, the data type supports FLOAT32.</li></ul></li></ul></td>
      <td>FLOAT32, FLOAT16, INT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out (aclTensorList*)</td>
      <td>Output</td>
      <td>Output tensor list of the subtraction operation, corresponding to `y` in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>All tensors in this parameter have the same data type.</li><li>The data type, data format, and shape of each tensor must be identical to those of the corresponding tensor in `x1`, and the tensor count must be the same as `x1` and `x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16, INT32, INT16, INT8, UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied for on the Device.</td>
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
      <td>The input x1, x2, alpha, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of x1, x2, alpha, or out is not in the supported range, or the dtype constraints within or between lists are violated.</td>
    </tr>
    <tr>
      <td>The shapes or tensor counts of x1, x2, and out do not meet the constraints.</td>
    </tr>
    <tr>
      <td>The dimensions of tensors in x1, x2, or out exceed 8.</td>
    </tr>
    <tr>
      <td>The number of elements in alpha is not 1 or its dimension exceeds 8.</td>
    </tr>
  </tbody>
  </table>

## aclnnForeachSubList

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
      <td>Pointer to the workspace memory allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace memory allocated on the Device, obtained by calling the first-phase API aclnnForeachSubListGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- The tensor counts of `x1`, `x2`, and `out` must be identical.
<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: The count of tensors in any single tensor list of `x1`, `x2`, and `out` shall not exceed 256.
<!-- end id7 -->
- When `x1`, `x2`, and `out` are all empty tensor lists, the API returns success without performing computation. Other tensor count mismatches return a parameter error.
- The `-1` (dynamic dimension) and `-2` (dynamic rank) markers are handled by Host inference according to the framework conventions. Tiling requires concrete shapes and non-negative element counts.
- Integer overflow wraps according to the bit width of the output data type.
- Deterministic computation:
  - aclnnForeachSubList defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_sub_list.h"

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
  // Boilerplate, resource initialization
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
  // Call aclrtMalloc to allocate device-side memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host-side data to device-side memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call the aclCreateTensor API to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (boilerplate) Device/stream initialization. For details, see the acl API Reference.
  // Fill in deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs. Custom construction is required based on the API.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> otherShape1 = {2, 3};
  std::vector<int64_t> otherShape2 = {1, 3};
  std::vector<int64_t> outShape1 = {2, 3};
  std::vector<int64_t> outShape2 = {1, 3};
  std::vector<int64_t> alphaShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* other1DeviceAddr = nullptr;
  void* other2DeviceAddr = nullptr;
  void* out1DeviceAddr = nullptr;
  void* out2DeviceAddr = nullptr;
  void* alphaDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* other1 = nullptr;
  aclTensor* other2 = nullptr;
  aclTensor* alpha = nullptr;
  aclTensor* out1 = nullptr;
  aclTensor* out2 = nullptr;
  std::vector<float> input1HostData = {3, 5, 7, 4, 5, 9};
  std::vector<float> input2HostData = {5, 4, 1};
  std::vector<float> other1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> other2HostData = {7, 8, 9};
  std::vector<float> out1HostData(6, 0);
  std::vector<float> out2HostData(3, 0);
  std::vector<float> alphaValueHostData = {1.2f};
  // Create input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other1 aclTensor.
  ret = CreateAclTensor(other1HostData, otherShape1, &other1DeviceAddr, aclDataType::ACL_FLOAT, &other1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create other2 aclTensor.
  ret = CreateAclTensor(other2HostData, otherShape2, &other2DeviceAddr, aclDataType::ACL_FLOAT, &other2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha aclTensor.
  ret = CreateAclTensor(alphaValueHostData, alphaShape, &alphaDeviceAddr, aclDataType::ACL_FLOAT, &alpha);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out1 aclTensor.
  ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out2 aclTensor.
  ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> tempInput1{input1, input2};
  aclTensorList* tensorListInput1 = aclCreateTensorList(tempInput1.data(), tempInput1.size());
  std::vector<aclTensor*> tempInput2{other1, other2};
  aclTensorList* tensorListInput2 = aclCreateTensorList(tempInput2.data(), tempInput2.size());
  std::vector<aclTensor*> tempOutput{out1, out2};
  aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());

  // 3. Call the CANN operator library API, and change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachSubList.
  ret = aclnnForeachSubListGetWorkspaceSize(tensorListInput1, tensorListInput2, alpha, tensorListOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachSubListGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachSubList.
  ret = aclnnForeachSubList(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachSubList failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task execution to end.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the results on the device memory to the host. Modify the code as required by the definition of the specific API.
  auto size = GetShapeSize(outShape1);
  std::vector<float> out1Data(size, 0);
  ret = aclrtMemcpy(out1Data.data(), out1Data.size() * sizeof(out1Data[0]), out1DeviceAddr,
                    size * sizeof(out1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out1 result[%ld] is: %f\n", i, out1Data[i]);
  }

  size = GetShapeSize(outShape2);
  std::vector<float> out2Data(size, 0);
  ret = aclrtMemcpy(out2Data.data(), out2Data.size() * sizeof(out2Data[0]), out2DeviceAddr,
                    size * sizeof(out2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out2 result[%ld] is: %f\n", i, out2Data[i]);
  }

  // 6. Release the aclTensor. Modify the code as required by the definition of the specific API.
  aclDestroyTensorList(tensorListInput1);
  aclDestroyTensorList(tensorListInput2);
  aclDestroyTensorList(tensorListOutput);
  aclDestroyTensor(alpha);

  // 7. Release the device resources. Modify the code as required by the definition of the specific API.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(other1DeviceAddr);
  aclrtFree(other2DeviceAddr);
  aclrtFree(out1DeviceAddr);
  aclrtFree(out2DeviceAddr);
  aclrtFree(alphaDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
