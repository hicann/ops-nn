# aclnnForeachMaximumScalarList

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T23:59:55.414Z -->

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

- Description: Performs element-wise comparison on the tensor list x and scalar list scalars, and computes the maximum value for each element.

- Formula:

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  {\rm y}_i = \max(x_i, \text{scalars}_i) (i=0,1,...n-1)
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnForeachMaximumScalarListGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call the "aclnnForeachMaximumScalarList" API to execute the computation.

```Cpp
aclnnStatus aclnnForeachMaximumScalarListGetWorkspaceSize(
  const aclTensorList *x,
  const aclScalarList *scalars,
  const aclTensorList *out,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachMaximumScalarList(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachMaximumScalarListGetWorkspaceSize

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
        <th>Dimension (shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>x (aclTensorList*)</td>
      <td>Input</td>
      <td>Input tensor list for the maximum computation, corresponding to <code>x</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The maximum length is 256.</li><li>All tensors in this parameter must have the same data type.</li></ul></td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalars (aclScalarList*)</td>
      <td>Input</td>
      <td>Input scalar list for the maximum computation, corresponding to <code>scalars</code> in the formula.</td>
      <td><ul><li>The number of elements equals the number of tensors in <code>x</code>, and the maximum length is 256; only the number of elements is constrained, and the rank is not limited.</li><li>The data type corresponds to the data type of input parameter <code>x</code> as follows:<ul><li>If the data type of input parameter <code>x</code> is FLOAT32, FLOAT16, or BFLOAT16, <code>scalars</code> supports FLOAT32; the aclnn API additionally supports DOUBLE (converted to FLOAT32 internally before computation).</li><li>If the data type of input parameter <code>x</code> is INT32, <code>scalars</code> supports INT64; the aclnn API additionally supports INT32 (converted to INT64 internally before computation).</li></ul></li></ul></td>
      <td>FLOAT32, DOUBLE, INT32, INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensorList*)</td>
      <td>Output</td>
      <td>Represents the output tensor list for the maximum operation, corresponding to <code>y</code> in the formula.</td>
      <td><ul><li>Empty tensors are supported.</li><li>The maximum length is 256, and the tensor list length must be the same as that of <code>x</code>.</li><li>All tensors in this parameter have the same data type.</li><li>The data type and data format are the same as those of input parameter <code>x</code>, and the shape size is greater than or equal to that of input parameter <code>x</code>.</li></ul></td>
      <td>FLOAT32, FLOAT16, INT32, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
      <td>x, scalars, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>The data types of x, scalars, or out are not supported.</td>
    </tr>
    <tr>
      <td>The data types of x and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The tensor list lengths of x and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The length of the scalars scalar list is not equal to the tensor list length of x.</td>
    </tr>
    <tr>
      <td>The data types of tensors in x or out are inconsistent.</td>
    </tr>
    <tr>
      <td>The dimensions of tensors in x or out exceed 8.</td>
    </tr>
    <tr>
      <td>The shapes of x and out do not meet the constraints.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The tensor list length of x or out exceeds 256.</td>
    </tr>
  </tbody></table>

## aclnnForeachMaximumScalarList

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
      <td>Workspace size requested on the Device, obtained by calling the first-phase API aclnnForeachMaximumScalarListGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnForeachMaximumScalarList defaults to a deterministic implementation.
- The number of tensors in input `x` and output `out` must be less than or equal to 256, and the tensor list length of `out` must be the same as that of `x`.
- The shape size of each tensor in `out` must be greater than or equal to that of the corresponding tensor in `x`.
- The scalar list length of input `scalars` must be the same as the tensor list length of `x`, and the maximum number of elements is 256.
- Input `x` and output `out` support non-contiguous tensors; non-contiguous inputs are converted to contiguous tensors before computation, and non-contiguous outputs are written back from the contiguous result.

## Example

The following sample code is for reference only. For details, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_maximum_scalar_list.h"

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
  // Boilerplate: resource initialization.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
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
  // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. The construction must be customized based on the API.
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> outShape1 = {2, 3};
  std::vector<int64_t> outShape2 = {1, 3};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* out1DeviceAddr = nullptr;
  void* out2DeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclScalar* alpha1 = nullptr;
  aclScalar* alpha2 = nullptr;
  aclTensor* out1 = nullptr;
  aclTensor* out2 = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> out1HostData(6, 0);
  std::vector<float> out2HostData(3, 0);
  float alpha1Value = 4.2f;
  float alpha2Value = 8.2f;
  // Create input1 aclTensor.
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create input2 aclTensor.
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create alpha1 aclScalar.
  alpha1 = aclCreateScalar(&alpha1Value, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha1 != nullptr, return ret);
  // Create alpha2 aclScalar.
  alpha2 = aclCreateScalar(&alpha2Value, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha2 != nullptr, return ret);
  // Create out1 aclTensor.
  ret = CreateAclTensor(out1HostData, outShape1, &out1DeviceAddr, aclDataType::ACL_FLOAT, &out1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create out2 aclTensor.
  ret = CreateAclTensor(out2HostData, outShape2, &out2DeviceAddr, aclDataType::ACL_FLOAT, &out2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<aclTensor*> tempInput{input1, input2};
  aclTensorList* tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());
  std::vector<aclTensor*> tempOutput{out1, out2};
  aclTensorList* tensorListOutput = aclCreateTensorList(tempOutput.data(), tempOutput.size());
  std::vector<aclScalar*> tempscalar{alpha1, alpha2};
  aclScalarList* scalarlist = aclCreateScalarList(tempscalar.data(), tempscalar.size());

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnForeachMaximumScalarList.
  ret = aclnnForeachMaximumScalarListGetWorkspaceSize(tensorListInput, scalarlist, tensorListOutput, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumScalarListGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnForeachMaximumScalarList.
  ret = aclnnForeachMaximumScalarList(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachMaximumScalarList failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result on the device memory to the host. Modify it based on the API definition of the specific API.
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

  // 6. Release aclTensor and aclScalar. Modify it based on the API definition of the specific API.
  aclDestroyTensorList(tensorListInput);
  aclDestroyTensorList(tensorListOutput);
  aclDestroyScalarList(scalarlist);

  // 7.Release device resources. Modify it as required by the specific API definition.
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(out1DeviceAddr);
  aclrtFree(out2DeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
