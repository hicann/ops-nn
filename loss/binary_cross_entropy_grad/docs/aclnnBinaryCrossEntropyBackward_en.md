# aclnnBinaryCrossEntropyBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:09:56.982Z -->

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

- Description: Computes the gradient value of binary cross-entropy backward propagation.

- Formula:

  The formula for binary cross-entropy is as follows:

  $$
  y_i = - weight_i \cdot (target_i \cdot ln(x_i) + (1 - target_i) \cdot ln(1- x_i))
  $$

  Where $x$ represents the output of the previous network layer, i.e., the forward prediction value; $target$ represents the label value of the sample. Find the partial derivative of the binary cross-entropy with respect to $x$:

  $$
  \begin{aligned}
  \frac {\partial y}{\partial x} &= -weight \cdot \frac {\partial (target_i \cdot ln(x_i) + (1-target_i) \cdot ln(1-x_i))}{\partial x} \\
  &= -weight \cdot (\frac {\partial (target \cdot ln(x))}{\partial x} + \frac {\partial ((1-target) \cdot ln(1-x))}{\partial x}) \\
  &= -weight \cdot (\frac {target}{x} - \frac {(1-target)}{1-x}) \\
  &= -weight \cdot \frac {target(1-x)-x(1-target)}{x(1-x)} \\
  &= -weight \cdot \frac {target-x}{x(1-x)} \\
  &= weight \cdot \frac {x-target}{x(1-x)}
  \end{aligned}
  $$

  $$
  out = grad\_output \cdot weight \cdot \frac {x-target}{x(1-x)} \\
  out = mean(grad\_output) \ if \ reduction = mean
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnBinaryCrossEntropyBackwardGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the procedure, and then call the "aclnnBinaryCrossEntropyBackward" API to perform the computation.

```Cpp
aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclTensor *target,
    const aclTensor *weightOptional,
    int64_t          reduction,
    aclTensor       *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBinaryCrossEntropyBackward(
    void             *workspace,
    uint64_t          workspaceSize,
    aclOpExecutor    *executor,
    const aclrtStream stream)
```

## aclnnBinaryCrossEntropyBackwardGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1537px"><colgroup>
  <col style="width: 161px">
  <col style="width: 121px">
  <col style="width: 287px">
  <col style="width: 260px">
  <col style="width: 252px">
  <col style="width: 166px">
  <col style="width: 149px">
  <col style="width: 141px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>gradOutput (aclTensor*)</td>
      <td>Input</td>
      <td>Gradient value from the previous step of the network backward propagation. The data type needs to be converted to the promotion type together with other parameters.</td>
      <td>The shape can <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Computation result of the previous layer in the network forward propagation.</td>
      <td>The data type needs to be converted to the promotion type together with other parameters.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>target (aclTensor*)</td>
      <td>Input</td>
      <td>Label value of the sample. The data type needs to be converted to the promotion type together with other parameters.</td>
      <td>The shape can be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>Must be the same as that of self.</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
       <tr>
      <td>weightOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Weight of the result.</td>
      <td><ul><li>The shape can <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self.</li><li>When weightOptional is null, a tensor of all ones must be created with the shape of self.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>Must be the same as that of self.</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>reduction (int64_t)</td>
      <td>Input</td>
      <td>Represents the reduce operation applied to the gradient computation result of binary cross entropy backward propagation.</td>
      <td><ul>Only 0 (none), 1 (mean), and 2 (sum) are supported:<li>0: no operation is performed.</li><li>1: computes the mean of the results.</li><li>2: computes the sum of the results.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>Stores the gradient computation result. The shape is the same as self.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>-</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
      <tr>
      <td>workspaceSize(uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size to be allocated on the Device.</td>
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
    - <term>Atlas inference products</term>, <term>Atlas training products</term>: The BFLOAT16 data type is not supported.

    <!-- end id7 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed gradOutput, self, target, and out are null pointers.</td>
      </tr>
      <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data types of gradOutput, self, target, and weightOptional are out of the supported range.</td>
      </tr>
      <tr>
      <td>The use case where self is a scalar but gradOutput is a non-scalar is not supported.</td>
      </tr>
      <tr>
      <td>The shapes of gradOutput, target, and weightOptional cannot be <a href="../../../docs/en/context/broadcast_relationship.md" target="_blank">broadcast</a> to the shape of self.</td>
      </tr>
      <tr>
      <td>The value of reduction is not 0, 1, or 2.</td>
      </tr>
      <tr>
      <td>The shape dimensions of gradOutput, self, target, and weightOptional are greater than 8.</td>
      </tr>
    </tbody>
    </table>

## aclnnBinaryCrossEntropyBackward

- **Parameters:**

   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
          <td>Workspace memory address allocated on the device.</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>Input</td>
          <td>Size of the workspace allocated on the device, obtained by calling the first-phase API aclnnBinaryCrossEntropyBackwardGetWorkspaceSize.</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>Input</td>
          <td>Op executor that contains the operator computation process.</td>
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

## Constraints

- Deterministic computation:
  - aclnnBinaryCrossEntropyBackward defaults to a deterministic implementation.

## Example

The sample code is as follows and is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_binary_cross_entropy_backward.h"

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
  // Boilerplate for resource initialization
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
  // Call aclrtMalloc to apply for memory on the Device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy data from the Host to the Device memory.
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
  // Handle check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output in a custom way based on the API.
  std::vector<int64_t> gradOutputShape = {2, 2};
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> targetShape = {2, 2};
  std::vector<int64_t> weightShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> gradOutputHostData = {0.1, 0.1, 0.1, 0.1};
  std::vector<float> selfHostData = {0.3, 0.3, 0.3, 0.3};
  std::vector<float> targetHostData = {0.5, 0.5, 0.5, 0.5};
  std::vector<float> weightHostData = {1, 1, 1, 1};
  std::vector<float> outHostData = {0, 0, 0, 0};
  int64_t reduction = 0;
  // Create the gradOutputaclTensor.
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the target aclTensor.
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the weight aclTensor.
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnBinaryCrossEntropyBackward API call example.
  // 3. Call the CANN operator library API. Replace it with the specific API name.
  // Call the first-phase API of aclnnBinaryCrossEntropyBackward.
  ret = aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(gradOutput, self, target, weight, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnBinaryCrossEntropyBackward.
  ret = aclnnBinaryCrossEntropyBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronize and wait until the task execution ends.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition.
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the actual API definition.
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
