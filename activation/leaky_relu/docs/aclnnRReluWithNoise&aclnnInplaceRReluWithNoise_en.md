# aclnnRReluWithNoise&aclnnInplaceRReluWithNoise

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T11:50:45.656Z -->

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
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Implements the RReLU (Randomized Leaky Rectified Linear Unit) activation function with noise. When the input is less than or equal to 0, the slope is a; when the input is greater than 0, the slope is 1.

- Formula:

  $$
  RReluWithNoise(self)=\begin{cases}
  self, & self\gt0 \\
  a*self, & self\le 0
  \end{cases}
  $$

  where a is a random variable that follows a uniform distribution $U$(lower, upper).
  In training mode (training == true), the noise is calculated as follows:

  $$
  noise_i = \begin{cases}
  1, & self_i \gt 0 \\
  a, & self_i \le 0
  \end{cases}
  $$

## Function Prototype

- aclnnRReluWithNoise and aclnnInplaceRReluWithNoise implement the same functionality. The differences in usage are as follows. Select the appropriate API based on your actual scenario.
  - aclnnRReluWithNoise: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceRReluWithNoise: Does not require creating a new output tensor object and directly stores the computation result in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnRReluWithNoiseGetWorkspaceSize" or "aclnnInplaceRReluWithNoiseGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size according to the process, and then call "aclnnRReluWithNoise" or "aclnnInplaceRReluWithNoise" to execute the computation.

```Cpp
aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *noise,
  const aclScalar *lower,
  const aclScalar *upper,
  bool             training,
  int64_t          seed,
  int64_t          offset,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRReluWithNoise(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

```Cpp
aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(
  const aclTensor* self,
  const aclTensor* noise,
  const aclScalar* lower,
  const aclScalar* upper,
  bool             training,
  int64_t          seed,
  int64_t          offset,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceRReluWithNoise(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnRReluWithNoiseGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1360px"><colgroup>
  <col style="width: 211px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
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
      <td>self (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for the RReluWithNoise computation, self in the formula.</td>
      <td><ul><li>The supported dimensions of shape do not exceed 25.</li><li>The data type needs to be consistent with the data type of out.</li><li>The shape needs to be consistent with the shape of out.</li><li>The data format needs to be consistent with the data format of out.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-25</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise (aclTensor*)</td>
      <td>Input</td>
      <td>noise_i in the formula.</td>
      <td><ul><li>The size must not be smaller than that of self (the shape is recommended to be consistent with that of self).</li><li>The data type needs to be consistent with the data type of self.</li><li>The shape needs to be consistent with the shape of self.</li><li>The supported dimension of shape does not exceed 25.</li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-25</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower (aclScalar*)</td>
      <td>Input</td>
      <td>lower in uniform distribution U.</td>
      <td>The data type needs to be consistent with self and out according to the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper(aclScalar*)</td>
      <td>Input</td>
      <td>The upper in the uniform distribution U.</td>
      <td>The data type needs to be consistent with self and out to satisfy the data type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training (bool)</td>
      <td>Input</td>
      <td>Distinguishes training from inference.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed (int64_t)</td>
      <td>Input</td>
      <td>Seed of the random number generator, which affects generation of the random number sequence.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset (int64_t)</td>
      <td>Input</td>
      <td>Offset of the random number generator, which affects the position in the generated random number sequence.</td>
      <td>After the offset is set, the generated random number sequence starts from the specified position.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>out (aclTensor*)</td>
      <td>Output</td>
      <td>upper in the uniform distribution U.</td>
      <td><ul><li>The data type needs to be consistent with the data type of self.</li><li>The shape needs to be consistent with the shape of self.</li><li>The data format needs to be consistent with the data format type of self.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator sub-computation stream.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

  <!-- npu="910" id7 -->
  - <term>Atlas training products</term>: supports data types FLOAT16 and FLOAT.
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
      <td>The self, noise, or out parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type of self or noise is not in the supported range.</td>
    </tr>
    <tr>
      <td>The size of self is greater than the size of noise.</td>
    </tr>
    <tr>
      <td>The data types or data formats of self, noise, and out are inconsistent.</td>
    </tr>
       <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of self or noise exceeds 25 dimensions.</td>
    </tr>
  </tbody></table>

## aclnnRReluWithNoise

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnRReluWithNoiseGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the operator computation process.</td>
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

## aclnnInplaceRReluWithNoiseGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1360px"><colgroup>
  <col style="width: 111px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 250px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>self</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>The supported dimensions of shape do not exceed 25.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-25</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>Input</td>
      <td>noise_i in the formula.</td>
      <td><ul><li>The size must not be smaller than that of self (the shape is recommended to be consistent with self).</li><li>The data type needs to be consistent with the data type of self.</li><li>The data format needs to be consistent with the data format of self.</li><li>The supported dimension of shape does not exceed 25.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0-25</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower</td>
      <td>Input</td>
      <td>lower of the uniform distribution U.</td>
      <td>The data type needs to be consistent with that of self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper</td>
      <td>Input</td>
      <td>upper of the uniform distribution U.</td>
      <td>The data type needs to be consistent with self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training</td>
      <td>Input</td>
      <td>Distinguishes training from inference.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the random number generator, which affects generation of the random number sequence.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the random number generator, which affects the position of the random number sequence generated.</td>
      <td>After the offset is set, the generated random number sequence starts from the specified position.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be applied on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which includes the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

   - <term>Atlas training products</term>: The supported data types are FLOAT16 and FLOAT.

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
      <td>The self or noise parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data type of self or noise is out of the supported range.</td>
    </tr>
       <tr>
      <td>The Size of self is greater than the Size of noise.</td>
    </tr>
       <tr>
      <td>The data type and data format of self are inconsistent with those of noise.</td>
    </tr>
    <tr>
      <td>The shape of self or noise has more than 25 dimensions.</td>
    </tr>
  </tbody></table>

## aclnnInplaceRReluWithNoise

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
      <td>Workspace memory address allocated on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device side, obtained by calling the first-phase API aclnnInplaceRReluWithNoiseGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which includes the operator sub-computation stream.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing tasks.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnRReluWithNoise&aclnnInplaceRReluWithNoise default to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_rrelu_with_noise.h"

#define CHECK_RET(cond, return_expr) \
 do {                                \
    if (!(cond)) {                   \
        return_expr;                 \
    }                                \
 } while (0)

#define LOG_PRINT(message, ...)      \
 do {                                \
    printf(message, ##__VA_ARGS__);  \
 } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i: shape) {
        shape_size *= i;
    }
    return shape_size;
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
  // Call aclrtMalloc to allocate memory on the Device side.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host-side data to the memory on the Device side.
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
  // Handle the check as needed.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Custom construction is required based on the API.
  std::vector<int64_t> selfShape = {2, 2};
  std::vector<int64_t> outShape = {2, 2};
  std::vector<int64_t> noiseShape = {2, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* noiseDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* noise = nullptr;
  aclScalar* lower = nullptr;
  aclScalar* upper = nullptr;
  std::vector<float> selfHostData = {1, 2, 3, 4};
  std::vector<float> outHostData = {0, 0, 0, 0};
  std::vector<float> noiseHostData = {4, 3, 2, 1};
  float lowerValue = 0.1f;
  float upperValue = 0.3f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create noise aclTensor
  ret = CreateAclTensor(noiseHostData, noiseShape, &noiseDeviceAddr, aclDataType::ACL_FLOAT, &noise);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create lower aclScalar
  lower = aclCreateScalar(&lowerValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lower != nullptr, return ret);
  // Create upper aclScalar
  upper = aclCreateScalar(&upperValue, aclDataType::ACL_FLOAT);
  CHECK_RET(upper != nullptr, return ret);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnRReluWithNoise API call example
  // 3. Call the first-phase API of aclnnRReluWithNoise
  ret = aclnnRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset,
                                            out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the workspaceSize calculated by the first-phase API
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRReluWithNoise
  ret = aclnnRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoise failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize and wait for the task to complete
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

  // aclnnInplaceRReluWithNoise API call example
  // step3. Call the first-phase API of aclnnInplaceRReluWithNoise.
  ret = aclnnInplaceRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset,
                                                   &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret);
                                          return ret);
  // Allocate device memory according to the workspaceSize calculated by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceRReluWithNoise.
  ret = aclnnInplaceRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoise failed. ERROR: %d\n", ret); return ret);

  // step4. (Boilerplate) Wait for the task to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // step5. Obtain the output value. Copy the result from device memory to host memory. Modify the code based on the specific API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(noise);
  aclDestroyScalar(lower);
  aclDestroyScalar(upper);

  // 7. Release device resources. Modify the code based on the specific API definition parameters.
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(noiseDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
