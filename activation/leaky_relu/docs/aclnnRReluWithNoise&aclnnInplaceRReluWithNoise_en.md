# aclnnRReluWithNoise&aclnnInplaceRReluWithNoise

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/activation/leaky_relu)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     √    |

## Function

- Description: Performs the activation function of the random rectified linear unit with noise. When the input is less than or equal to 0, the slope is a. When the input is greater than 0, the slope is 1.

- Formula:

  $$
  RReluWithNoise(self)=\begin{cases}
  self, & self\gt0 \\
  a*self, & self\le 0
  \end{cases}
  $$

  a is a random variable that follows an even distribution of $U$(lower,upper).
  If the training mode (training == true) is used, the noise calculation formula is as follows:
  
  $$
  noise_i = \begin{cases}
  1, & self_i \gt 0 \\
  a, & self_i \le 0
  \end{cases}
  $$

## Prototype

- **aclnnRReluWithNoise** and **aclnnInplaceRReluWithNoise** implement the same function in different ways. Select a proper operator based on your requirements.
  - **aclnnRReluWithNoise**: An output tensor object needs to be created to store the computation result.
  - **aclnnInplaceRReluWithNoise**: No output tensor object needs to be created, and the computation result is stored in the memory of the input tensor.
- Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnRReluWithNoiseGetWorkspaceSize** or **aclnnInplaceRReluWithNoiseGetWorkspaceSize** is called to obtain input parameters and compute the required workspace size based on the computation process. Then, **aclnnRReluWithNoise** or **aclnnInplaceRReluWithNoise** is called to perform computation.

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>self</td>
      <td>Input</td>
      <td>Input parameter for the RReluWithNoise computation, corresponding to self in the formula.</td>
      <td><ul><li>The shape supports a maximum of 32 dimensions. </li><li>The data type must be the same as that of out. </li><li>The shape must be the same as that of out. </li><li>The data format must be the same as that of out. </li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0–32</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>Input</td>
      <td>noise_i in the formula.</td>
      <td><ul><li>The size must be greater than or equal to that of self. It is recommended that the shape be the same as that of self. </li><li>The data type must be the same as that of self. </li><li>The shape must be the same as that of self. </li><li>The shape supports a maximum of 32 dimensions. </li><li>Empty tensors are supported.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0–32</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower</td>
      <td>Input</td>
      <td>Lower bound in the even distribution U.</td>
      <td>Its data type and the data types of self and out must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper</td>
      <td>Input</td>
      <td>Upper bound in the even distribution U.</td>
      <td>Its data type and the data types of self and out must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md" target="_blank">Deduction Relationship</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training</td>
      <td>Input</td>
      <td>Whether training or inference is performed.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the random number generator, which affects the generated random number sequence.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the random number generator, which affects the position of the generated random number sequence.</td>
      <td>After the offset is set, the generated random number sequence starts from the specified position.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>out</td>
      <td>Output</td>
      <td>Upper bound in the even distribution U.</td>
      <td><ul><li>The data type must be the same as that of self. </li><li>The shape must be the same as that of self. </li><li>The data format must be the same as that of self.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
   - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT.

  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>The passed self, noise, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of self or noise is not supported.</td>
    </tr>
    <tr>
      <td>The size of self is greater than that of noise.</td>
    </tr>
    <tr>
      <td>The data types or data formats of self, noise, and out are inconsistent.</td>
    </tr>
       <tr>
      <td>The shapes of self and out are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of self or noise has more than 32 dimensions.</td>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnRReluWithNoiseGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

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
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
      <tr>
      <td>self</td>
      <td>Input</td>
      <td>self in the formula.</td>
      <td>The shape supports a maximum of 32 dimensions.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0–32</td>
      <td>√</td>
    </tr>
    <tr>
      <td>noise</td>
      <td>Input</td>
      <td>noise_i in the formula.</td>
      <td><ul><li>The size must be greater than or equal to that of self. It is recommended that the shape be the same as that of self. </li><li>The data type must be the same as that of self. </li><li>The data format must be the same as that of self. </li><li>The shape supports a maximum of 32 dimensions.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
      <td>0–32</td>
      <td>√</td>
    </tr>
      <tr>
      <td>lower</td>
      <td>Input</td>
      <td>Lower bound in the even distribution U.</td>
      <td>The data type must be the same as that of self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>upper</td>
      <td>Input</td>
      <td>Upper bound in the even distribution U.</td>
      <td>The data type must be the same as that of self.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>training</td>
      <td>Input</td>
      <td>Whether training or inference is performed.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>seed</td>
      <td>Input</td>
      <td>Seed of the random number generator, which affects the generated random number sequence.</td>
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Offset of the random number generator, which affects the position of the generated random number sequence.</td>
      <td>After the offset is set, the generated random number sequence starts from the specified position.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
   - <term>Atlas training series products</term>: The data type can be FLOAT16 or FLOAT.  


  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>The passed self or noise is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The data type of self or noise is not supported.</td>
    </tr>
       <tr>
      <td>The size of self is greater than that of noise.</td>
    </tr>
       <tr>
      <td>The data types or data formats of self and noise are inconsistent.</td>
    </tr>
    <tr>
      <td>The shape of self or noise has more than 32 dimensions.</td>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnInplaceRReluWithNoiseGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic compute:
  - **aclnnRReluWithNoise** and **aclnnInplaceRReluWithNoise** default to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
  // (Fixed writing) Initialize resources.
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
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API.
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
  // Create a self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a noise aclTensor.
  ret = CreateAclTensor(noiseHostData, noiseShape, &noiseDeviceAddr, aclDataType::ACL_FLOAT, &noise);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a lower aclScalar.
  lower = aclCreateScalar(&lowerValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lower != nullptr, return ret);
  // Create an upper aclScalar.
  upper = aclCreateScalar(&upperValue, aclDataType::ACL_FLOAT);
  CHECK_RET(upper != nullptr, return ret);
  bool training = false;
  int64_t seed = 0;
  int64_t offset = 0;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  
  // aclnnRReluWithNoise API call example
  // 3. Call the first-phase API of aclnnRReluWithNoise.
  ret = aclnnRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset, 
                                            out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on the computed workspaceSize.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnRReluWithNoise.
  ret = aclnnRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRReluWithNoise failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceRReluWithNoise API call example
  // 3. Call the first-phase API of aclnnInplaceRReluWithNoise.
  ret = aclnnInplaceRReluWithNoiseGetWorkspaceSize(self, noise, lower, upper, training, seed, offset, 
                                                   &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoiseGetWorkspaceSize failed. ERROR: %d\n", ret); 
                                          return ret);
  // Allocate device memory based on the computed workspaceSize.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second-phase API of aclnnInplaceRReluWithNoise.
  ret = aclnnInplaceRReluWithNoise(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceRReluWithNoise failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor. Modify the configuration based on the API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(noise);
  aclDestroyScalar(lower);
  aclDestroyScalar(upper);

  // 7. Release device resources. Set the parameters based on the API definition.
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
