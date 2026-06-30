# aclnnAddmmWeightNz

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/mat_mul_v3)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: Computes the sum of the product of α multiplied by mat1 and mat2 and the product of β multiplied by self. Compared with the original **addmm** API, the new API supports the NZ format for mat2.
- Formula:

  $$
  out = β  self + α  (mat1 @ mat2)
  $$
  
- Example:
  * For **aclnnAddmmWeightNz**, the shape of self is [n,], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the matrix multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For **aclnnAddmmWeightNz**, the shape of self is [1, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the matrix multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For **aclnnAddmmWeightNz**, the shape of self is [m, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], and the shape of the matrix multiplication result of mat1 and mat2 is [m, n].

## Prototype


Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAddmmWeightNzGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnAddmmWeightNz** is called to perform computation.

```cpp
aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(
  const aclTensor *self, 
  const aclTensor *mat1, 
  const aclTensor *mat2, 
  const aclScalar *beta, 
  const aclScalar *alpha, 
  aclTensor       *out, 
  int8_t           cubeMathType, 
  uint64_t        *workspaceSize, 
  aclOpExecutor  **executor)
```
```cpp
aclnnStatus aclnnAddmmWeightNz(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```


## aclnnAddmmWeightNzGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1563px"><colgroup>
  <col style="width: 153px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 437px">
  <col style="width: 203px">
  <col style="width: 120px">
  <col style="width: 149px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>Input</td>
      <td>Bias matrix, which is self in the formula.</td>
      <td><ul><li>The data types of self and mat1@mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>It must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with mat1@mat2. </li> <li>When mat1 is not transposed, the dimensions are (m, k). </li><li>When mat1 is transposed, the dimensions are (k, m).</li> </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat1</td>
      <td>Input</td>
      <td>First matrix for matrix multiplication, which is mat1 in the formula.</td>
      <td><ul><li>Its data type and the data type of mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>It must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with self and mat2. </li> <li>When mat1 is not transposed, the dimensions are (m, k). </li><li>When mat1 is transposed, the dimensions are (k, m).</li></ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Second matrix for matrix multiplication, which is mat2 in the formula.</td>
      <td><ul><li>Its data type and the data type of mat1 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>). </li><li>The size of the Reduce dimension of mat2 must be the same as the size of the Reduce dimension of mat1. </li><li>It must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with self and mat2.</li> </ul>
      <li>When mat2 is not transposed, the dimensions of the AI processor affinity data layout are (n1, k1, k0, n0), where k0 = 16 and n0 = 16. k in the shape of mat1 and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.</li> 
     <li>When mat2 is transposed, the dimensions of the AI processor affinity data format are (k1, n1, n0, k0), where n0 = 16 and k0 = 16. k in the shape of mat1 and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.</li> 
      </td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
    <tr>
      <td>beta(β)</td>
      <td>Input</td>
      <td>β in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha(α)</td>
      <td>Input</td>
      <td>α in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <td>out</td>
      <td>Output</td>
      <td>Output matrix of matrix multiplication, which is out in the formula.</td>
      <td>Its data type and the data type deduced from self and mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Computation logic of the Cube unit.</td>
      <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:<ul>
        <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision.</li>
        <li>2: USE_FP16. The input data can be downgraded to FLOAT16 for computation.</li>
        <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation.</li></ul>
      </td>
      <td>INT8</td>
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
  </tbody></table>

  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>:
    - The data type BFLOAT16 is not supported.
    - cubeMathType=0 is not supported when the input data type is FLOAT32.
    - cubeMathType=1: If the input data type is FLOAT32, it is converted to FLOAT16 for computation. If the input data type is not FLOAT32, no processing is performed.
    - cubeMathType=3 is not supported.
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
      - cubeMathType=1: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, no processing is performed.
      - cubeMathType=2: If the input data type is BFLOAT16, this option is not supported.
      - cubeMathType=3: If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, this option is not supported.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 283px">
  <col style="width: 120px">
  <col style="width: 747px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed self, mat1, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or format of self or mat2 is not supported.</td>
    </tr>
    <tr>
      <td>mat1 and mat2 do not meet the condition of multiplication.</td>
    </tr>
    <tr>
      <td>The shape of out is inconsistent with that of mat1@mat2.</td>
    </tr>
  </tbody>
  </table>


## aclnnAddmmWeightNz

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAddmmWeightNzGetWorkspaceSize.</td>
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

- Determinism:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>: **aclnnAddmmWeightNz** defaults to a deterministic implementation.

- Compute consistency:
  - <term>Atlas training series products</term> and <term>Atlas inference series products</term>:
    - If strong consistency compute is enabled, the compute result is deterministic, meaning that multiple executions will generate the same result. In addition, the compute result is irrelevant to the data location.
    - **aclnnAddmmWeightNz** defaults to a non-consistent implementation. You can call **aclrtCtxSetSysParamOpt** to enable consistency compute.
    - For example, when performing matrix multiplication, the order of accumulation across different basic blocks may vary, which may lead to slight differences in results for the same data in different rows. However, when strong consistency compute is enabled, the results will remain consistent across rows as long as the inputs are the same.

- If one of the inputs of mat1 and mat2 is BFLOAT16 and the other is FLOAT or FLOAT16, the data type cannot be deduced.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_addmm.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_cast.h"

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

  // Convert the uint16_t representation of FP16 to its corresponding float value.
  float Fp16ToFloat(uint16_t h) {
    int s = (h >> 15) & 0x1;              // sign
    int e = (h >> 10) & 0x1F;             // exponent
    int f =  h        & 0x3FF;            // fraction
    if (e == 0) {
      // Zero or Denormal
      if (f == 0) {
        return s ? -0.0f : 0.0f;
      } 
      // Denormals
      float sig = f / 1024.0f;
      float result = sig * pow(2, -24);
      return s ? -result : result;
    } else if (e == 31) {
        // Infinity or NaN
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    // Normalized
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
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
    // Call aclrtMemcpy to copy the data from the host to the device.
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

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));

    const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
    size *= sizeof(T);

    // Call aclrtMalloc to allocate memory on the device.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data from the host to the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {16};
    std::vector<int64_t> mat1Shape = {16, 32};
    std::vector<int64_t> mat2Shape = {32, 16};
    std::vector<int64_t> outShape = {16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat1DeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat1 = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    aclScalar* alpha = nullptr;
    aclScalar* beta = nullptr;

    std::vector<uint16_t> selfHostData(16, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> mat1HostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> mat2HostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> outHostData(256, 0);
    float alphaValue = 1.0f;
    float betaValue = 1.0f;

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a mat1 aclTensor.
    ret = CreateAclTensor(mat1HostData, mat1Shape, &mat1DeviceAddr, aclDataType::ACL_FLOAT16, &mat1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a mat2 aclTensor.
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an alpha aclScalar.
    alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
    CHECK_RET(alpha != nullptr, return ret);
    // Create a beta aclScalar.
    beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
    CHECK_RET(beta != nullptr, return ret);


    // 3. Call the CANN operator library API, which needs to be replaced with the actual API name.
    int8_t cubeMathType = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call TransWeight.
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnTransMatmulWeight.
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnAddmmWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnAddmmWeightNzGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnAddmmWeightNz.
    ret = aclnnAddmmWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // In C language, FP16 values cannot be printed directly. They must be read as uint16_t and then converted from their binary representation into a float value.
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat1);
    aclDestroyTensor(mat2);
    aclDestroyScalar(beta);
    aclDestroyScalar(alpha);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(mat1DeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    if (workspaceSizeMm > 0) {
      aclrtFree(workspaceAddrMm);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
```
