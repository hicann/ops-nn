# aclnnBatchMatMulWeightNz

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- This API is used to perform matrix multiplication between the tensor self and the tensor mat2. mat2 supports only the NZ format, and self must be 3D and mat2 must be 5D.

- Formula:

  $$
  out = self@mat2
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, aclnnBatchMatMulWeightNzGetWorkspaceSize is called to obtain the input parameters and compute the required workspace size based on the process. Then, aclnnBatchMatMulWeightNz is called to perform computation.

```cpp
aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(
  const aclTensor *self, 
  const aclTensor *mat2, 
  aclTensor       *out, 
  int8_t           cubeMathType, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnBatchMatMulWeightNz(
  void            *workspace, 
  uint64_t         workspaceSize, 
  aclOpExecutor   *executor, 
  aclrtStream      stream)
```

## aclnnBatchMatMulWeightNzGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 200px">
  <col style="width: 480px">
  <col style="width: 200px">
  <col style="width: 111px">
  <col style="width: 111px">
  <col style="width: 145px">
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
      <td>First matrix for matrix multiplication, which is self in the formula.</td>
      <td><ul><li>Its data type and the data type of mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>When self is not transposed, the dimensions are (b, m, k).</li>
      <li>When self is transposed, the dimensions are (b, k, m).</li>
      <li>The first dimension b of self must meet the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the first dimension b of mat2.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Second matrix for matrix multiplication, which is mat2 in the formula.</td>
      <td><ul><li>Its data type and the data type of self must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>). </li><li>The size of the Reduce dimension of mat2 must be the same as the size of the Reduce dimension of self.</li> 
      <li>When matrix B is not transposed, the dimensions in the NZ format are (b, n1, k1, k0, n0), where k0 = 16 and n0 = 16. k in the shape of self and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.</li>
      <li>When matrix B is transposed, the dimensions in the NZ format are (b, k1, n1, n0, k0), where n0 = 16 and k0 = 16. k in the shape of self and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.</li>
      <li>The first dimension b of mat2 must be in the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with the first dimension b of self.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>NZ</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output matrix of matrix multiplication, which is out in the formula.</td>
      <td><ul><li>Its data type and the data type deduced from self and mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>The dimensions are represented as (b, m, n). m is the same as that of self. n and n1 and n0 of mat2 meet the relationship ceil(n/n0) = n1. b must be consistent with the result generated after broadcast deduction of b of self and b of mat2.</li>
      </ul></td>
      <td>BFLOAT16, FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Computation logic of the Cube unit.</td>
      <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:<ul>
        <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision. This mode is not supported.</li>
        <li>2: USE_FP16. The input data can be downgraded to FLOAT16 for computation. If the input data type is BFLOAT16, this option is not supported.</li>
        <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation. This mode is not supported.</li></ul>
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

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 650px">
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
      <td>The passed self, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or format of self or mat2 is not supported.</td>
    </tr>
    <tr>
      <td>Data type deduction cannot be performed for self and mat2.</td>
    </tr>
    <tr>
      <td>The deduced data type cannot be converted to the data type of out.</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchMatMulWeightNz

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnBatchMatMulWeightNzGetWorkspaceSize.</td>
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

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Determinism:
  - The default deterministic implementation of aclnnBatchMatMulWeightNz is used.

- If one input is BFLOAT16 and the other is FLOAT16, the data type cannot be deduced.
- self supports only three dimensions, and mat2 supports only the proprietary Ascend format. Before calling this API, you must convert mat2 from the ND format to the proprietary Ascend format.
- The last two axes of mat2 cannot be 1 (k = 1 or n = 1).

## Example

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
The data types of self and mat2 are float16. The sample code when mat2 is in NZ format is as follows (for reference only). For details about the compilation and running process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_batch_matmul.h"
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
    // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
    // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
    std::vector<int64_t> selfShape = {2, 16, 32};
    std::vector<int64_t> mat2Shape = {2, 32, 16};
    std::vector<int64_t> outShape = {2, 16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData(1024, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> mat2HostData(1024, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> outHostData(512, 0);
    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an other aclTensor.
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    int8_t cubeMathType = 0;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call TransWeight.
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnTransMatmulWeight.
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnBatchMatMulWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnBatchMatMulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // In C language, FP16 values cannot be printed directly. They must be read as uint16 and then converted from their binary representation into a float value.
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
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

- Ascend 950PR/Ascend 950DT:
The data types of self and mat2 are float16. The sample code when mat2 is in NZ format is as follows (for reference only). For details about the compilation and running process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_batch_matmul.h"
  #include "aclnnop/aclnn_npu_format_cast.h"
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
    // Call aclrtMemcpy to copy the data from the host to the memory on the device.
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
  int CreateAclTensorWithFormat(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, int64_t** storageShape,
      uint64_t* storageShapeSize, void** deviceAddr, aclDataType dataType, aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data from the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, format, *storageShape, *storageShapeSize, *deviceAddr);
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
    std::vector<int64_t> selfShape = {2, 16, 32};
    std::vector<int64_t> mat2Shape = {2, 32, 16};
    std::vector<int64_t> outShape = {2, 16, 16};

    std::vector<uint16_t> selfHostData(1024, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> mat2HostData(1024, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> outHostData(512, 0);

    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mat2NZ = nullptr;

    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dstDeviceAddr = nullptr;

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an other aclTensor.
    ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Convert the weight tensor from ND to NZ by calling the npu_format_cast API.
    aclDataType additionalDtype = aclDataType::ACL_FLOAT16;
    aclDataType srcDtype = aclDataType::ACL_FLOAT16;
    int64_t* dstShape = nullptr;
    uint64_t dstShapeSize = 0;
    int actualFormat;
    int8_t cubeMathType = 0;
    aclOpExecutor* executor = nullptr;

    uint64_t workspaceSize = 0;
    uint64_t workspaceSizeMm = 0;

    void* workspaceAddr = nullptr;
    void* workspaceAddrMm = nullptr;
    // Calculate the shape and format of the target tensor.
    ret = aclnnNpuFormatCastCalculateSizeAndFormat(mat2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

    ret = CreateAclTensorWithFormat(
        mat2HostData, mat2Shape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &mat2NZ,
        static_cast<aclFormat>(actualFormat));
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCastGetWorkspaceSize(mat2, mat2NZ, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnNpuFormatCastGetWorkspaceSize.
    ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

    // 4. Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNzGetWorkspaceSize(self, mat2NZ, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    if (workspaceSizeMm > 0) {
        ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnBatchMatMulWeightNz.
    ret = aclnnBatchMatMulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 5. (Fixed) Synchronously wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 6. Obtain the output value and copy the result from the device memory to the host. The code needs to be modified based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // In C language, FP16 values cannot be printed directly. They must be read as uint16 and then converted from their binary representation into a float value.
    for (int64_t i = 0; i < size; i++) {
        float fp16Float = Fp16ToFloat(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 7. Release the aclTensor and aclScalar. The code needs to be modified based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 8. Release device resources. Modify the code based on the API definition.
    aclrtFree(selfDeviceAddr);
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
