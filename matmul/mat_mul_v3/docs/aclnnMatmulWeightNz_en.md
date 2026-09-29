# aclnnMatmulWeightNz

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Description: Performs matrix multiplication between the self tensor and the mat2 tensor. The mat2 tensor supports only the NZ format. The self tensor must be 2-dimensional, and the mat2 tensor must be 4-dimensional.
  Similar APIs include aclnnMatmul (mat2 supports only ND), aclnnMm (supporting matrix multiplication with 2-dimensional tensors as the input), and aclnnBatchMatmul (supporting only 3-dimensional matrix multiplication, where the first dimension is batch).
- Formula:

  $$
  result=self @ mat2
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMatmulWeightNzGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMatmulWeightNz** is called to perform computation.

```cpp
aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat2,
  aclTensor       *out,
  int8_t          cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMatmulWeightNz(
  void           *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream    stream)
```

## aclnnMatmulWeightNzGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
    <col style="width: 151px">
    <col style="width: 121px">
    <col style="width: 301px">
    <col style="width: 331px">
    <col style="width: 237px">
    <col style="width: 111px">
    <col style="width: 111px">
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
        <td>The first matrix of matrix multiplication, that is, self in the formula.</td>
        <td>The data types of self and mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br>
        - When self is not transposed, the dimensions are (m, k).<br>
        - When self is transposed, the dimensions are represented as follows: (k, m) </td>
        <td>BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mat2</td>
        <td>Input</td>
        <td>The second matrix of matrix multiplication, that is, mat2 in the formula.</td>
        <td>The data types of mat2 and self must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br>
        The reduced dimensions of mat2 and self must have the same size.<br>
        When matrix B is not transposed, the dimensions in the NZ format are (n1, k1, k0, n0), where k0 = 16 and n0 = 16. k in the shape of self and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.<br>
        When matrix B is transposed, the dimensions in the NZ format are (k1, n1, n0, k0), where n0 = 16 and k0 = 16. k in the shape of self and k1 in the shape of mat2 must meet the following relationship: ceil(k, k0) = k1. n1 in the shape of mat2 and n of out must meet the following relationship: ceil(n, n0) = n1.<br>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>NZ</td>
        <td>4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output matrix of matrix multiplication, that is, out in the formula.</td>
        <td>Its data type and the data type deduced from self and mat2 must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br> The dimensions are represented as (m, n). m is the same as that of self. n and n1 and n0 of mat2 meet the relationship ceil(n/n0) = n1.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>Input</td>
        <td>Compute logic of the Cube unit.</td>
        <td>If the input data types can be deduced from each other, this parameter processes the deduced data type by default. The supported enumerated values are as follows:<ul>
          <li>0: KEEP_DTYPE. The input data type is retained for computation.</li>
          <li>1: ALLOW_FP32_DOWN_PRECISION. The input data can be computed with reduced precision. If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, no processing is performed.</li>
          <li>2: USE_FP16. The input data can be downgraded to FLOAT16 for computation. If the input data type is BFLOAT16, this option is not supported.</li>
          <li>3: USE_HF32. The input data can be downgraded to HFLOAT32 for computation. If the input data type is FLOAT32, it is converted to HFLOAT32 for computation. If the input data type is not FLOAT32, this option is not supported.</li>
          <li>4: USE_FP32_ADD, indicating that high-precision calculation is supported.</li></ul>
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

  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - Before calling this API, you must use the aclnnTransMatmulWeight API to convert the original format of mat2 from ND to NZ.
    - When **cubeMathType** is set to **4**, if the input data type is FLOAT32 and the k-axis is greater than 2048, grouped accumulation is used for computation. If the input data type is not FLOAT32 or the k-axis is less than 2048, no processing is performed.
  - Ascend 950PR/Ascend 950DT:
    - Before calling this API, you must use the aclnnNpuFormatCast API to convert the original format of mat2 from ND to NZ.
    - The option of cubeMathType being 1 (ALLOW_FP32_DOWN_PRECISION) is not supported.
    - The option of cubeMathType being 3 (USE_HF32) is not supported.
    - The option of cubeMathType being 4 (USE_FP32_ADD) is not supported.

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation. The following errors may be returned:
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The passed self, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
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

## aclnnMatmulWeightNz

- **Parameters**

  <div style="overflow-x: auto;">
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnMatmulWeightNzGetWorkspaceSize.</td>
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
  </div>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - <term>Atlas training products</term>, <term>Atlas inference products</term>, and Ascend 950PR/Ascend 950DT: The default implementation of aclnnMatmulWeightNz is deterministic.

- Computational Consistency Description
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - If strong consistency compute is enabled, the compute result is deterministic, meaning that multiple executions will generate the same result. In addition, the compute result is irrelevant to the data location.
    - **aclnnMatmulWeightNz** defaults to a non-consistent implementation. You can call **aclrtCtxSetSysParamOpt** to enable consistency compute.
    - For example, during matrix multiplication, the accumulation sequence of different basic blocks may be different. As a result, the computation results of the same data in different rows may be slightly different. However, when strong consistency compute is enabled, the results will remain consistent across rows as long as the inputs are the same.

- If one input is BFLOAT16 and the other is FLOAT16, the data type cannot be deduced.
- self supports only two dimensions, and mat2 supports only the proprietary Ascend format. Before calling this API, you must convert mat2 from the ND format to the proprietary Ascend format.
- The last two axes of mat2 cannot be 1 (k = 1 or n = 1).
- It is recommended that this API be used in the MTE2 bound scenario, for example, when the M axis is small, matrix A is repeatedly moved, and matrix B is not repeatedly moved.

## Example

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  The data types of self and mat2 are float16. The sample code when mat2 is NZ format is as follows (for reference only). For details about the compilation and running process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_matmul.h"
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
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
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
    std::vector<int64_t> selfShape = {16, 32};
    std::vector<int64_t> mat2Shape = {32, 16};
    std::vector<int64_t> outShape = {16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> mat2HostData(512, 0x3C00); // 0x3C00 in float16_t represents 1 in int_16.
    std::vector<uint16_t> outHostData(256, 0);
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
    int8_t cubeMathType = 1;
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

    // Call the first-phase API of aclnnMatmulWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMatmulWeightNz.
    ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

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
  The data types of self and mat2 are bfloat16. The sample code when mat2 is NZ format is as follows (for reference only). For details about the compilation and running process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  
  ```Cpp
    #include <iostream>
    #include <vector>
    #include "acl/acl.h"
    #include "aclnnop/aclnn_matmul.h"
    #include "aclnnop/aclnn_npu_format_cast.h"

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

    // Convert the uint16_t representation of bfloat16 to the float representation.
    float Bf16ToFloat(uint16_t h) {
      uint32_t sign = (h & 0x8000U) ? 0x80000000U : 0x00000000U;  // sign bit
      uint32_t exponent = (h >> 7) & 0x00FFU;                      // exponent bits
      uint32_t mantissa =  h & 0x007FU;                           // mantissa bits

      // The exponent offset remains unchanged.
      // The mantissa is shifted left by 23 minus 7, and the rest is padded with 0s.
      uint32_t f_bits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Forcibly convert to float.
      return *reinterpret_cast<float*>(&f_bits);
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

    template <typename T>
    int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                                  int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                  aclDataType dataType, aclTensor** tensor, aclFormat format) {
        auto size = hostData.size() * sizeof(T);
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

        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                    format, *storageShape, *storageShapeSize, *deviceAddr);
        return 0;
    }

    int main() {
      //1. Initialize the device and stream. For details, see the ACL API manual. (This is a fixed method.)
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API.
      int64_t m = 16;
      int64_t k = 32;
      int64_t n = 16;
      std::vector<int64_t> selfShape = {m, k};
      std::vector<int64_t> mat2Shape = {k, n};
      std::vector<int64_t> outShape = {m, n};
      void* selfDeviceAddr = nullptr;
      void* mat2DeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      void* dstDeviceAddr = nullptr;

      aclTensor* self = nullptr;
      aclTensor* mat2 = nullptr;
      aclTensor* out = nullptr;
      aclTensor* mat2NZ = nullptr;

      std::vector<uint16_t> selfHostData(m * k, 0x3F80); // bfloat16_t indicates 1 of uint_16 by using 0x3F80.
      std::vector<uint16_t> mat2HostData(k * n, 0x3F80); // bfloat16_t indicates 1 of uint_16 by using 0x3F80.
      std::vector<uint16_t> outHostData(m * n, 0);
      // The space required by weightNz is greater than or equal to that of the original [k, n] matrix, and needs to be aligned with 16.
      int64_t kAlign = (k + 16 - 1) / 16 * 16;
      int64_t nAlign = (n + 16 - 1) / 16 * 16;
      std::vector<uint16_t> dstTensorHostData(kAlign * nAlign, 0x3F80);

      aclDataType srcDtype = aclDataType::ACL_BF16;
      aclDataType additionalDtype = aclDataType::ACL_BF16;
      // Create a self aclTensor.
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, srcDtype, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a mat2 aclTensor.
      ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, srcDtype, &mat2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, srcDtype, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      int8_t cubeMathType = 1;
      aclOpExecutor* executor = nullptr;

      // Convert the weight tensor from ND to NZ by calling the npu_format_cast API.
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;
      uint64_t workspaceSize = 0;
      void* workspaceAddr = nullptr;

      uint64_t workspaceSizeMm = 0;
      void* workspaceAddrMm = nullptr;

      // Calculate the shape and format of the target tensor.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(mat2, 29, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, mat2Shape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &mat2NZ, static_cast<aclFormat>(actualFormat));
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

      // Call the first-phase API of aclnnMatmulWeightNz.
      ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2NZ, out, cubeMathType, &workspaceSizeMm, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory based on workspaceSize computed by the first-phase API.
      if (workspaceSizeMm > 0) {
        ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnMatmulWeightNz.
      ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        float bf16Float = Bf16ToFloat(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, bf16Float);
      }

      // 6. Release aclTensors and aclScalars.
      aclDestroyTensor(self);
      aclDestroyTensor(mat2);
      aclDestroyTensor(out);
      aclDestroyTensor(mat2NZ);

      // 7. Release device resources.
      aclrtFree(selfDeviceAddr);
      aclrtFree(mat2DeviceAddr);
      aclrtFree(outDeviceAddr);
      aclrtFree(dstDeviceAddr);

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
