# aclnnMatmulWeightNz

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:19:03.553Z -->

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
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: Performs matrix multiplication between tensor self and tensor mat2. mat2 only supports the NZ format. Only supports self with 2 dimensions and mat2 with 4 dimensions.
  Similar APIs include aclnnMatmul (mat2 only supports ND), aclnnMm (supports matrix multiplication with 2D tensors as input), and aclnnBatchMatmul (only supports 3D matrix multiplication, where the first dimension is batch).
- Formula:

  $$
  result=self @ mat2
  $$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call the "aclnnMatmulWeightNzGetWorkspaceSize" API first to obtain the workspace size required for computation and the executor that contains the operator computation flow, and then call the "aclnnMatmulWeightNz" API to perform the computation.

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
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Shape</th>
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>self</td>
        <td>Input</td>
        <td>Represents the first matrix in the matrix multiplication, that is, self in the formula.</td>
        <td>The data type needs to be with mat2 to satisfy the data type deduction rules (see also <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br>
        - If self is not transposed, each dimension representation: (m, k).<br>
        - If self is transposed, each dimension representation: (k, m).<br></td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mat2</td>
        <td>Input</td>
        <td>Represents the second matrix in the matrix multiplication, that is, mat2 in the formula.</td>
        <td>The data type needs to satisfy the data type derivation rules with self (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br>
        The Reduce dimension of mat2 needs to be of the same size as the Reduce dimension of self.<br>
        When matrix B is not transposed, the dimension representation of the NZ format is (n1, k1, k0, n0), where k0 = 16 and n0 = 16. k in the self shape and k1 in the mat2 shape need to meet the following relationship: ceil(k, k0) = k1, and n1 in the mat2 shape and n of out need to meet the following relationship: ceil(n, n0) = n1.<br>
        When matrix B is transposed, the dimension representation of the NZ format is (k1, n1, n0, k0), where n0 = 16 and k0 = 16. k in the self shape and k1 in the mat2 shape need to meet the following relationship: ceil(k, k0) = k1, and n1 in the mat2 shape and n of out need to meet the following relationship: ceil(n, n0) = n1.<br>
        </td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>NZ</td>
        <td>4</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>Represents the output matrix of the matmul computation, that is, out in the formula.</td>
        <td>The data type needs to satisfy the derivation rules with the data type derived from self and mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a> and <a href="#constraints">Constraints</a>).<br> The dimension representation is (m, n), where m is the same as the m of self, and n satisfies the relationship ceil(n / n0) = n1 with n1 and n0 of mat2.</td>
        <td>BFLOAT16, FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>Input</td>
        <td>Specifies the computation logic of the Cube unit.</td>
        <td>If the input data types have a deduction relationship, this parameter processes the mutually derived data type by default. The supported enumeration values are as follows:<ul>
          <li>0: KEEP_DTYPE, which keeps the input data type for computation.</li>
          <li>1: ALLOW_FP32_DOWN_PRECISION, which supports downgrading the input data precision for computation. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of another data type, no processing is performed.</li>
          <li>2: USE_FP16, which supports downgrading the input precision to FLOAT16 for computation. This option is not supported when the input data type is BFLOAT16.</li>
          <li>3: USE_HF32, which supports downgrading the input to data type HFLOAT32 for computation. When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. This option is not supported when the input is of another data type.</li>
          <li>4: USE_FP32_ADD, which supports computation in high precision.</li></ul>
        </td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

  <!-- npu="A3,910b,310p" id7 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Atlas inference products</term>:
    - Before calling this API, you must use the aclnnTransMatmulWeight API to convert the original input Format of mat2 from ND to NZ.
    - When the data type of mat2 is FLOAT32, aclnnTransMatmulWeight does not support the conversion of this data type (the first API returns 161002). In this case, use the aclnnNpuFormatCast API to convert mat2 from ND to NZ.
    - cubeMathType=4: When the input data type is FLOAT32 and the k-axis is greater than 2048, group accumulation is used for computation. When the input is of other data types or the k-axis is less than 2048, no processing is performed.
  <!-- end id7 -->
  <!-- npu="950" id8 -->
  - <term>Ascend 950PR&950DT products</term>:
    - Before calling this API, you must use aclnnNpuFormatCast to convert the original input format of mat2 from ND to NZ.
    - The option cubeMathType = 1: ALLOW_FP32_DOWN_PRECISION is not supported.
    - The option cubeMathType = 3: USE_HF32 is not supported.
    - The option cubeMathType = 4: USE_FP32_ADD is not supported.
    - Not supported: self and mat2 with data type FLOAT32

  <!-- end id8 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Value</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input self, mat2, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The data type and data format of self and mat2 are not in the supported range.</td>
    </tr>
    <tr>
      <td>The data types of self and mat2 cannot be derived.</td>
    </tr>
    <tr>
      <td>The derived data type cannot be converted to the type of the specified output out.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>The workspace memory address requested on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the Device, obtained by the first-phase API aclnnMatmulWeightNzGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>
  </div>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="950,910,310p" id9 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>, <term>Ascend 950PR&950DT products</term>: aclnnMatmulWeightNz defaults to a deterministic implementation.

  <!-- end id9 -->

<!-- npu="950" id10 -->
- <term>Ascend 950PR&950DT products</term>:
  - When the data types of self and mat2 are both FLOAT16 or both BFLOAT16, and the data type of out is FLOAT32, the multiply-add result is accumulated in FLOAT32, and the final result is written to out in FLOAT32.

<!-- end id10 -->
- Computation consistency description

  <!-- npu="910,310p" id11 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - When the strong consistency computation feature is enabled, the computation result is deterministic, and multiple executions produce the same output. In addition, the computation result is independent of the data location.
    - aclnnMatmulWeightNz defaults to a non-deterministic implementation and supports enabling deterministic computation via aclrtCtxSetSysParamOpt.
    - For example, during Matrix Multiplication, the accumulation order of different basic blocks may vary, which can cause slight differences in the computation results of the same data across different rows. However, when deterministic computation is enabled, the computation results will be identical as long as the inputs are the same, even across different rows.

  <!-- end id11 -->

- Data type deduction between BFLOAT16 and FLOAT16 inputs is not supported.
- self only supports 2 dimensions, and mat2 only supports the Ascend private format. Before calling this API, you must complete the conversion of mat2 from ND to the Ascend private format.
- The case where one of the last two axes of mat2 is 1, i.e., k=1 or n=1, is not supported.

<!-- npu="310p" id12 -->
- <term>Atlas inference products</term>: aclnnMatmulWeightNz does not support BFLOAT16 as the input data type.
<!-- end id12 -->
- Recommended use case: It is recommended to use this API in mte2 bound scenarios, for example, when the M axis is small, matrix A involves repeated data movement, and matrix B does not.

## Example

<!-- npu="A3,910b" id13 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  The following is the sample code for the use case where the data types of self and mat2 are float16 and mat2 is in NZ format. It is for reference only. For the compilation and execution procedure, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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

  // Convert the uint16_t representation of FP16 to a float representation.
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to allocate device memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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

    // Call aclrtMalloc to allocate device memory.
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy data from the host to device memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute strides of the contiguous tensor.
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
    // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> selfShape = {16, 32};
    std::vector<int64_t> mat2Shape = {32, 16};
    std::vector<int64_t> outShape = {16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData(512, 0x3C00); // 1.0 of float16 is represented by 0x3C00
    std::vector<uint16_t> mat2HostData(512, 0x3C00); // 1.0 of float16 is represented by 0x3C00
    std::vector<uint16_t> outHostData(256, 0);
    // Create the self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the other aclTensor.
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Change it to the specific API name.
    int8_t cubeMathType = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call TransWeight.
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API aclnnTransMatmulWeight.
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API aclnnMatmulWeightNz.
    uint64_t workspaceSizeMm = 0;
    ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API aclnnMatmulWeightNz.
    ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the actual API definition.
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // In C, fp16 data cannot be printed directly. Read it as uint16 and manually convert the binary to the fp16 value in float representation.
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. Release aclTensor and aclScalar. Modify the code based on the actual API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 7. Release device resources. Modify the code based on the actual API definition.
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

<!-- end id13 -->
<!-- npu="A3,910b" id14 -->
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
  The following sample code is for the linear layer or transposed input weight use case (mathematical semantics: `out = self @ mat2.T`) and is for reference only. In this example, the physical layout of weight is `[N, K]`. After configuring `view_shape=[K, N]`, the corresponding `view_strides`, and `storage_shape=[N, K]` at the same time via `aclCreateTensor`, call `aclnnTransMatmulWeight` and `aclnnMatmulWeightNz`. For details about the compilation and execution, see the [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_matmul.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_cast.h"

  // Linear layer / transposed input weight use case: out = self @ weight.T, physical layout of weight [N,K], logical view [K,N]

  using IntArrayPtr = std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray*)>;

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
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      // Boilerplate: resource initialization
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
                      aclDataType dataType, aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
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

  // Create the weight with physical layout [N, K] as an aclTensor with logical view [K, N] (B transposed).
  // MatmulWeightNz still validates the matmul based on the view_shape of mat2, so view_shape must be [K, N].
  template <typename T>
  int CreateAclTensorWeightBTrans(const std::vector<T>& hostData, const std::vector<int64_t>& viewShape,
                                  const std::vector<int64_t>& storageShape, void** deviceAddr, aclDataType dataType,
                                  aclTensor** tensor)
  {
      // viewShape: [K, N], storageShape: [N, K]
      CHECK_RET(viewShape.size() == 2 && storageShape.size() == 2, LOG_PRINT("viewShape/storageShape dim must be 2.\n");
                return 1);
      CHECK_RET(viewShape[0] == storageShape[1] && viewShape[1] == storageShape[0],
                LOG_PRINT("viewShape must be transpose of storageShape.\n");
                return 1);

      // The space required by NZ is calculated based on logical [K, N].
      auto size = static_cast<uint64_t>(GetShapeSize(viewShape));
      aclIntArray* mat2SizeRaw = aclCreateIntArray(viewShape.data(), viewShape.size());
      CHECK_RET(mat2SizeRaw != nullptr, LOG_PRINT("aclCreateIntArray failed.\n"); return 1);
      IntArrayPtr mat2Size(mat2SizeRaw, aclDestroyIntArray);
      auto ret = aclnnCalculateMatmulWeightSize(mat2Size.get(), &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
      size *= sizeof(T);

      // Call aclrtMalloc to allocate device memory (must cover the subsequent ND->NZ conversion).
      ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      // Copy only the physical ND data [N, K].
      auto hostBytes = static_cast<size_t>(GetShapeSize(storageShape)) * sizeof(T);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), hostBytes, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Transposed view [K, N] of the contiguous [N, K]: strides = [1, K]
      std::vector<int64_t> strides = {1, storageShape[1]};

      // view_shape, view_strides, and storage_shape must all be correct.
      *tensor = aclCreateTensor(viewShape.data(), viewShape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                storageShape.data(), storageShape.size(), *deviceAddr);
      return 0;
  }

  int main()
  {
      // 1. (Boilerplate) Device/stream initialization. For details, see the acl API manual.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output (linear layer / input weight transposition).
      // self: [M, K], weight physical [N, K], logical view [K, N], out: [M, N]
      // Mathematical semantics: out = self @ weight.T
      int64_t m = 32;
      int64_t k = 5120;
      int64_t n = 8704;
      std::vector<int64_t> selfShape = {m, k};
      std::vector<int64_t> mat2ViewShape = {k, n};    // Logical shape, used in MatmulWeightNz verification
      std::vector<int64_t> mat2StorageShape = {n, k}; // Physical layout
      std::vector<int64_t> outShape = {m, n};
      void* selfDeviceAddr = nullptr;
      void* mat2DeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* self = nullptr;
      aclTensor* mat2 = nullptr;
      aclTensor* out = nullptr;
      std::vector<uint16_t> selfHostData(static_cast<size_t>(m * k), 0x3C00); // float16_t uses 0x3C00 to represent 1
      std::vector<uint16_t> mat2HostData(static_cast<size_t>(n * k), 0x3C00); // Physical [N, K]
      std::vector<uint16_t> outHostData(static_cast<size_t>(m * n), 0);
      // Create the self aclTensor.
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mat2 aclTensor: configure [N, K] as the transposed view [K, N].
      ret = CreateAclTensorWeightBTrans(mat2HostData, mat2ViewShape, mat2StorageShape, &mat2DeviceAddr,
                                        aclDataType::ACL_FLOAT16, &mat2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the actual API name.
      int8_t cubeMathType = 1;
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      // Call TransWeight: ND -> NZ.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      // Allocate device memory according to the workspaceSize computed by the first-phase API.
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
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      // Allocate device memory according to the workspaceSize computed by the first-phase API.
      void* workspaceAddrMm = nullptr;
      if (workspaceSizeMm > 0) {
          ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnMatmulWeightNz.
      ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Synchronously wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result from device memory to host memory. Modify the code based on the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<aclFloat16> resultData(static_cast<size_t>(size), 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      // The output is large in size. Only the first few elements are printed for reference.
      int64_t printNum = size < 8 ? size : 8;
      for (int64_t i = 0; i < printNum; i++) {
          float fp16Float = aclFloat16ToFloat(resultData[static_cast<size_t>(i)]);
          LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
      }

      // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
      aclDestroyTensor(self);
      aclDestroyTensor(mat2);
      aclDestroyTensor(out);

      // 7. Release device resources. Modify the code based on the specific API definition.
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

<!-- end id14 -->
<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term>:
  The following is the sample code for the use case where the data types of self and mat2 are bfloat16 and mat2 is in NZ format, for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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

      // Keep the exponent offset unchanged.
      // Shift the mantissa left by 23 - 7, and pad the remaining bits with 0s.
      uint32_t f_bits = sign | (exponent << 23) | (mantissa << (23 - 7));
      // Force cast to float.
      return *reinterpret_cast<float*>(&f_bits);
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
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy data from the host to the device memory.
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

    template <typename T>
    int CreateAclTensorWithFormat(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
                                  int64_t** storageShape, uint64_t* storageShapeSize, void** deviceAddr,
                                  aclDataType dataType, aclTensor** tensor, aclFormat format) {
        auto size = hostData.size() * sizeof(T);
        // Call aclrtMalloc to allocate memory on the device.
        auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
        // Call aclrtMemcpy to copy data from the host to the device memory.
        ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

        // Calculate the strides of the contiguous tensor.
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = shape.size() - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                    format, *storageShape, *storageShapeSize, *deviceAddr);
        return 0;
    }

    int main() {
      // 1. Device/stream initialization. For details, see the acl API manual (boilerplate).
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output. Custom construction is required based on the API.
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

      std::vector<uint16_t> selfHostData(m * k, 0x3F80); // bfloat16_t uses 0x3F80 to represent 1 in uint_16.
      std::vector<uint16_t> mat2HostData(k * n, 0x3F80); // bfloat16_t uses 0x3F80 to represent 1 in uint_16.
      std::vector<uint16_t> outHostData(m * n, 0);
      // weightNz requires space greater than or equal to the original [k,n] matrix and must be aligned to 16.
      int64_t kAlign = (k + 16 - 1) / 16 * 16;
      int64_t nAlign = (n + 16 - 1) / 16 * 16;
      std::vector<uint16_t> dstTensorHostData(kAlign * nAlign, 0x3F80);

      aclDataType srcDtype = aclDataType::ACL_BF16;
      aclDataType additionalDtype = aclDataType::ACL_BF16;
      // Create the self aclTensor.
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, srcDtype, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mat2 aclTensor.
      ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, srcDtype, &mat2);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, srcDtype, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. Call the CANN operator library API. Replace it with the specific API name.
      int8_t cubeMathType = 0; // 0: KEEP_DTYPE, which keeps the input data type for computation.
      aclOpExecutor* executor = nullptr;

      // Convert the weight tensor from ND to NZ by calling the npu_format_cast API.
      int64_t* dstShape = nullptr;
      uint64_t dstShapeSize = 0;
      int actualFormat;
      uint64_t workspaceSize = 0;
      void* workspaceAddr = nullptr;

      uint64_t workspaceSizeMm = 0;
      void* workspaceAddrMm = nullptr;

      // Calculate the shape and format of the target tensor. The target format is NZ.
      ret = aclnnNpuFormatCastCalculateSizeAndFormat(mat2, ACL_FORMAT_FRACTAL_NZ, additionalDtype, &dstShape, &dstShapeSize, &actualFormat);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastCalculateSizeAndFormat failed. ERROR: %d\n", ret); return ret);

      ret = CreateAclTensorWithFormat(dstTensorHostData, mat2Shape, &dstShape, &dstShapeSize, &dstDeviceAddr, srcDtype, &mat2NZ, static_cast<aclFormat>(actualFormat));
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensorWithFormat failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCastGetWorkspaceSize(mat2, mat2NZ, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory using the workspaceSize calculated by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      // Call the second-phase API aclnnNpuFormatCastGetWorkspaceSize.
      ret = aclnnNpuFormatCast(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuFormatCast failed. ERROR: %d\n", ret); return ret);

      // 4. Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API aclnnMatmulWeightNz.
      ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2NZ, out, cubeMathType, &workspaceSizeMm, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      // Allocate device memory using the workspaceSize calculated by the first-phase API.
      if (workspaceSizeMm > 0) {
        ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API aclnnMatmulWeightNz.
      ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

      // 4. Wait synchronously for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value, and copy the result on the device side to the host side. Modify it according to the specific API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
        float bf16Float = Bf16ToFloat(resultData[i]);
        LOG_PRINT("result[%ld] is: %f\n", i, bf16Float);
      }

      // 6. Release aclTensor and aclScalar.
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

<!-- end id15 -->
