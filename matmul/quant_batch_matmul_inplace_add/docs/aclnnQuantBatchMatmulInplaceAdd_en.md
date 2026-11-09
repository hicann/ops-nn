# aclnnQuantBatchMatmulInplaceAdd

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: In micro-batch training, gradient accumulation across micro-batches leads to many use cases where QuantBatchMatmul is followed by InplaceAdd. The QuantBatchMatmulInplaceAdd operator fuses these operators to improve network performance. It implements quantized matrix multiplication and addition. Its basic function is a combination of matrix multiplication and addition.

- Formula:

  $X_1$ and $X_2$ are the matrices after applying the transpose attributes, with shapes $(M,K)$ and $(K,N)$, respectively. $Y_{\mathrm{in}}$ is `yRef` before the call, and $Y_{\mathrm{out}}$ is the result after the in-place update. Both have shape $(M,N)$.

  - MX (Microscaling) quantization:

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+\sum_{j=0}^{L-1}
    \left(\sum_{t=0}^{g_j-1}X_1[m,jg_K+t]X_2[jg_K+t,n]\right)S_1[m,j]S_2[j,n]
    $$

    Here, $g_K=32$, $L=\lceil K/g_K\rceil$, and $g_j=\min(g_K,K-jg_K)$. $S_1$ and $S_2$ are the scales for the corresponding groups in `x1ScaleOptional` and `x2Scale`, respectively. For their storage shapes, see [Constraints](#constraints).

  - HIFLOAT8 T-T quantization:

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+
    \left(\sum_{k=0}^{K-1}X_1[m,k]X_2[k,n]\right)s_1s_2
    $$

    $s_1$ and $s_2$ correspond to `x1ScaleOptional` and `x2Scale`, respectively. Both have shape $(1,)$.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call the "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation process, and then call the "aclnnQuantBatchMatmulInplaceAdd" API to perform the computation.

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  const aclTensor* x1ScaleOptional,
  const aclTensor* x2Scale,
  aclTensor*       yRef,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize

- **Parameters**
  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1 (const aclTensor*)</td>
        <td>Input</td>
        <td>Input x1 in the Formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2 (const aclTensor*)</td>
        <td>Input</td>
        <td>Input x2 in the Formula.</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN, FLOAT8_E5M2, FLOAT4_E2M1, HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1ScaleOptional (const aclTensor*)</td>
        <td>Input</td>
        <td>Indicates the scaling factor introduced by x1 quantization in the quantization parameters.</td>
        <td>For the overall constraints, see <a href="#constraints" target="_blank">Constraints</a>.<br/>This parameter is required for both supported modes, MX and HIFLOAT8 T-T, and cannot be nullptr.</td>
        <td>FLOAT8_E8M0, FLOAT32</td>
        <td>ND</td>
        <td>3, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale (const aclTensor*)</td>
        <td>Input</td>
        <td>Indicates the scaling factor introduced by x2 quantization in the quantization parameters.</td>
        <td>For details about the constraints, see <a href="#constraints" target="_blank">Constraints</a>.</td>
        <td>FLOAT8_E8M0, FLOAT32</td>
        <td>ND</td>
        <td>3, 1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>The accumulation input and the output updated in place, corresponding to Y_in and Y_out in the formulas.</td>
        <td>If x1 is an empty tensor with m=0 or x2 is an empty tensor with n=0, the output is an empty tensor.<br/>A contiguous tensor is required.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1 (bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x1 is transposed.</td>
        <td>Only true is currently supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2 (bool)</td>
        <td>Input</td>
        <td>Indicates whether the input shape of x2 is transposed.</td>
        <td>Only false is currently supported.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize (int64_t)</td>
        <td>Input</td>
        <td>Integer parameter that specifies the quantization group sizes along the m, n, and k dimensions.</td>
        <td>Consists of the three concatenated values groupSizeM, groupSizeN, and groupSizeK, each occupying 16 bits and together occupying the lower 48 bits of the int64_t groupSize (the upper 16 bits are invalid). The formula is shown below the table.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to allocate on the device, in bytes.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  - Formula: <a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return/Status Code</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>If the input parameter is a mandatory input, output, or mandatory attribute and is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>The data types or formats of x1, x2, x1ScaleOptional, x2Scale, or yRef, or the value of groupSize, are outside the supported ranges.</li><li>The shapes of x1, x2, x1ScaleOptional, x2Scale, or yRef fail validation.</li><li>The K dimension of x1 or x2 is 0.</li><li>When x1 and x2 are FLOAT4_E2M1, M or N is odd; or, when M, N, and K are all positive, M × K ≤ 2 or N × K ≤ 2.</li><li>groupSize fails validation, or groupSize is 0 and cannot be inferred from the shape relationships between x1/x2 and x1ScaleOptional/x2Scale.</li></ul></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>Failed to create the operator executor.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>An internal tensor operation returned a null pointer during tensor creation, preprocessing, operator computation, or result copying.</td>
      </tr>
    </tbody>
  </table>

## aclnnQuantBatchMatmulInplaceAdd

- **Parameters**
  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace (void*)</td>
        <td>Input</td>
        <td>The workspace memory address applied on the Device.</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t)</td>
        <td>Input</td>
        <td>Workspace size in bytes to allocate on the device, obtained from the first-stage aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize interface.</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor*)</td>
        <td>Input</td>
        <td>Op executor. Contains the operator computation process.</td>
      </tr>
      <tr>
        <td>stream (aclrtStream)</td>
        <td>Input</td>
        <td>Specifies the AscendCL stream for task execution.</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  Returns the aclnnStatus status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  ACLNN_ERR_INNER (561000) is returned if executor execution fails.

## Constraints

- Deterministic computation: aclnnQuantBatchMatmulInplaceAdd defaults to a deterministic implementation.
- Only transposeX1=true and transposeX2=false are currently supported. M and N must be nonnegative integers, and K must be a positive integer.
- Constraints on groupSize:
  - groupSize is decomposed into groupSizeM, groupSizeN, and groupSizeK using the following formulas. If one or more components are 0, they are inferred from the shapes of x1, x2, x1ScaleOptional, and x2Scale before computation. For example, groupSizeM=0 requests inference of the group size along M: groupSizeM = m / scaleM, where m must be divisible by scaleM. Here, m is the M dimension of x1, and scaleM is the M dimension of x1ScaleOptional.
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
- Dynamic quantization (MX quantization) use case constraints:
  - The input and output support the following data type combinations:

    | x1 | x2 | x1ScaleOptional | x2Scale | yRef |
    | :-------: | :-------: | :------- | :------ | :------ |
    | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |

  - Relationships among the data types of x1 and x2, the shapes of x1, x2, x1ScaleOptional, and x2Scale, and groupSize:

      | x1 data type | x2 data type | x1 shape | x2 shape | x1ScaleOptional Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      | :-------: | :-------: | :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 4295032864 |
      | FLOAT4_E2M1 | FLOAT4_E2M1 | (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 4295032864 |

  - Note: In the table above, gsM, gsK, and gsN denote groupSizeM, groupSizeK, and groupSizeN respectively. Dimensions with gsM, gsK, or gsN set to 0 are automatically inferred. The table above lists the case where automatic inference is not used.
  - Additional constraints for FLOAT4_E2M1:
    - The M dimension of x1 and the N dimension of x2 must be even.
    - When M, N, and K are all positive, both M × K > 2 and N × K > 2 must hold.

- HIFLOAT8 T-T use case constraints:
  - The input and output support the following data type combinations:

    | x1 | x2 | x1ScaleOptional | x2Scale | yRef |
    | :-------: | :-------: | :------- | :------ | :------ |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 |

  - Relationships among the data types of x1 and x2, the shapes of x1, x2, x1ScaleOptional, and x2Scale, and groupSize:

      | x1 data type | x2 data type | x1 shape | x2 shape | x1ScaleOptional Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      | :-------: | :-------: | :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      | HIFLOAT8 | HIFLOAT8 | (k, m) | (k, n) | (1) | (1) | (m, n) | [0, 0, 0] | 0 |

## Example

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>: The sample code for the mxFP8 quantization use case is as follows, where `x1/x2` is mxFP8, `x1Scale/x2Scale` is FLOAT8_E8M0, and `groupSize=32`. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

  ```cpp
  #include <cstdint>
  #include <cstdio>
  #include <iostream>
  #include <vector>
  #include <memory>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate device memory.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy host data to device memory.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call the aclCreateTensor API to create an aclTensor.
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  template <typename T1, typename T2>
  auto CeilDiv(T1 a, T2 b) -> T1
  {
      if (b == 0) {
          return a;
      }
      return (a + b - 1) / b;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnQuantBatchMatmulInplaceAddTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct inputs and outputs, which need to be custom constructed based on the API.
      int64_t M = 8;
      int64_t K = 16;
      int64_t N = 8;

      std::vector<int64_t> x1Shape = {K, M};
      std::vector<int64_t> x2Shape = {K, N};
      std::vector<int64_t> x2ScaleShape = {CeilDiv(K, 64), N, 2};
      std::vector<int64_t> yInputShape = {M, N};
      std::vector<int64_t> x1ScaleShape = {CeilDiv(K, 64), M, 2};
      std::vector<int64_t> yOutShape = {M, N};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yInputDeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* yOutputDeviceAddr = nullptr;

      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yInput = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* yOutput = nullptr;

      std::vector<uint8_t> x1HostData(M * K, 1); // Sample fill value.
      std::vector<uint8_t> x2HostData(N * K, 1); // Sample fill value.
      std::vector<uint8_t> x2ScaleHostData(CeilDiv(K, 64) * N * 2, 1);
      std::vector<float> yInputHostData(M * N, 1);                        // 1.0 in fp32
      std::vector<uint8_t> x1ScaleHostData(M * CeilDiv(K, 64) * 2, 1);
      std::vector<float> yOutputHostData(M * N, 1);

      // Create the x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the yInput aclTensor.
      ret = CreateAclTensor(yInputHostData, yInputShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT, &yInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yInputTensorPtr(yInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yInputDeviceAddrPtr(yInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Create the x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      bool transposeX1 = true;
      bool transposeX2 = false;
      int64_t groupSize = 32;

      // 3. Call the CANN operator library API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      void* workspaceAddr = nullptr;

      // Call the first-phase API of aclnnQuantBatchMatmulInplaceAdd.
      ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(x1, x2, x1Scale, x2Scale, yInput, transposeX1, transposeX2, groupSize, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on the workspaceSize computed by the first-phase API.
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // Call the second-phase API of aclnnQuantBatchMatmulInplaceAdd.
      ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      // 4. (boilerplate) Synchronize and wait for the task execution to complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the Device side to the Host side.
      auto size = GetShapeSize(yInputShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yInputDeviceAddr,
                       size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (boilerplate) Initialize the device/stream. For details, see the acl API reference.
      // Set deviceId based on the actual device.
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = AclnnQuantBatchMatmulInplaceAddTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT products</term>: The following example demonstrates HIFLOAT8 TT quantization with the supported combination `transposeX1=true`, `transposeX2=false`, HIFLOAT8 `x1/x2`, FLOAT32 `x1ScaleOptional/x2Scale`, and `groupSize=0`. For compilation and execution instructions, see [Compiling and Running Examples](../../../docs/en/context/compile_and_run_sample.md).

  ```cpp
  /**
   * Copyright (c) 2026 Huawei Technologies Co., Ltd.
   * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
   * CANN Open Software License Agreement Version 2.0 (the "License").
   * Please refer to the License for details. You may not use this file except in compliance with the License.
   * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   * See LICENSE in the root of the software repository for the full text of the License.
   */

  /*!
  * \file test_aclnn_quant_batch_matmul_inplace_add_TT.cpp
  * \brief
  */

  #include <cstdint>
  #include <cstdio>
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnQuantBatchMatmulInplaceAddTTDemo(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      int64_t m = 8;
      int64_t k = 16;
      int64_t n = 8;

      // The supported call combination is transposeX1=true, transposeX2=false,
      // so the shapes of x1/x2 need to be constructed as (k, m) and (k, n), respectively.
      std::vector<int64_t> x1Shape = {k, m};
      std::vector<int64_t> x2Shape = {k, n};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> yRefShape = {m, n};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yRefDeviceAddr = nullptr;

      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yRef = nullptr;

      std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape), 0b1000); // 0b1000 indicates 1.0 in hifloat8.
      std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape), 0b1000);
      std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1.0f);
      std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1.0f);
      std::vector<float> yRefHostData(GetShapeSize(yRefShape), 1.0f);

      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(yRefHostData, yRefShape, &yRefDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yRefTensorPtr(yRef, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yRefDeviceAddrPtr(yRefDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      bool transposeX1 = true;
      bool transposeX2 = false;
      int64_t groupSize = 0;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

      ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, yRef, transposeX1, transposeX2, groupSize, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }

      ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      std::vector<float> resultData(GetShapeSize(yRefShape), 0.0f);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yRefDeviceAddr,
          resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

      for (size_t i = 0; i < resultData.size(); ++i) {
          LOG_PRINT("result[%zu] is: %f\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = AclnnQuantBatchMatmulInplaceAddTTDemo(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTTDemo failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id8 -->
