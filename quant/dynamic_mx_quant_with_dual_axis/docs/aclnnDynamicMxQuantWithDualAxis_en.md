# aclnnDynamicMxQuantWithDualAxis

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_mx_quant_with_dual_axis)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- Function: performs MX quantization on the -1 and -2 axes, with the destination data type being FLOAT4 or FLOAT8. On the given -1 and -2 axes, the quantization scales mxscale1 and mxscale2 corresponding to the two groups of numbers are calculated every 32 numbers, and are used as the corresponding parts of the output mxscale1Out and mxscale2Out. Then, all elements in the two groups of numbers are divided by the corresponding mxscale1 or mxscale2, and converted to the corresponding dstType based on the round_mode to obtain the quantization results y1 and y2, which are used as the corresponding parts of the output y1Out and y2Out.

- Formulas:
  - Currently, only scaleAlg=0 is supported, that is, the OCP implementation is as follows:
  - The input x is grouped into 32 numbers along the -1 axis. A group of 32 numbers $\{\{V_i\}_{i=1}^{32}\}$ is quantized into $\{mxscale1, \{P_i\}_{i=1}^{32}\}$.

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - In addition, the input x is grouped into 32 groups along the -2 axis, and a group of 32 numbers $\{\{V_j\}_{j=1}^{32}\}$ is quantized into $\{mxscale2, \{P_j\}_{j=1}^{32}\}$.

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - -The quantized $P_{i}$ on axis 1 is used to form the output y1Out according to the position of the corresponding $V_{i}$, and the mxscale1 is used to form the output mxscale1Out according to the group on the corresponding -1 axis. The quantized $P_{j}$ on axis -2 is used to form the output y2Out according to the position of the corresponding $V_{j}$, and the mxscale2 is used to form the output mxscale2Out according to the group on the corresponding -2 axis.

  - emax: exponent bit of the maximum regular number of the corresponding data type.

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnDynamicMxQuantWithDualAxis to perform the computation.

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          scaleAlg, 
  const aclTensor *y1Out, 
  const aclTensor *mxscale1Out, 
  const aclTensor *y2Out, 
  const aclTensor *mxscale2Out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxis(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Indicates the input x, which corresponds to V<sub>i</sub> in the formula.</td>
      <td>When the target type is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of x must be an even number. Empty tensors are not supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>Input</td>
      <td>Indicates the data conversion mode, which corresponds to round_mode in the formula.</td>
      <td><ul><li>When dstType is 40 or 41, and the data types of the output y1Out and y2Out are FLOAT4_E2M1/FLOAT4_E1M2, {"rint", "floor", "round"} is supported.</li><li>When dstType is 35 or 36, and the data types of the output y1Out and y2Out are FLOAT8_E5M2/FLOAT8_E4M3FN, only {"rint"} is supported.</li><li>If a null pointer is passed, the rint mode is used.</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>Input</td>
      <td>Indicates the types of y1Out and y2Out after data conversion.</td>
      <td>The input value range is {35, 36, 40, 41}, which corresponds to the data types of y1Out and y2Out being {35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}</td>.
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>Input</td>
      <td>Indicates the calculation method of mxscale1Out and mxscale2Out.</td>
      <td>Currently, only the value 0 is supported, indicating the OCP implementation.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of the input x quantized along axis -1, corresponding to P<sub>i</sub> in the formula.</td>
      <td>The shape is the same as that of the input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group on axis -1, corresponding to mxscale1 in the formula.</td>
      <td>The value of axis -1 of shape x is rounded up by 32 and then padded with an even number. The padding value is 0.</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the result of quantizing axis -2 of the input x, which corresponds to P<sub>j</sub> in the formula.</td>
      <td>The shape is the same as that of the input x.</td>
      <td>FLOAT4_E2M1, FLOAT4_E1M2, FLOAT8_E4M3FN, FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>Output</td>
      <td>Indicates the quantization scale corresponding to each group of axis -2, which corresponds to mxscale2 in the formula.</td>
      <td><ul><li>The value of axis -2 of shape x is rounded up by 32 and then padded with an even number. The padding value is 0. </li><li>The mxscale2Out output needs to be interleaved for every two rows of data.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
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

- **Return Value**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The pointer x is null.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, and mxscale2Out are not supported.</td>
    </tr>
    <tr>
      <td>The shape of x, y1Out, y2Out, mxscale1Out, or mxscale2Out does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>The values of roundModeOptional, dstType, and scaleAlg are not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not supported.</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuantWithDualAxis

- **Parameters**
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, which is obtained by the first API aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

 - The restrictions on the shapes of x, mxscale1Out, and mxscale2Out are as follows:
    - rank(mxscale1Out) = rank(x) + 1.
    - rank(mxscale2Out) = rank(x) + 1.
    - mxscale1Out.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2.
    - mxscale2Out.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2.
    - mxscale1Out.shape[-1] = 2.
    - mxscale2Out.shape[-1] = 2.
    - Other dimensions are the same as those of the input x.
    - For example, if the shape of the input x is [B, M, N] and the destination data type is FP8, the shapes of y1 and y2 are [B, M, N], the shape of mxscale1 is [B, M, (ceil(N/32)+2-1)/2, 2], and the shape of mxscale2 is [B, (ceil(M/32)+2-1)/2, N, 2].
 - Deterministic description: The default deterministic implementation of aclnnDynamicMxQuantWithDualAxis is used.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_mx_quant_with_dual_axis.h"

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

      int64_t
      GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
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
                      aclDataType dataType, aclTensor** tensor)
  {
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnDynamicMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs based on the API definition.
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> y1OutShape = {1, 4};
      std::vector<int64_t> y2OutShape = {1, 4};
      std::vector<int64_t> mxscale1OutShape = {1, 1, 2};
      std::vector<int64_t> mxscale2OutShape = {1, 4, 2};
      void* xDeviceAddr = nullptr;
      void* y1OutDeviceAddr = nullptr;
      void* mxscale1OutDeviceAddr = nullptr;
      void* y2OutDeviceAddr = nullptr;
      void* mxscale2OutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* y1Out = nullptr;
      aclTensor* mxscale1Out = nullptr;
      aclTensor* y2Out = nullptr;
      aclTensor* mxscale2Out = nullptr;
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      std::vector<uint8_t> y1OutHostData = {0, 72, 96, 120};
      std::vector<uint8_t> y2OutHostData = {0, 0, 0, 0};
      std::vector<uint8_t> mxscale1OutHostData = {128, 0};
      std::vector<uint8_t> mxscale2OutHostData = {0, 0, 122, 0, 125, 0, 128, 0};
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t scaleAlg = 0;
      // Create an x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y1Out aclTensor.
      ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mxscale1Out aclTensor.
      ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the y2Out aclTensor.
      ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the mxscale2Out aclTensor.
      ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // Call the first API of aclnnDynamicMxQuantWithDualAxis.
      ret = aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second API call of aclnnDynamicMxQuantWithDualAxis.
      ret = aclnnDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

      // (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
      auto size1 = GetShapeSize(y1OutShape);
      auto size2 = GetShapeSize(y2OutShape);
      std::vector<uint8_t> y1OutData(
          size1, 0); // The fp4 data cannot be directly printed in C language. You need to read the data using uint8 and convert it to fp4 in binary mode.
      std::vector<uint8_t> y2OutData(
          size2, 0); // The fp4 data cannot be directly printed in C language. You need to read the data using uint8 and convert it to fp4 in binary mode.
      ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(y1OutData[0]), y1OutDeviceAddr,
                        size1 * sizeof(y1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST) && 
                        aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(y2OutData[0]), y2OutDeviceAddr,
                        size2 * sizeof(y2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out and y2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("y1Out[%ld] is: %d\n", i, y1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("y2Out[%ld] is: %d\n", i, y2OutData[i]);
      }
      size1 = GetShapeSize(mxscale1OutShape);
      size2 = GetShapeSize(mxscale2OutShape);
      std::vector<uint8_t> mxscale1OutData(
          size1, 0); // In C language, fp8 data cannot be directly printed. You need to read the data using uint8 and convert it to fp8 in binary mode.
      std::vector<uint8_t> mxscale2OutData(
          size2, 0); // In C language, fp8 data cannot be directly printed. You need to read the data using uint8 and convert it to fp8 in binary mode.
      ret = aclrtMemcpy(mxscale1OutData.data(), mxscale1OutData.size() * sizeof(mxscale1OutData[0]), mxscale1OutDeviceAddr,
                        size1 * sizeof(mxscale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(mxscale2OutData.data(), mxscale2OutData.size() * sizeof(mxscale2OutData[0]), mxscale2OutDeviceAddr,
                        size2 * sizeof(mxscale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("mxscale1Out[%ld] is: %d\n", i, mxscale1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("mxscale2Out[%ld] is: %d\n", i, mxscale2OutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicMxQuantWithDualAxisTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
```
