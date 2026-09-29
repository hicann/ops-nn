# aclnnDynamicDualLevelMxQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_dual_level_mx_quant)

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

- API description: performs MX quantization with the destination data type being FLOAT4. Only the last axis is quantized. All the previous axes are fused together. The input is divided into multiple data blocks based on the given level0BlockSize, and level-1 quantization is performed on each data block to output the quantization scale level0ScaleOut. Then, the level-1 quantization result is used as the new input, and is divided into multiple data blocks based on the given level1BlockSize. Level-2 quantization is performed on each data block to output the quantization scale level1ScaleOut. The data type is converted based on the round_mode to obtain the quantization result yOut. For details, see the figure (see../figures/DynamicDualLevelMxQuant.png).

- Formulas:
  - The input x is grouped into $k_0$ = level0BlockSize groups along the last axis. A group of $k_0$ elements $\{\{x_i\}_{i=1}^{k_0}\}$ is dynamically quantized into $\{level0Scale, \{temp_i\}_{i=1}^{k_0}\}$, where $k_0$ = level0BlockSize. Then, the temp is grouped into $k_1$ = level1BlockSize groups along the last axis. A group of $k_1$ elements $\{\{temp_i\}_{i=1}^{k_1}\}$ is dynamically quantized into $\{level1Scale, \{y_i\}_{i=1}^{k_1}\}$, where $k_1$ = level1BlockSize.

  $$
  input\_max_i = max_i(abs(x_i))
  $$

  $$
  level0Scale = input\_max_i / (FP4\_E2M1\_MAX)
  $$

  $$
  temp_i = cast\_to\_x\_type(x_i / level0Scale), \space i\space from\space 1\space to\space level0BlockSize
  $$

  $$
  shared\_exp = floor(log_2(max_i(|temp_i|))) - emax
  $$

  $$
  level1Scale = 2^{shared\_exp}
  $$

  $$
  y_i = cast\_to\_FP4\_E2M1(temp_i/level1Scale, round\_mode), \space i\space from\space 1\space to\space level1BlockSize
  $$

  - ​The quantized $y_{i}$ forms the output yOut based on the corresponding positions of $x_{i}$. The level0Scale forms the output level0ScaleOut based on the corresponding groups of the last axis. The level1Scale forms the output level1ScaleOut based on the corresponding groups of the last axis.

  - max_i indicates the maximum value in the ith group.

  - emax: exponent bit of the maximum regular number of the corresponding data type.

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnDynamicDualLevelMxQuantGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnDynamicDualLevelMxQuant to perform the computation.

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *smoothScaleOptional, 
  char            *roundModeOptional, 
  int64_t          level0BlockSize, 
  int64_t          level1BlockSize, 
  const aclTensor *yOut, 
  const aclTensor *level0ScaleOut, 
  const aclTensor *level1ScaleOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicDualLevelMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicDualLevelMxQuantGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
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
      <td>Input x, corresponding to <em>x</em><sub>i</sub> in the formula.</td>
      <td><ul><li>The last dimension of x must be an even number.</li><li>An empty tensor is not supported.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Optional input smoothScaleOptional.</td>
      <td>Currently, this function is not supported. Only nullptr can be input.</td>
      <td>Same as the input x.</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*) </td>
      <td>Input</td>
      <td>Quantization mode, corresponding to round_mode in the formula.</td>
      <td><ul><li>{"rint", "round", "floor"} are supported.</li><li>The default value is "rint".</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level0BlockSize (int64_t)</td>
      <td>Input</td>
      <td>Block size for level-1 quantization, corresponding to level0BlockSize in the formula.</td>
      <td>The value range is {512}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>level1BlockSize (int64_t)</td>
      <td>Input</td>
      <td>Block size for level-2 quantization, corresponding to level1BlockSize in the formula.</td>
      <td>The value range is {32}.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>Quantized result of the input x, corresponding to <em>y</em><sub>i</sub> in the formula.</td>
      <td><ul><li>The shape is the same as that of the input x.</li><li>An empty tensor is not supported.</li></ul></td>
      <td>FLOAT4_E2M1</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level0ScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale of level-1 quantization, corresponding to level0Scale in the formula.</td>
      <td><ul><li>The value of the last axis is the value of the last axis of x divided by level0BlockSize and rounded up.</li><li>An empty tensor is not supported.</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>level1ScaleOut (aclTensor*)</td>
      <td>Output</td>
      <td>Scale of level-2 quantization, corresponding to level1Scale in the formula.</td>
      <td><ul><li>The size of the shape is x + 1.</li><li>The values of the last two axes of the shape are ((ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2, 2), and even padding is performed on the values. The padding value is 0.</li><li>Empty tensors are not supported.</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
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
  </tbody>
  </table>

- **Returns**

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
      <td>Pointer x is null.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data types and formats of x, smoothScaleOptional, yOut, level0ScaleOut, and level1ScaleOut are not supported.</td>
    </tr>
    <tr>
      <td>The shape of x, smoothScaleOptional, yOut, level0ScaleOut, or level1ScaleOut does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>The values of roundModeOptional, level0BlockSize, and level1BlockSize are not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>The current platform is not supported.</td>
    </tr>
  </tbody></table>

## aclnnDynamicDualLevelMxQuant

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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnDynamicDualLevelMxQuantGetWorkspaceSize.</td>
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

- The restrictions on the shapes of x, level0ScaleOut, and level1ScaleOut are as follows:
    - rank(level1ScaleOut) = rank(x) + 1.
    - level0ScaleOut.shape[-1] = ceil(x.shape[-1] / level0Blocksize).
    - level1ScaleOut.shape[-2] = (ceil(x.shape[-1] / level1Blocksize) + 2 - 1) / 2.
    - level1ScaleOut.shape[-1] = 2.
    - The shapes of other dimensions are consistent with those of input x.
- Deterministic description: The aclnnDynamicDualLevelMxQuant is implemented in deterministic mode by default.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_dual_level_mx_quant.h"
  
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
  
  int aclnnDynamicDualLevelMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. Construct the inputs and outputs based on the API definition.
      std::vector<int64_t> xShape = {1, 512};
      std::vector<int64_t> smoothScaleOptionalShape = {1};
      std::vector<int64_t> yOutShape = {1, 512};
      std::vector<int64_t> level0ScaleOutShape = {1, 1};
      std::vector<int64_t> level1ScaleOutShape = {1, 8, 2};
      void* xDeviceAddr = nullptr;
      void* smoothScaleOptionalDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* level0ScaleOutDeviceAddr = nullptr;
      void* level1ScaleOutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* smoothScaleOptional = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* level0ScaleOut = nullptr;
      aclTensor* level1ScaleOut = nullptr;

      // Value corresponding to BF16 (0 -> 0, 16640 -> 8, 17024 -> 64, 17408 -> 512)
      std::vector<uint16_t> xHostData(512, 16640);
      std::vector<uint16_t> smoothScaleOptionalHostData = {0};
      // Value corresponding to float4_e2m1 (0 -> 0, 72 -> 4, 96 -> 32, 120 -> 256)
      std::vector<uint8_t> yOutHostData(512, 0);
      // Value corresponding to float32 (0 -> 0)
      std::vector<float> level0ScaleOutHostData = {{0}};
      // Value corresponding to float8_e8m0 (128 -> 2)
      std::vector<std::vector<std::vector<uint8_t>>> level1ScaleOutHostData(1, std::vector<std::vector<uint8_t>>(8, std::vector<uint8_t>(2, 0)));
      const char* roundModeOptional = "rint";
      int64_t level0Blocksize = 512;
      int64_t level1Blocksize = 32;

      // Create an x aclTensor.
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create smoothScaleOptional aclTensor.
      ret = CreateAclTensor(smoothScaleOptionalHostData, smoothScaleOptionalShape, &smoothScaleOptionalDeviceAddr, aclDataType::ACL_BF16, &smoothScaleOptional);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> smoothScaleOptionalTensorPtr(smoothScaleOptional, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> smoothScaleOptionalDeviceAddrPtr(smoothScaleOptionalDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the yOut aclTensor.
      ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &yOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the level0ScaleOut aclTensor.
      ret = CreateAclTensor(level0ScaleOutHostData, level0ScaleOutShape, &level0ScaleOutDeviceAddr, aclDataType::ACL_FLOAT, &level0ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level0ScaleOutTensorPtr(level0ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level0ScaleOutDeviceAddrPtr(level0ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create level1ScaleOut aclTensor.
      ret = CreateAclTensor(level1ScaleOutHostData, level1ScaleOutShape, &level1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &level1ScaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> level1ScaleOutTensorPtr(level1ScaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> level1ScaleOutDeviceAddrPtr(level1ScaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
     
      // Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
   
      // Call the first API of aclnnDynamicDualLevelMxQuant.
      ret = aclnnDynamicDualLevelMxQuantGetWorkspaceSize(x, smoothScaleOptional, (char*)roundModeOptional, level0Blocksize, level1Blocksize, yOut, level0ScaleOut, level1ScaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second API of aclnnDynamicDualLevelMxQuant.
      ret = aclnnDynamicDualLevelMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuant failed. ERROR: %d\n", ret); return ret);
  
      // (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(yOutShape) / 2;
      std::vector<uint8_t> yOutData(
          size, 0); // In the C language, the FP4 data cannot be directly printed. You need to use uint8 to read the data and convert it to FP4 in binary mode.
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      size = GetShapeSize(level0ScaleOutShape);
      std::vector<float> level0ScaleOutData(
          size, 0);
      ret = aclrtMemcpy(level0ScaleOutData.data(), level0ScaleOutData.size() * sizeof(level0ScaleOutData[0]), level0ScaleOutDeviceAddr,
                        size * sizeof(level0ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level0ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level0ScaleOut[%ld] is: %f\n", i, level0ScaleOutData[i]);
      }
      size = GetShapeSize(level1ScaleOutShape);
      std::vector<uint8_t> level1ScaleOutData(
          size, 0); // In C language, the fp8 data cannot be directly printed. You need to read the data using uint8 and convert it to fp8 through binary conversion.
      ret = aclrtMemcpy(level1ScaleOutData.data(), level1ScaleOutData.size() * sizeof(level1ScaleOutData[0]), level1ScaleOutDeviceAddr,
                        size * sizeof(level1ScaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy level1ScaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("level1ScaleOut[%ld] is: %d\n", i, level1ScaleOutData[i]);
      }
      return ACL_SUCCESS;
  }
  
  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicDualLevelMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicDualLevelMxQuantTest failed. ERROR: %d\n", ret); return ret);
  
      Finalize(deviceId, stream);
      return 0;
  }
```
