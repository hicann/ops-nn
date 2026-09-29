# aclnnFusedQuantMatmul

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Function: fuses the quantized matrix multiplication and GELU computation.
- Formula:

  - With x1Scale, bias (INT32) (no offset in this scenario):

    $$
    qbmmout = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - With x1Scale, bias BFLOAT16/FLOAT16/FLOAT32 (no offset in this scenario):

    $$
    qbmmout = x1@x2 * x2scale * x1Scale + bias
    $$

  - With x1Scale, no bias:

    $$
    qbmmout = x1@x2 * x2Scale * x1Scale
    $$

  - The OP type is defined by the input fusedOpType. The following types are supported:

    - gelu_tanh operation:

      $$
      out = gelu\_tanh(qbmmout)
      $$

    - gelu_erf operation:

      $$
      out = gelu\_erf(qbmmout)
      $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. You must call aclnnFusedQuantMatmulGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnFusedQuantMatmul to perform the computation.

```c++
aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScaleOptional,
  const aclTensor *x1OffsetOptional,
  const aclTensor *x2OffsetOptional,
  const aclTensor *yOffsetOptional,
  const aclTensor *biasOptional,
  const aclTensor *x3Optional,
  const char      *fusedOpType,
  int64_t          groupSizeOptional,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```c++
aclnnStatus aclnnFusedQuantMatmul(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnFusedQuantMatmulGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 397px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 138px">
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
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td> 
          <ul>
            <li><a href="../../../docs/en/context/non_contiguous_tensor.md">Non-contiguous tensors</a> are supported only when the last m and k axes are transposed. In other scenarios, non-contiguous tensors are not supported.</li>
            <li>The size of the last dimension cannot exceed 65535.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>
          <ul>
            <li>In ND format, non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, <a href="../../../docs/en/context/non_contiguous_tensor.md">non-contiguous tensors</a> are not supported.</li>
            <li>The size of the last dimension cannot exceed 65535.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1Scale</td>
        <td>Input</td>
        <td>Quantization parameter, which is the input x1Scale in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Quantization parameter, corresponding to the input x2Scale in the formula.</td>
        <td>-</td>
        <td>FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScaleOptional</td>
        <td>Input</td>
        <td>Quantization scale parameter of the output y, used for static quantization.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1OffsetOptional</td>
        <td>Input</td>
        <td>Input x1Offset in the formula.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2OffsetOptional</td>
        <td>Input</td>
        <td>Input x2Offset in the formula.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffsetOptional</td>
        <td>Input</td>
        <td>Input yOffset in the formula.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>Input</td>
        <td>Input bias in the formula.</td>
        <td>If there is no bias, set this parameter to nullptr.</td>
        <td>INT32, FLOAT32, BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>1, 3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x3Optional</td>
        <td>Input</td>
        <td>Input of the fusion binary operation.</td>
        <td>Currently, only the fusion unary operation is supported. This parameter is reserved and is not supported in the current version. It needs to be set to nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>Input</td>
        <td>The input fusedOpType in the formula indicates the fusion mode supported by the QuantBatchMatmul operator.</td>
        <td>The fusion mode must be either "gelu_erf" or "gelu_tanh".</td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSizeOptional</td>
        <td>Input</td>
        <td>Quantization group size in the m, n, and k directions.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>-</td>
        <td>FLOAT16, BFLOAT16</td>
        <td>ND</td>
        <td>2-6</td>
        <td>✓</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1083px"><colgroup>
  <col style="width: 251px">
  <col style="width: 129px">
  <col style="width: 703px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="1">161001</td>
      <td>The input x1, x2, x1Scale, x2Scale, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The shape of x1, x2, biasOptional, x1Scale, x2Scale, or out does not meet the verification conditions.</td>
    </tr>
    <tr>
      <td>The data type and format of x1, x2, biasOptional, x1Scale, x2Scale, or out are not supported.</td>
    </tr>
    <tr>
      <td>The input x1, x2, biasOptional, x1Scale, x2Scale, or out is an empty tensor.</td>
    </tr>
    <tr>
      <td>The input fusedOpType is not one of "gelu_tanh" and "gelu_erf".</td>
    </tr>
    <tr>
      <td>The input parameters yScaleOptional, x1OffsetOptional, x2OffsetOptional, yOffsetOptional, x3Optional, and groupSizeOptional are not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnFusedQuantMatmul

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
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
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnFusedQuantMatmulGetWorkspaceSize.</td>
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
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>: The default deterministic implementation of aclnnFusedQuantMatmul is used.

- The following table describes the supported input and output data type combinations.

  | x1                        | x2                        | x1Scale     | x2Scale         | x2OffsetOptional    | yScaleOptional   | biasOptional         | yOffsetOptional    | out                                    |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT8                      | FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  
- Currently, the API supports x1 per-token quantization and x2 per-channel/per-tensor quantization. The input dtype combinations of x1, x2, x1Scale, and x2Scale supported by different quantization modes are as follows:
  - The data type of **x1** can be INT8, INT32, or INT4.
    - When the data type is INT32 or INT4, the INT4 quantization scenario is as follows:
      - Currently, only the ND format is supported.
      - Currently, only non-transposed inputs are supported.
      - The inner axis of x1 must be an even number.
    - When the data type is INT32, each INT32 data entry stores eight INT4 data entries, with shape (batch, m, k // 8), where **k** must be a multiple of 8.
  - The data type of **x2** can be INT8, INT32, or INT4.
    - When the data type is INT32 or INT4, the INT4 quantization scenario is as follows:
      - This API supports only the 2D ND format.
      - The inner axis of x2 must be an even number.
    - When the data type is INT32, eight INT4 data elements are stored in each INT32 data element.
    - The **aclnnConvertWeightToINT4Pack** API can be used to convert **x2** from INT32 (one int32 space stores one int4 data entry in bits 0–3) to INT32 (one int32 space stores eight int4 data entries) or INT4 (one int4 space stores one int4 data entry). For details, see [aclnnConvertWeightToINT4Pack](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md).
  - The constraints for **x1Scale** are as follows:
    - The shape supports one dimension, which is (m,). The data type can be FLOAT32.
  - The constraints for **x2Scale** are as follows:
    - The shape supports one dimension, which is (n,) or (1,). The value of n is the same as that of x2. The data type can be FLOAT32 or BFLOAT16.
  - The restrictions on biasOptional are as follows:
    - The shape supports one or three dimensions. In the INT4 quantization scenario, biasOptional supports only one dimension, and the shape is (n). In the three-dimensional scenario, the shape of biasOptional is (batch, 1, n).
    - The data type can be int32, float32, bfloat16, or float16.
  - The constraints for **out** are as follows:
    - The shape supports 2 to 6 dimensions, which is (batch, m, n). The data type can be FLOAT16 or BFLOAT16.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

- <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>:
x1 is of type INT8, x2 is of type INT8, x1Scale is of type FLOAT32, and x2Scale is of type FLOAT32.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_fused_quant_matmul.h"

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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
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
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnFusedQuantMatmulTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the inputs and outputs based on the API definition.
      std::vector<int64_t> x1Shape = {2, 8192};   // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128}; // (k,n)
      std::vector<int64_t> x1ScaleShape = {2}; // x1ScaleShape = [M]
      std::vector<int64_t> x2ScaleShape = {128}; // x2ScaleShape = [N]
      std::vector<int64_t> outShape = {2, 128};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int8_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      int64_t groupSize = 0;
      const char fusedOpType[] = "gelu_tanh";

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;

      ret = aclnnFusedQuantMatmulGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, fusedOpType, groupSize, out,
          &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second API of aclnnFusedQuantMatmul.
      ret = aclnnFusedQuantMatmul(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmul failed. ERROR: %d\n", ret); return ret);

      // 4. (Boilerplate) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0); // In the C language, fp16 data cannot be directly printed. You need to read the data using uint16 and convert it to fp16 using the binary format.
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnFusedQuantMatmulTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedQuantMatmulTest failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
  