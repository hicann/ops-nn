# aclnnQuantMatmulV3

**Note: This API will be deprecated in later versions. Use the latest aclnnQuantMatmulV5 API instead.**

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    √    |
| <term>Atlas training products</term>                             |    ×    |

## Function

- Description: Performs quantized matrix multiplication, supporting at least two-dimensional input and at most six-dimensional input. Similar APIs include **aclnnMm** (only two-dimensional tensors can be used as the input of matrix multiplication) and **aclnnBatchMatMul** (only three-dimensional matrix multiplication is supported, whose first dimension is the **batch** dimension). It supports the T-C && T-T [quantization mode](../../../docs/en/context/quant_more_introduction.md).
- Formula:
  - No bias:

    $$
    out = x1@x2 * scale + offset
    $$

  - bias (INT32):

    $$
    out = (x1@x2 + bias) * scale + offset
    $$

  - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT:
    bias BFLOAT16/FLOAT32 is supported (no offset in this scenario).

    $$
    out = x1@x2 * scale + bias
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantMatmulV3GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantMatmulV3** is called to perform computation.

```cpp
aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *scale,
    const aclTensor *offset,
    const aclTensor *bias,
    bool             transposeX1,
    bool             transposeX2,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMatmulV3(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantMatmulV3GetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
    <col style="width: 130px">
    <col style="width: 97px">
    <col style="width: 308px">
    <col style="width: 488px">
    <col style="width: 197px">
    <col style="width: 77px">
    <col style="width: 115px">
    <col style="width: 95px">
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
        <td><ul>
        <li>Non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, non-contiguous tensors are not supported.</li>
        <li>When transposeX1 is set to false, the shape is (batch, m, k). When transposeX1 is set to true, the shape is (batch, k, m). The batch dimension may not exist.
        </li></ul>
        </td>
        <td>INT8, INT32, INT4</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td><ul>
        <li>ND:
        <ul><li>Non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, non-contiguous tensors are not supported.</li>
        <li>When transposeX1 is set to false, the shape is (batch, n, k). When transposeX1 is set to true, the shape is (batch, k, n). The batch dimension may not exist, and k is the same as that in the shape of x1.</li></ul></li>
        <li>NZ:
          <ul><li>When transposeX2 is true, the shape is (batch, k1, n1, n0, k0), where batch may not exist, k0 = 32, n0 = 16, and k in the shape of x1 and k1 in the shape of x2 must meet the ceil (k / 32) = k1</li></ul> condition.
          <ul><li>When transposeX2 is false, the shape is (batch, n1, k1, k0, n0), where batch may not exist, k0 = 16, n0 = 32, and k in the shape of x1 and k1 in the shape of x2 must meet the ceil (k / 16) = k1</li></ul> condition.
        </li></ul></td>
        <td>INT8, INT32, INT4</td>
        <td>ND and NZ</td>
        <td>2-8 (ND), 4-8 (NZ)
        </td>
        <td>√</td>
      </tr>
      <tr>
        <td>scale</td>
        <td>Input</td>
        <td>Quantization parameter, corresponding to the input scale in the formula.</td>
        <td>(t,), where t = 1 or n, and n is the same as that of x2.<br>If the original input type does not conform with the data type combinations described in [Constraints](#constraints), call the **aclnn** API of the TransQuantParamV2 operator to convert **scale** to the INT64 or UINT64 type in advance.</td>
        <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>offset</td>
        <td>Input</td>
        <td>Input offset in the formula.</td>
        <td>(t,), where t = 1 or n, and n is the same as that of x2.<br>If the output data type is INT8, the offset can exist. For other input types, nullptr needs to be passed.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>×</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Optional input</td>
        <td>Input bias in the formula.</td>
        <td>The shape can be 1D (n,) or 3D (batch, 1, n). The value of n is the same as that of x2.<br>When the output shape is 2, 4, 5, or 6D, the bias shape supports only 1D.</td>
        <td>INT32, BFLOAT16, FLOAT32, </td>
        <td>ND</td>
        <td>1, 3</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1</td>
        <td>Input</td>
        <td>Whether the input shape of x1 is transposed.</td>
        <td>When transposeX1 is false, the shape is (batch, m, k).<br>When transposeX1 is true, the shape is (batch, k, m).<br>The batch may not exist.</td>
        <td>bool</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>Input</td>
        <td>Whether the input shape of x2 is transposed.</td>
        <td>
        <ul>
        <li>ND: When transposeX2 is false, the shape is (batch, k, n).<br>When transposeX2 is true, the shape is (batch, n, k).<br>The batch dimension may not exist. k is the same as k in the shape of x1.</li>
        <li>NZ:
        <ul>
        <li>When transposeX2 is true, the shape is (batch, k1, n1, n0, k0). The batch dimension may not exist. k0 = 32, n0 = 16. The k in the shape of x1 and k1 in the shape of x2 must meet the following relationship: ceil (k / 32) = k1</li>
        <li>When transposeX2 is false, the shape is (batch, n1, k1, k0, n0). The batch dimension may not exist. k0 = 16, n0 = 32. The k in the shape of x1 and k1 in the shape of x2 must meet the following relationship: ceil (k / 16) = k1</li>
        </ul>
        </li>
        </ul>
        </td>
        <td>bool</td>
        <td>-</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>`out` in the formula.</td>
        <td>
        (batch, m, n). The batch dimension may not exist. Broadcasting between the batch dimensions of x1 and x2 is supported.<br>The output batch is the same as the broadcast batch, m is the same as the m of x1, and n is the same as the n of x2.</td>
        <td>FLOAT16, INT8, BFLOAT16, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>√</td>
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

  - <term>Atlas inference products</term>:
    - x1 and x2 support INT8.
    - scale supports UINT64 and INT64.
    - bias supports INT32.
    - out supports FLOAT16 and INT8.
  - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
    - x1 and x2 support INT8, INT32, and INT4.
    - scale supports UINT64, INT64, FLOAT32, and BFLOAT16.
    - bias supports INT32, BFLOAT16, and FLOAT32. When x1 and x2 are of type INT32 or INT4, the shape of bias supports only one dimension (n,).
    - When x1 and x2 are of type INT32 or INT4, transposeX1 can only be false.
    - out supports FLOAT16, INT8, BFLOAT16, and INT32.
  - Ascend 950PR/Ascend 950DT:
    - x1 and x2 support INT8.
    - scale supports UINT64, INT64, FLOAT32, and BFLOAT16.
    - scale supports INT32, BFLOAT16, and FLOAT32.
    - out supports FLOAT16, INT8, BFLOAT16, and INT32.
    - When x2 is in ND format, if x1 is an empty tensor with m = 0 or x2 is an empty tensor with n = 0, the output is an empty tensor. When x2 is in FRACTAL_NZ format, if x1 is an empty tensor with m = 0, the output is an empty tensor.

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first phase API verifies the input parameters. An error is reported in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 291px">
  <col style="width: 135px">
  <col style="width: 723px">
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
        <td>The input x1, x2, scale, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>The data type and format of x1, x2, bias, scale, offset, or out are not supported.</td>
      </tr>
      <tr>
        <td>The shape of x1, x2, bias, scale, offset, or out does not meet the verification conditions.</td>
      </tr>
      <tr>
        <td>x1, x2, bias, x2Scale, x2Offset, or out is an empty tensor.</td>
      </tr>
    </tbody>
    </table>

## aclnnQuantMatmulV3

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter Description</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnQuantMatmulV3GetWorkspaceSize API.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, including the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing a task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic description:
  - <term>Atlas training products</term> and <term>Atlas inference products</term>: **aclnnQuantMatmulV3** defaults to a deterministic implementation.
  - Ascend 950PR/Ascend 950DT: The aclnnQuantMatmulV3 function is implemented in a deterministic manner by default.

- <term>Atlas inference products</term>:
  - The size of the last dimension of x1 cannot exceed 65535. The last dimension of x1 refers to m when transposeX1 is true or k when transposeX1 is false.
  - The size of the last dimension of x2 cannot exceed 65535. The last dimension of x2 refers to k when transposeX2 is true or n when transposeX2 is false.
    - When x2 is in NZ format, transposeX2 cannot be false.
  - Before calling this API, you can use [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) to process x2 whose format is ND to obtain the NZ format.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  - The size of the last dimension of x1 cannot exceed 65535. The last dimension of x1 refers to m when transposeX1 is true or k when transposeX1 is false. When the data type of x1 is INT32 or INT4, the INT4 quantization scenario is used. Currently, only the scenario where transposeX1 is false is supported. When the data type of **x1** is INT4, the shape is represented by (batch, m, k), where **k** must be an even number. When the data type of **x1** is INT32, each INT32 data entry stores eight INT4 data entries, with shape represented by (batch, m, k // 8), where **k** must be a multiple of 8.
  - The size of the last dimension of x2 cannot exceed 65535. The last dimension of x2 refers to k when transposeX2 is true or n when transposeX2 is false. When x2 is in NZ format, transposeX2 cannot be false.
    - If the data type is INT4 and transposeX2 is true, the shape is (n, k), and k must be an even number. If transposeX2 is false, the shape is (k, n), and n must be an even number.
    - If the data type is INT32, eight INT4 data elements are stored in each INT32 data element. When transposeX2 is true, the shape is (n, k // 8), and k must be a multiple of 8. When transposeX2 is false, the shape is (k, n // 8), and n must be a multiple of 8.
    - The **aclnnConvertWeightToINT4Pack** API can be used to convert **x2** from INT32 (one int32 space stores one int4 data entry in bits 0–3) to INT32 (one int32 space stores eight int4 data entries) or INT4 (one int4 space stores one int4 data entry). For details, see aclnnConvertWeightToINT4Pack.
  - Before this API is called, you can call the [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) API to convert the format of x2 from ND to NZ.
  
- Ascend 950PR/Ascend 950DT:
  - When the last two axes are 1 (that is, n = 1 or k = 1), x2 does not support the proprietary format and supports only the ND format.
  - Before this API is called, you can call the [aclnnTransMatmulWeight](https://gitcode.com/cann/ops-math/blob/master/conversion/trans_data/docs/aclnnTransMatmulWeight.md) or [aclnnNpuFormatCast](https://gitcode.com/cann/ops-math/blob/master/conversion/npu_format_cast/docs/aclnnNpuFormatCast.md) API to convert the format of x2 from ND to NZ.
  - If the last two dimensions of the original ND shape are 1, you are not advised to convert the format to NZ. By default, x2 is discontinuous, and only discontinuous x2 tensors are supported.

The following data type combinations are supported for the input and output. The combinations support T-C && T-T [quantization mode](../../../docs/en/context/quant_more_introduction.md).

  > Note: If the original input type does not meet the following combinations, call the aclnn API of the TransQuantParamV2 operator to convert the scale to the INT64 or UINT64 data type.

- <term>Atlas inference products</term>:

  | x1 | x2 | scale | offset | bias | out |
  | ------- | ------- | ------ | ------ | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32  |  FLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32  |  INT8 |

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:

  | x1 | x2 | scale | offset | bias | out |
  | ------- | ------- | ------ | ------ | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32  |  FLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32  |  INT8 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32/BFLOAT16/FLOAT32  |  BFLOAT16 |
  | INT4/INT32 | INT4/INT32 | UINT64/INT64 | null | null/INT32  |  FLOAT16 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32  | INT32 |

- Ascend 950PR/Ascend 950DT:

  | x1 | x2 | scale | offset | bias | out |
  | ------- | ------- | ------ | ------ | ------- | ------- |
  | INT8 | INT8 | UINT64/INT64 | null | null/INT32  |  FLOAT16/BFLOAT16 |
  | INT8 | INT8 | UINT64/INT64 | null/FLOAT32 | null/INT32  |  INT8 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32/BFLOAT16/FLOAT32  |  BFLOAT16 |
  | INT8 | INT8 | FLOAT32/BFLOAT16 | null | null/INT32  | INT32 |

## Example

- For the <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT:
The following is a code example in general scenarios, which is for reference only. For details about the compilation and execution process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v3.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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

  int aclnnQuantMatmulV3Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Half-precision float16
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3GetWorkspaceSize(x1, x2, quantParam, nullptr, bias, transposeX1, transposeX2, out,
                                              &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV3.reset(workspaceAddr);
      }

      // Call the second-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- For the <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, the sample code in the NZ scenario (transposeX2=false) is as follows, which is for reference only. For details about the compilation and execution process, see [compilation_running_sample.md](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_v3.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
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

  int aclnnQuantMatmulV3Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Half-precision float16
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the NZ x2 aclTensor.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;
      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV3.
      workspaceSize = 0;
      ret = aclnnQuantMatmulV3GetWorkspaceSize(x1, x2, quantParam, nullptr, bias, transposeX1, transposeX2, out,
                                              &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV3.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- For the <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, the sample code for INT4 quantization is as follows (x1 and x2 are of the INT4 type, and transposeX2 is false). This is for reference only. For details about the compilation and execution process, see [compilation_running_sample.md](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_quant_matmul_v3.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      // Obtain the number of bytes of the allocated and copied memory through hostData.
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

  int aclnnQuantMatmulV3Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      int64_t m = 16;
      int64_t k = 8;
      int64_t n = 32;
      aclDataType x1Dtype = aclDataType::ACL_INT4;
      aclDataType x2Int4PackDtype = aclDataType::ACL_INT4;
      std::vector<int64_t> x1Shape = {m, k};
      std::vector<int64_t> x2Shape = {k, n};
      std::vector<int64_t> x2Int4PackShape = {k, n};
      std::vector<int64_t> biasShape = {n};
      std::vector<int64_t> scaleShape = {n};
      std::vector<int64_t> outShape = {m, n};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2Int4PackDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Int4Pack = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(m * k / 2, 17);  // int8: 0001 0001
      std::vector<int8_t> x2HostData(k * n, 1);
      std::vector<int8_t> x2Int4PackHostData(n * k / 2, 1);
      std::vector<int32_t> biasHostData(n, 1);
      std::vector<float> scaleHostData(n, 1);
      std::vector<uint16_t> outHostData(m * n, 1);

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, x1Dtype, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Int4Pack aclTensor.
      ret =
          CreateAclTensor(x2Int4PackHostData, x2Int4PackShape, &x2Int4PackDeviceAddr, x2Int4PackDtype, &x2Int4Pack);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2Int4PackTensorPtr(x2Int4Pack,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2Int4PackDeviceAddrPtr(x2Int4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      // Call aclnnConvertWeightToINT4Pack to construct the x2 input data.
      // Call the first-phase API of aclnnConvertWeightToINT4Pack.
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(x2, x2Int4Pack, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceINT4PackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceINT4PackAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnConvertWeightToINT4Pack.
      ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3GetWorkspaceSize(x1, x2Int4Pack, quantParam, nullptr, bias, transposeX1, transposeX2,
                                              out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV3.reset(workspaceAddr);
      }

      // Call the second-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
  
- <term>Atlas inference products</term>: The sample code (transposeX2 = true) when x2 is in AI processor affinity data layout format is as follows (for reference only). For details about the compilation and running process, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_permute.h"
  #include "aclnnop/aclnn_quant_matmul_v3.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_trans_quant_param_v2.h"

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

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
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

  template <typename T>
  int CreateAclTensorX2(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                        aclDataType dataType, aclTensor **tensor)
  {
      auto size = static_cast<uint64_t>(GetShapeSize(shape));

      const aclIntArray *mat2Size = aclCreateIntArray(shape.data(), shape.size());
      auto ret = aclnnCalculateMatmulWeightSizeV2(mat2Size, dataType, &size);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSizeV2 failed. ERROR: %d\n", ret);
                return ret);
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

  int aclnnQuantMatmulV3Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {5, 2};
      std::vector<int64_t> x2Shape = {2, 3};
      std::vector<int64_t> x2TransposedShape = {3, 2};
      std::vector<int64_t> biasShape = {3};
      std::vector<int64_t> offsetShape = {3};
      std::vector<int64_t> scaleShape = {3};
      std::vector<int64_t> outShape = {5, 3};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2TransposedDeviceAddr = nullptr;
      void *scaleDeviceAddr = nullptr;
      void *quantParamDeviceAddr = nullptr;
      void *offsetDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *x2Transposed = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *scale = nullptr;
      aclTensor *quantParam = nullptr;
      aclTensor *offset = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2HostData = {1, 1, 1, 1, 1, 1};
      std::vector<int8_t> x2TransposedHostData = {1, 1, 1, 1, 1, 1};
      std::vector<int32_t> biasHostData = {1, 1, 1};
      std::vector<float> scaleHostData = {1, 1, 1};
      std::vector<float> offsetHostData = {1, 1, 1};
      std::vector<uint16_t> outHostData = {1, 1, 1, 1, 1, 1, 1, 1,
                                          1, 1, 1, 1, 1, 1, 1};  // Half-precision float16
      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the NZ x2 aclTensor.
      ret = CreateAclTensorX2(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2HPTensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2HPDeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create the NZ x2Transposed aclTensor.
      ret = CreateAclTensorX2(x2TransposedHostData, x2TransposedShape, &x2TransposedDeviceAddr,
                              aclDataType::ACL_INT8, &x2Transposed);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TransposedHPTensorPtr(x2Transposed,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2TransposedHPDeviceAddrPtr(x2TransposedDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a scale aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a quantParam aclTensor.
      ret = CreateAclTensor(scaleHostData, scaleShape, &quantParamDeviceAddr, aclDataType::ACL_UINT64, &quantParam);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> quantParamTensorPtr(quantParam,
                                                                                        aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> quantParamDeviceAddrPtr(quantParamDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an offset aclTensor.
      ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> offsetTensorPtr(offset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> offsetDeviceAddrPtr(offsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_INT32, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = true;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      void *workspaceAddr = nullptr;

      // The shape of x2 needs to be transposed to the nk format before TransData.
      std::vector<int64_t> dimsData = {1, 0};
      // Create a dims aclIntArray.
      aclIntArray *dims = aclCreateIntArray(dimsData.data(), dimsData.size());
      // Call the first-phase API of aclnnPermute.
      ret = aclnnPermuteGetWorkspaceSize(x2, dims, x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrPermute(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrPermute.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnPermute.
      ret = aclnnPermute(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnPermuteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      workspaceSize = 0;
      // Call the first-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeightGetWorkspaceSize(x2Transposed, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      // Allocate device memory based on workspaceSize computed by the first-phase API.
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrTrans(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrTrans.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransMatmulWeight.
      ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

      // Call the aclnn API of the TransQuantParamV2 operator in advance for scale of the FLOAT data type.
      // Call the first-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2GetWorkspaceSize(scale, offset, quantParam, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV2(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV2.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnTransQuantParamV2.
      ret = aclnnTransQuantParamV2(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParamV2 failed. ERROR: %d\n", ret); return ret);

      // Call the first-phase API of aclnnQuantMatmulV3.
      workspaceSize = 0;
      ret = aclnnQuantMatmulV3GetWorkspaceSize(x1, x2Transposed, quantParam, nullptr, bias, transposeX1, transposeX2,
                                              out, &workspaceSize, &executor);

      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.

      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtrV3(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtrV3.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV3.
      ret = aclnnQuantMatmulV3(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %hu\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV3Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV3Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
