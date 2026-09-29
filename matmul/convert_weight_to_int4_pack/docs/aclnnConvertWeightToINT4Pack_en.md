# aclnnConvertWeightToINT4Pack

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/convert_weight_to_int4_pack)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

Preprocesses the input **weight** data to convert the layout of low-bit data from sparse storage to dense storage. When the  [Data Format](../../../docs/en/context/data_format.md) of the output **weightInt4Pack** is set to **FRACTAL_NZ**, this operator converts the  [Data Format](../../../docs/en/context/data_format.md) from **ND** to **FRACTAL_NZ**.

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: The input **weight** data of type INT32 is packed into INT4 data in a compact layout.

- Ascend 950PR/Ascend 950DT: packs the weight data of the INT32 type into tightly packed INT4 data and packs the weight data of the FLOAT type into tightly packed FLOAT4_E2M1 data.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnConvertWeightToINT4PackGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnConvertWeightToINT4Pack** is called to perform computation.

```Cpp
aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
  const aclTensor *weight,
  aclTensor       *weightInt4Pack,
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnConvertWeightToINT4Pack(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor, 
  aclrtStream      stream)
```

## aclnnConvertWeightToINT4PackGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1078px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 320px">
  <col style="width: 183px">
  <col style="width: 183px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Low-bit quantized weight for Matmul-like operators. The weight values are 4-bit but stored in a 32-bit data type.</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>2–3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightInt4Pack</td>
      <td>Output</td>
      <td>Low-bit quantized weight for Matmul-like operators. The weight values are 4-bit and densely packed.</td>
      <td>-</td>
      <td>INT4,INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
    <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>The shape of the passed weight or weightInt4Pack does not meet the requirements.</td>
    </tr>
    <tr>
      <td>The data type of weight or weightInt4Pack is not supported.</td>
    </tr>
    <tr>
      <td>The shape size of weight or weightInt4Pack does not meet the constraints.</td>
    </tr>
    <tr>
      <td>An empty tensor is passed.</td>
    </tr>
    <tr>
      <td>The format of the input tensor is not ND.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_RUNTIME_ERROR</td>
      <td rowspan="2">361001</td>
      <td>An exception occurred in copying data from the host to the device.</td>
    </tr>
    <tr>
      <td>An exception occurred in copying data from the device to the host.</td>
    </tr>
  </tbody>
  </table>

## aclnnConvertWeightToINT4Pack

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnConvertWeightToINT4PackGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
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

- Deterministic description:

  - <term>Atlas A2 training products/Atlas A2 inference products</term>, <term>Atlas A3 training products/Atlas A3 inference products</term>, and Ascend 950PR/Ascend 950DT: The default deterministic implementation of aclnnConvertWeightToINT4Pack is used.

- The relationships between data types and data formats of the parameters are listed as follows.

    - <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight Data Type</th>
          <th>weight Data Format</th>
          <th>weightInt4Pack Data Type</th>
          <th>weightInt4Pack Data Format</th>
          <th>weight Shape</th>
          <th>weightInt4Pack View Shape</th>
          <th>weightInt4Pack Storage Shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32 (carrying INT4 data, with a value range of [–8, 7])</td>
          <td>ND</td>
          <td>INT4</td>
          <td>ND</td>
          <td>Last dimension aligned to 2</td>
          <td>Same as the input weight, that is, (dim0, dim1)</td>
          <td>Same as the view shape</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, with a value range of [–8, 7])</td>
          <td>ND</td>
          <td>INT4</td>
          <td>FRACTAL_NZ</td>
          <td>Last dimension aligned to 2</td>
          <td>Same as the input weight, that is, (dim0, dim1)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 64)</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, with a value range of [–8, 7])</td>
          <td>ND</td>
          <td>INT32 (each storing eight INT4 values)</td>
          <td>ND</td>
          <td>Last dimension aligned to 8</td>
          <td>Last dimension = Last dimension of weight/8, that is, (dim0, dim1/8)</td>
          <td>Same as the view shape</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, with a value range of [–8, 7])</td>
          <td>ND</td>
          <td>INT32 (each storing eight INT4 values)</td>
          <td>FRACTAL_NZ</td>
          <td>Last dimension aligned to 8</td>
          <td>Last dimension = Last dimension of weight/8, that is, (dim0, dim1/8)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 8)</td>
        </tr>
      </tbody></table>

    - Ascend 950PR/Ascend 950DT:
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight Data Type</th>
          <th>weight Data Format</th>
          <th>weightInt4Pack Data Type</th>
          <th>weightInt4Pack Data Format</th>
          <th>weight Shape</th>
          <th>weightInt4Pack View Shape</th>
          <th>weightInt4Pack Storage Shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32 (data of the INT4 type, ranging from -8 to 7) or FLOAT (data of the FLOAT4_E2M1 type, ranging from -6.0 to 6.0)</td>
          <td>ND</td>
          <td>INT4 or FLOAT4_E2M1</td>
          <td>ND</td>
          <td>Last dimension aligned to 2</td>
          <td>Same as the input weight, that is, (dim0, dim1)</td>
          <td>Same as the view shape</td>
        </tr>
        <tr>
          <td>INT32 (data of the INT4 type, ranging from -8 to 7) or FLOAT (data of the FLOAT4_E2M1 type, ranging from -6.0 to 6.0)</td>
          <td>ND</td>
          <td>INT4 or FLOAT4_E2M1</td>
          <td>FRACTAL_NZ</td>
          <td>Last dimension aligned to 2</td>
          <td>Same as the input weight, that is, (dim0, dim1)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 16)</td>
        </tr>
        <tr>
          <td>INT32 (data of the INT4 type, ranging from –8 to 7) or FLOAT (data of the FLOAT4_E2M1 type, ranging from –6.0 to 6.0)</td>
          <td>ND</td>
          <td>INT32 (one INT32 element stores eight INT4 elements) or FLOAT (one FLOAT element stores eight FLOAT4_E2M1 elements)</td>
          <td>ND</td>
          <td>Last dimension aligned to 8</td>
          <td>Last dimension = Last dimension of weight/8, that is, (dim0, dim1/8)</td>
          <td>Same as the view shape</td>
        </tr>
        <tr>
          <td>INT32 (data of the INT4 type, ranging from –8 to 7) or FLOAT (data of the FLOAT4_E2M1 type, ranging from –6.0 to 6.0)</td>
          <td>ND</td>
          <td>INT32 (one INT32 element stores eight INT4 elements) or FLOAT (one FLOAT element stores eight FLOAT4_E2M1 elements)</td>
          <td>FRACTAL_NZ</td>
          <td>Last dimension aligned to 8</td>
          <td>Last dimension = Last dimension of weight/8, that is, (dim0, dim1/8)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 2)</td>
        </tr>
        <tr>
          <td>INT32 (carrying INT4 data, with a value range of [–8, 7])</td>
          <td>ND</td>
          <td>INT4 (one INT32 element stores eight INT4 elements)</td>
          <td>FRACTAL_NZ</td>
          <td>Last dimension aligned to 8</td>
          <td>Last dimension = Last dimension of weight/8, that is, (dim0, dim1/8)</td>
          <td>(⌈dim1/32⌉, ⌈dim0/16⌉, 16, 32)</td>
        </tr>
      </tbody></table>

## Example

- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>:
  The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  **aclnnWeightQuantBatchMatmulV2** and **aclnnWeightQuantBatchMatmulV3** are available for fake quantization. **aclnnWeightQuantBatchMatmulV2** is used as an example.

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
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

    // Calculate the strides of the contiguous tensor.
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
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 64};
      } else {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 8};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = true;

    // 2. Construct inputs and outputs based on API definitions.
    int64_t m = 16;
    int64_t k = 72;
    int64_t n = 17;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightInt4PackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); // Each INT8 element stores two INT4 values, so the value is divided by 2.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 32) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // Create a weightInt4Pack aclTensor.
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); // Each INT32 element stores eight INT4 values, so the value is divided by 8.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 8) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                  weightInt4PackDtype, &weightInt4Pack, weightFormat);
      } else {
          // Create a weightInt4Pack aclTensor.
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // Create a y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an antiquantScale aclTensor.
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an xFp16 aclTensor.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an antiquantScale aclTensor.
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a yFp16 aclTensor.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // Convert weight from INT32 to INT4 pack format.
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // When weight is transposed and the weightInt4Pack format is NZ, call aclInitTensor to convert it to a non-contiguous tensor.
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 64), CEIL_DIV(n, 16), 16, 8};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 64;
      }
      aclInitTensor(weightInt4Pack, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightInt4PackDeviceAddr);
    }

    // Call cast to generate the FP16 input.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Convert the output to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightInt4Pack);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightInt4PackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }

- Ascend 950PR/Ascend 950DT:

 The following example (INT32 input) is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  The fake-quantization APIs include aclnnWeightQuantBatchMatmulV2 and aclnnWeightQuantBatchMatmulV3. aclnnWeightQuantBatchMatmulV2 is used as an example here.

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
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

    // Calculate the strides of the contiguous tensor.
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
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
      } else {
          nzShape = {CEIL_DIV(shape[1], 2), CEIL_DIV(shape[0], 16), 16, 2};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = false;

    // 2. Construct inputs and outputs based on API definitions.
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightPackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightPacked = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); // Each INT8 element stores two INT4 values, so the value is divided by 2.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 8) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // Create weightPacked aclTensor.
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                weightInt4PackDtype, &weightPacked, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); // Each INT32 element stores eight INT4 values, so the value is divided by 8.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 2) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                  weightInt4PackDtype, &weightPacked, weightFormat);
      } else {
          // Create weightPacked aclTensor.
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightPackDeviceAddr,
                                weightInt4PackDtype, &weightPacked);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // Create a y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an antiquantScale aclTensor.
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an xFp16 aclTensor.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an antiquantScale aclTensor.
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a yFp16 aclTensor.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // Convert weight from INT32 to INT4 pack format.
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // If the weight is transposed and the weightPacked shape is NZ, aclInitTensor needs to be called to convert the weight to a non-contiguous tensor.
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 16), CEIL_DIV(n, 16), 16, 2};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 16;
      }
      aclInitTensor(weightPacked, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightPackDeviceAddr);
    }

    // Call cast to generate the FP16 input.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightPacked, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Convert the output to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightPacked);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightPackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

- Ascend 950PR/Ascend 950DT:
  The following example (FLOAT input) is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).
  aclnnWeightQuantBatchMatmulV2 and aclnnWeightQuantBatchMatmulV3 are available for fake quantization. aclnnWeightQuantBatchMatmulV2 is used as an example.

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

  int Init(int32_t deviceId, aclrtStream* stream) {
    // (Boilerplate) Initialize resources.
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

    // Calculate the strides of the contiguous tensor.
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
  int CreateAclTensorB4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
    auto size = hostData.size() * sizeof(T);
    // Call aclrtMalloc to allocate memory on the device.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy the data on the host to the memory on the device.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call aclCreateTensor to create an aclTensor.
    if (format == aclFormat::ACL_FORMAT_ND) {
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
          nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
      } else {
          nzShape = {CEIL_DIV(shape[1], 2), CEIL_DIV(shape[0], 16), 16, 2};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightPackedDtype = aclDataType::ACL_FLOAT4_E2M1; // (Optional) ACL_FLOAT type
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // (Optional) ACL_FORMAT_ND
    bool isWeightTransposed = false; // ND: true or false. NZ: false

    // 2. Construct inputs and outputs based on API definitions.
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t antiquantGroupSize = 32;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightPackedShape;
    if (weightPackedDtype == aclDataType::ACL_FLOAT4_E2M1) {
      weightPackedShape = {weightDim0, weightDim1};
    } else {
      weightPackedShape = {weightDim0, weightDim1 / 8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightPackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightPacked = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<float> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {k / antiquantGroupSize, n};
    std::vector<uint8_t> antiquantScaleHostData(n * k / antiquantGroupSize, 1); // Uses uint8 to Carry float8_e8m0 Data

    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    if (weightPackedDtype == aclDataType::ACL_FLOAT4_E2M1) {
      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // One B8 data stores two B4 data. Therefore, the value is divided by 2.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightB4PackHostData.resize(CEIL_ALIGN(weightDim1 / 2, 8) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // Create the weightPacked aclTensor.
      ret = CreateAclTensorB4(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                weightPackedDtype, &weightPacked, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightB4PackHostData(n * k / 8, 1); // One int32 data stores eight int4 data. Therefore, the value is divided by 8.
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightB4PackHostData.resize(CEIL_ALIGN(weightDim1 / 8, 2) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorB4(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                  weightPackedDtype, &weightPacked, weightFormat);
      } else {
          // Create weightPacked aclTensor.
          ret = CreateAclTensor(weightB4PackHostData, weightPackedShape, &weightPackDeviceAddr,
                                weightPackedDtype, &weightPacked);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // Create a y aclTensor.
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an xFp16 aclTensor.
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an antiquantScale aclTensor.
    void* antiquantScaleFp8DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp8 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp8DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &antiquantScaleFp8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a yFp16 aclTensor.
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // Convert weight from INT32 to INT4 pack format.
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // If weight is in the transpose scenario and the weightPacked shape is ND, call aclInitTensor to convert the tensor to a non-contiguous tensor.
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_ND) {
      weightPackedShape = {k, n};
      std::vector<int64_t> strides = {1, k};
      std::vector<int64_t> storage_shape = {n, k};
      if (weightPackedDtype == aclDataType::ACL_FLOAT) {
        weightPackedShape = {k / 8, n};
        strides = {1, k / 8};
        storage_shape = {n, k / 8};
      }
      aclInitTensor(weightPacked, weightPackedShape.data(), weightPackedShape.size(), weightPackedDtype, strides.data(), 0,
                    weightFormat, storage_shape.data(), storage_shape.size(), weightPackDeviceAddr);
    }

    // Call cast to generate the FP16 input.
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // Call the first-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightPacked, antiquantScaleFp8, nullptr, nullptr, nullptr, nullptr, antiquantGroupSize, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnWeightQuantBatchMatmulV2.
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // Convert the output to FP32.
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightPacked);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp8);
    aclDestroyTensor(yFp16);

    // 7. Release device resources.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightPackDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp8DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```
