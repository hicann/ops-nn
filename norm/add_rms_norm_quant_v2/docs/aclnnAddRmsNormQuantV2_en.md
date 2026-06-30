# aclnnAddRmsNormQuantV2

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm_quant_v2)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: The RmsNorm operator is a standardization operation commonly used in foundation models. Compared with the LayerNorm operator, the RmsNorm operator removes the part of subtracting the mean value. The AddRmsNormQuant operator fuses the Add operator before RmsNorm and the normalized output of RmsNorm to one or two Quantize operators to reduce move-in and move-out operations. Compared with AddRmsNormQuant, the AddRmsNormQuantV2 operator adds the **betaOptional** parameter (that is, `beta` in the formula) to the RmsNorm computation process.

- Formula:

  $$
  x_i={x1}_i+{x2}_i
  $$

  $$
  y_i=\frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * gamma_i + beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  $$
  rmsNormOut_i=\frac{1}{\operatorname{Rms}(x_i)} * x_i * gamma_i
  $$

  - When **divMode** is **True**:

    $$
    y1Out=round((y/scales1)+zeroPoints1Optional)
    $$

    $$
    y2Out=round((y/scales2)+zeroPoints2Optional)
    $$
  - When **divMode** is **False**:

    $$
    y1Out=round((y*scales1)+zeroPoints1Optional)
    $$

    $$
    y2Out=round((y*scales2)+zeroPoints2Optional)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddRmsNormQuantV2GetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnAddRmsNormQuantV2` is called to perform computation.

```Cpp
aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *scales1,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  const aclTensor *betaOptional,
  int64_t          axis,
  double           epsilon,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *rmsNormOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormQuantV2GetWorkspaceSize

- **Parameters:**


  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
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
        <th>Precaution</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>x1</td>
      <td>Input</td>
      <td>Source data tensor in the standardization process. It corresponds to `x1` in the formula.</td>
      <td>Empty tensors are supported.</td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Source data tensor in the standardization process. It corresponds to `x2` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type must be the same as those of `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>Weight tensor in the standardization process. It corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of `x1`. </li><li>The shape must be the same as the dimension to be normalized in `x1`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Input</td>
      <td>scales tensor for obtaining y1Out during quantization, which corresponds to `scales1` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of `gamma`. </li><li>When divMode is set to True, this parameter cannot be set to 0.</li></ul></td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    <tr>
      <td>scales2Optional</td>
      <td>Input</td>
      <td>scales tensor used to obtain y2Out during quantization. It corresponds to `scales2` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>(Optional) A null pointer can be passed. </li><li>The shape and data type must be the same as those of `scales1`. </li><li>When `divMode` is set to True, this parameter cannot be set to 0.</li></ul></td>
      <td>FLOAT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional</td>
      <td>Input</td>
      <td>offset tensor used to obtain y1Out during quantization. It corresponds to `zeroPoints1Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>(Optional) A null pointer can be passed. </li><li>The shape must be the same as that of `gamma`.</li></ul></td>
      <td>INT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional</td>
      <td>Input</td>
      <td>offset tensor used to obtain y2Out during quantization. It corresponds to `zeroPoints2Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>(Optional) A null pointer can be passed. </li><li>The data type must be the same as that of `zeroPoints1Optional`. </li><li>The shape must be the same as that of `gamma`.</li></ul></td>
      <td>INT32, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional</td>
      <td>Input</td>
      <td>Bias in the standardization process. It corresponds to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>(Optional) A null pointer can be passed. </li><li>The shape and data type must be the same as those of `gamma`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>Input</td>
      <td>elewise axis to be quantized. Other axes are broadcast. The specified axis cannot exceed the number of dimensions of input `x1`.</td>
      <td>Currently, only -1 is supported. Other values do not take effect.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>`epsilon` in the formula, which is used to prevent division-by-zero errors.</td>
      <td>You are advised to pass a small positive number.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode</td>
      <td>Input</td>
      <td>Parameter that determines whether the quantization formula uses division, which corresponds to `divMode` in the formula.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out</td>
      <td>Output</td>
      <td>Quantized output tensor, which corresponds to `y1Out` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape must be the same as that of input `x1`/`x2`.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out</td>
      <td>Output</td>
      <td>Quantized output tensor, which corresponds to `y2Out` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional output. </li><li>The shape must be the same as that of input `x1`/`x2`. </li><li>If `scales2Optional` is empty, the output value is invalid.</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut</td>
      <td>Output</td>
      <td>Sum of x1 and x2, which corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>The shape and data type must be the same as those of input `x1`/`x2`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rmsNormOut</td>
      <td>Output</td>
      <td>Result after RmsNorm, which corresponds to `rmsNormOut` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>Optional output. </li><li>The shape and data type must be the same as those of input `x1`/`x2`.</li></ul></td>
      <td>FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
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
  </tbody>
  </table>
  
  - <term>Atlas inference series products</term>: The data types of `x1`, `x2`, `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, `zeroPoints2Optional`, `betaOptional`, `xOut`, and `rmsNormOut` cannot be BFLOAT16.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>The input parameter is a required input, output, or attribute, and is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>The shape relationship between the input and output does not meet the expectation.</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormQuantV2

- **Parameters:**

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAddRmsNormQuantV2GetWorkspaceSize.</td>
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

- <term>Atlas inference series products</term>: The number of dimensions to be normalized in `x1` and `x2` must be greater than or equal to 32. The number of `gamma`, `betaOptional`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, and `zeroPoints2Optional` data records cannot be less than 32.

- The supported scenarios and combinations of **gamma**, **scales1**, **scales2Optional**, **zeroPoints1Optional**, **zeroPoints2Optional**, **betaOptional**, **divMode**, **y1Out**, **y2Out**, **xOut**, and **rmsNormOut** are as follows:

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:

    | gamma | scales1 | scales2Optional | zeroPoints1Optional | zeroPoints2Optional | betaOptional | divMode | y1Out | y2Out | xOut | rmsNormOut |
    | --------| --------| --------| --------| --------| --------| --------| --------| --------| --------| :------ |
    | The shape is [last dimension of x1] or [1, last dimension of x1].| The shape is [1].| Null pointer| Required. The shape is [1].| Null pointer| Required. The shape must be the same as that of gamma.|True | Required| Invalid output| Required| Null pointer|
    | The shape is [last dimension of x1] or [1, last dimension of x1].| The shape is [1].| Null pointer| Required. The shape is [1].| Null pointer| Null pointer|True | Required| Invalid output| Null pointer| Required|
    | The shape is the same as the to-be-normalized dimension of x1.| The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.|True/False | Required| When scales2Optional is empty, this output is invalid. When scales2Optional is not empty, this output is valid.| Required| Null pointer|

  - <term>Atlas inference series products</term>:

    | gamma | scales1 | scales2Optional | zeroPoints1Optional | zeroPoints2Optional | betaOptional | divMode | y1Out | y2Out | xOut | rmsNormOut |
    | --------| --------| --------| --------| --------| --------| --------| --------| --------| --------| :------ |
    | The shape is the same as the to-be-normalized dimension of x1.| The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.| Optional. The shape is the same as that of gamma.|True/False | Required| When scales2Optional is empty, this output is invalid. When scales2Optional is not empty, this output is valid.| Required| Null pointer|

- Description of boundary value scenarios:

  - <term>Atlas inference series products</term>: The input cannot contain inf and NaN.
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: When the input is inf, the output is inf. When the input is NaN, the output is NaN.

- Dimension boundary description:

  The size of each dimension in the shape of `x1`, `x2`, `gamma`, `scales1`, `scales2Optional`, `zeroPoints1Optional`, `zeroPoints2Optional`, `betaOptional`, `y1Out`, `y2Out`, `xOut`, and `rmsNormOut` must be less than or equal to the maximum value 2147483647 of INT32.
  
- Data format description:

    The ND format is recommended for all input and output tensors. If other data formats are used, the framework converts them into the ND format by default for processing.

- Description of data types supported by different product models:
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:

     | x1| x2| gamma| scales1| scales2Optional| zeroPoints1Optional| zeroPoints2Optional| betaOptional| y1Out| y2Out| xOut| rmsNormOut|
     | - | - | - | - | - | - | - | - | - | - | - | - |
     | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |
     | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 |

  - <term>Atlas inference series products</term>:

    | x1| x2| gamma| scales1| scales2Optional| zeroPoints1Optional| zeroPoints2Optional| betaOptional| y1Out| y2Out| xOut| rmsNormOut|
    | - | - | - | - | - | - | - | - | - | - | - | - |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | INT32 | INT32 | FLOAT16 | INT8 | INT8 | FLOAT16 | FLOAT16 |

- Deterministic compute:
  - **aclnnAddRmsNormQuantV2** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_quant_v2.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

    // Call aclrtMemcpy to copy the data from the host to the device.
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

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output based on the API.
    std::vector<int64_t> xShape = {64, 32};
    std::vector<int64_t> gammaShape = {32};
    std::vector<int64_t> yShape = {64, 32};
    long long xShapeSize = GetShapeSize(xShape);
    long long gammaShapeSize = GetShapeSize(gammaShape);
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* scales1DeviceAddr = nullptr;
    void* zeroPoints1DeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* scales1 = nullptr;
    aclTensor* zeroPoints1 = nullptr;
    aclTensor* y1 = nullptr;
    aclTensor* y2 = nullptr;
    aclTensor* x = nullptr;
    std::vector<int16_t> x1HostData(xShapeSize, 0);
    std::vector<int16_t> x2HostData(xShapeSize, 0);
    std::vector<int16_t> gammaHostData(gammaShapeSize, 0);
    std::vector<int16_t> betaHostData(gammaShapeSize, 0);
    std::vector<float> scales1HostData(gammaShapeSize, 1);
    std::vector<int32_t> zeroPoints1HostData(gammaShapeSize, 100);
    std::vector<int8_t> y1HostData(xShapeSize, 0);
    std::vector<int8_t> y2HostData(xShapeSize, 0);
    std::vector<int16_t> xHostData(xShapeSize, 0);
    float epsilon = 1e-6;
    int64_t axis = -1;
    bool divMode = true;
    // Create an x1 aclTensor.
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x2 aclTensor.
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a gamma aclTensor.
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a beta aclTensor.
    ret = CreateAclTensor(betaHostData, gammaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a scales1 aclTensor.
    ret = CreateAclTensor(scales1HostData, gammaShape, &scales1DeviceAddr, aclDataType::ACL_FLOAT, &scales1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a zeroPoints1 aclTensor.
    ret =
        CreateAclTensor(zeroPoints1HostData, gammaShape, &zeroPoints1DeviceAddr, aclDataType::ACL_INT32, &zeroPoints1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y1 aclTensor.
    ret = CreateAclTensor(y1HostData, yShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a y2 aclTensor.
    ret = CreateAclTensor(y2HostData, yShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnAddRmsNormQuantV2.
    ret = aclnnAddRmsNormQuantV2GetWorkspaceSize(
        x1, x2, gamma, scales1, nullptr, zeroPoints1, nullptr, beta, axis, epsilon, divMode, y1, y2, x, nullptr,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAddRmsNormQuantV2.
    ret = aclnnAddRmsNormQuantV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddRmsNormQuantV2 failed. ERROR: %d\n", ret); return ret);
    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(yShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), y1DeviceAddr, size * sizeof(int8_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(beta);
    aclDestroyTensor(scales1);
    aclDestroyTensor(zeroPoints1);
    aclDestroyTensor(y1);
    aclDestroyTensor(y2);
    aclDestroyTensor(x);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(scales1DeviceAddr);
    aclrtFree(zeroPoints1DeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(xDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
