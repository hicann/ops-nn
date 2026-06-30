# aclnnAddLayerNorm

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     √    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Implements the AddLayerNorm function.
- Formula:

  $$
  x = x1 + x2 + biasOptional
  $$

  $$
  rstd = {{1}\over\sqrt {Var(x)+eps}}
  $$

  $$
  y = (x-E(x)) * rstd * gamma + beta
  $$

  Where, E(x) indicates the mean value and Var(x) indicates the variance. Both the mean value and variance must be calculated within the operator.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnAddLayerNormGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnAddLayerNorm` is called to perform computation.

```Cpp
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  double           epsilon,
  bool             additionalOutput,
  const aclTensor *yOut,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *xOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGetWorkspaceSize

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
      <td>Input of the addition operation in AddLayerNorm. The operator performs the x1 + x2 + biasOptional operation and normalizes the result by layer. It corresponds to `x1` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The value of any dimension of the input cannot be 0.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input of the addition operation in AddLayerNorm. The operator performs the x1 + x2 + biasOptional operation and normalizes the result by layer. It corresponds to `x2` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is the same as that of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>beta parameter for layer normalization. It corresponds to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The dimension value of shape is the same as that of the dimension to be normalized in `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma parameter for layer normalization. It corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The dimension value of shape is the same as that of the dimension to be normalized in `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>(Optional) Input of the addition operation in AddLayerNorm. The operator performs the x1 + x2 + biasOptional operation and normalizes the result by layer. It corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape can be consistent with that of `gamma`/`beta` or `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Input</td>
      <td>Value added to the denominator to ensure numerical stability. It corresponds to `epsilon` in the formula.</td>
      <td>The value can only be 1e-5.</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput</td>
      <td>Input</td>
      <td>Indicates whether to enable the output of x = x1 + x2 + biasOptional.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>Output</td>
      <td>Indicates the mean value of the result of (x1 + x2 + biasOptional) in the LayerNorm computation process. It corresponds to `E(x)` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with `x1`. (The first several dimensions are the same as those of `x1`, the rest dimensions are 1, and the total number of dimensions is the same as that of `x1`. The first several dimensions are the dimensions of `x1` minus the dimensions of gamma, indicating the dimensions that do not require normalization.)</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>Output</td>
      <td>Result of `rstd` in the LayerNorm computation process. It corresponds to `rstd` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must meet the <a href="../../../docs/en/context/broadcast_relationship.md">Broadcast Relationship</a> with <idp:inline displayname="code" id="code11514169406">x1</idp:inline>. (The first several dimensions are the same as those of <idp:inline displayname="code" id="code121513167400">x1</idp:inline>, the rest dimensions are 1, and the total number of dimensions is the same as that of <idp:inline displayname="code" id="code1151416104013">x1</idp:inline>. The first several dimensions are the dimensions of <idp:inline displayname="code" id="code61591617400">x1</idp:inline> minus the dimensions of gamma, indicating the dimensions that do not require normalization.)</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>Output</td>
      <td>Output of the LayerNorm result. It corresponds to `y` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut</td>
      <td>Output</td>
      <td>Output `x` of the Add result. It corresponds to `x` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape must be the same as that of input `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
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

  - <term>Atlas inference series products</term>:
    - The data types of the `x1`, `x2`, `beta`, `gamma`, `biasOptional`, `yOut`, and `xOut` parameters cannot be BFLOAT16.
    - The `meanOut` and `rstdOut` parameters are invalid in the current product.

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
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="9">561002</td>
      <td>Failed to obtain the shape of x1, x2, gamma, beta, yOut, meanOut, rstdOut, or xOut in the tiling phase.</td>
    </tr>
    <tr>
      <td>The shape of x1 or gamma is greater than 8D or less than 0D.</td>
    </tr>
    <tr>
      <td>The number of dimensions is inconsistent for x1, x2, yOut, meanOut, rstdOut, and xOut.</td>
    </tr>
    <tr>
      <td>The number of dimensions of x1 is less than that of gamma.</td>
    </tr>
    <tr>
      <td>Any dimension of x1, gamma, or meanOut is equal to 0.</td>
    </tr>
    <tr>
      <td>The shapes of x1, x2, yOut, and xOut are not the same.</td>
    </tr>
    <tr>
      <td>The shapes of gamma and beta are not the same.</td>
    </tr>
    <tr>
      <td>The shapes of meanOut and rstdOut are not the same.</td>
    </tr>
    <tr>
      <td>The dimension of gamma is different from the to-be-normalized dimension of x, the dimension of meanOut is different from the dimension of x that does not to be normalized, or the to-be-normalized dimension of meanOut is not 1.</td>
    </tr>
  </tbody></table>


## aclnnAddLayerNorm

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAddLayerNormGetWorkspaceSize.</td>
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

- **Function dimension**
  - The supported data types are as follows:
    - <term>Atlas inference series products</term>: x1, x2, beta, gamma, and biasOptional support FLOAT32 and FLOAT16.
    - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>: x1, x2, beta, gamma, and biasOptional support FLOAT32, FLOAT16, and BFLOAT16.
    - rstdOut and meanOut support FLOAT32.
  - The data format can be ND.
  - <term>Atlas inference series products</term>: The length of the last axis of the inputs x1, x2, beta, gamma, and biasOptional must be greater than or equal to 32 bytes.
- **Description of unsupported types**
  - DOUBLE: DOUBLE is not supported.
- **Description of boundary value scenarios**
  - When the input is Inf, the output is Inf.
  - When the input is NaN, the output is NaN.
- **Description of data types supported by different products**
  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
    | x1| x2| gamma| beta| biasOptional| yOut| meanOut| rstdOut| xOut|
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT32  | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | BFLOAT16 | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
    | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32  | FLOAT32  | FLOAT16  |
    | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32  | FLOAT32  | BFLOAT16 |
    | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  |
  - <term>Atlas inference series products</term>:
    | x1| x2| gamma| beta| biasOptional| yOut| meanOut| rstdOut| xOut|
    | -------- | -------- | ------------- | ------------- | ----------- | --------- | --------- | --------- | :-------- |
    | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 |
    | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT32 | FLOAT32 | FLOAT16 |
- Deterministic compute:
  - **aclnnAddLayerNorm** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm.h"

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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2.
    // Construct the input and output based on the API. In this example, the case without biasOptional and the case with biasOptional are called once.
    float eps = 1e-6;
    bool additionalOutput = true;

    std::vector<int64_t> x1Shape = {1, 2, 8};
    std::vector<int64_t> x2Shape = {1, 2, 8};
    std::vector<int64_t> gammaShape = {8};
    std::vector<int64_t> betaShape = {8};
    std::vector<int64_t> biasOptionalShape = {8};

    std::vector<int64_t> outputYShape = {1, 2, 8};
    std::vector<int64_t> outputMeanShape = {1, 2, 1};
    std::vector<int64_t> outputRstdShape = {1, 2, 1};
    std::vector<int64_t> outputXShape = {1, 2, 8};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* biasOptionalDeviceAddr = nullptr;

    // Used for output device address without biasOptional
    void* outputYDeviceAddr = nullptr;
    void* outputMeanDeviceAddr = nullptr;
    void* outputRstdDeviceAddr = nullptr;
    void* outputXDeviceAddr = nullptr;

    // Used for output device address with biasOptional
    void* outputYDeviceAddrbiasOptional = nullptr;
    void* outputMeanDeviceAddrbiasOptional = nullptr;
    void* outputRstdDeviceAddrbiasOptional = nullptr;
    void* outputXDeviceAddrbiasOptional = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* biasOptional = nullptr;

    // Used for aclTensor without biasOptional
    aclTensor* outputY = nullptr;
    aclTensor* outputMean = nullptr;
    aclTensor* outputRstd = nullptr;
    aclTensor* outputX = nullptr;

    // Used for aclTensor with biasOptional
    aclTensor* outputYbiasOptional = nullptr;
    aclTensor* outputMeanbiasOptional = nullptr;
    aclTensor* outputRstdbiasOptional = nullptr;
    aclTensor* outputXbiasOptional = nullptr;

    std::vector<float> x1HostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> x2HostData = {4, 4, 4, 4, 4, 4, 4, 4, -3, -3, -3, -3, -3, -3, -3, -3};
    std::vector<float> gammaHostData = {2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<float> betaHostData = {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1};
    std::vector<float> biasOptionalHostData = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};

    // Used for HostData without biasOptional
    std::vector<float> outputYHostData(1 * 2 * 8);
    std::vector<float> outputMeanHostData(2);
    std::vector<float> outputRstdHostData(2);
    std::vector<float> outputXHostData(1 * 2 * 8);

    // Used for HostData with biasOptional
    std::vector<float> outputYHostDatabiasOptional(1 * 2 * 8);
    std::vector<float> outputMeanHostDatabiasOptional(2);
    std::vector<float> outputRstdHostDatabiasOptional(2);
    std::vector<float> outputXHostDatabiasOptional(1 * 2 * 8);

    // Create a self aclTensor.
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        biasOptionalHostData, biasOptionalShape, &biasOptionalDeviceAddr, aclDataType::ACL_FLOAT, &biasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an aclTensor without biasOptional.
    ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(outputXHostData, outputXShape, &outputXDeviceAddr, aclDataType::ACL_FLOAT, &outputX);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an aclTensor with biasOptional.
    ret = CreateAclTensor(
        outputYHostDatabiasOptional, outputYShape, &outputYDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputYbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputMeanHostDatabiasOptional, outputMeanShape, &outputMeanDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputMeanbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputRstdHostDatabiasOptional, outputRstdShape, &outputRstdDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputRstdbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputXHostDatabiasOptional, outputXShape, &outputXDeviceAddrbiasOptional, aclDataType::ACL_FLOAT,
        &outputXbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // aclnnAddLayerNorm API call example, including both the case with biasOptional and the case without biasOptional
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.

    // 3.1 Example that does not include biasOptional
    // Call the first-phase API of aclnnAddLayerNorm.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    LOG_PRINT("\nUse aclnnAddLayerNorm Non-biasOptional Port.");
    // Pass nullptr directly to the biasOptional parameter.
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, nullptr, eps, additionalOutput, outputY, outputMean, outputRstd, outputX, &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAddLayerNorm.
    ret = aclnnAddLayerNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 3.2 Example that includes biasOptional
    // Call the first-phase API of aclnnAddLayerNorm.
    uint64_t workspaceSizebiasOptional = 0;
    aclOpExecutor* executorbiasOptional;
    LOG_PRINT("\nUse aclnnAddLayerNorm biasOptional Port.");
    // Pass biasOptional.
    ret = aclnnAddLayerNormGetWorkspaceSize(
        x1, x2, gamma, beta, biasOptional, eps, additionalOutput, outputYbiasOptional, outputMeanbiasOptional,
        outputRstdbiasOptional, outputXbiasOptional, &workspaceSizebiasOptional, &executorbiasOptional);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddrbiasOptional = nullptr;
    if (workspaceSizebiasOptional > 0) {
        ret = aclrtMalloc(&workspaceAddrbiasOptional, workspaceSizebiasOptional, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnAddLayerNorm.
    ret = aclnnAddLayerNorm(workspaceAddrbiasOptional, workspaceSizebiasOptional, executorbiasOptional, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNorm failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.

    // 5.1 Copy the output that does not include biasOptional.
    auto outputYSize = GetShapeSize(outputYShape);
    std::vector<float> resultDataY(outputYSize, 0);
    ret = aclrtMemcpy(
        resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
        outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: y output");
    for (int64_t i = 0; i < outputYSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
    }

    auto outputMeanSize = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMean(outputMeanSize, 0);
    ret = aclrtMemcpy(
        resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), outputMeanDeviceAddr,
        outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
    }

    auto outputRstdSize = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstd(outputRstdSize, 0);
    ret = aclrtMemcpy(
        resultDataRstd.data(), resultDataRstd.size() * sizeof(resultDataRstd[0]), outputRstdDeviceAddr,
        outputRstdSize * sizeof(resultDataRstd[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
    }

    auto outputXSize = GetShapeSize(outputXShape);
    std::vector<float> resultDataX(outputXSize, 0);
    ret = aclrtMemcpy(
        resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), outputXDeviceAddr,
        outputXSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm non-biasOptional: x output");
    for (int64_t i = 0; i < outputXSize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
    }

    // 5.2 Copy the output that includes biasOptional.
    auto outputYSizebiasOptional = GetShapeSize(outputYShape);
    std::vector<float> resultDataYbiasOptional(outputYSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataYbiasOptional.data(), resultDataYbiasOptional.size() * sizeof(resultDataYbiasOptional[0]),
        outputYDeviceAddrbiasOptional, outputYSizebiasOptional * sizeof(resultDataYbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: y output");
    for (int64_t i = 0; i < outputYSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataYbiasOptional[i]);
    }

    auto outputMeanSizebiasOptional = GetShapeSize(outputMeanShape);
    std::vector<float> resultDataMeanbiasOptional(outputMeanSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataMeanbiasOptional.data(), resultDataMeanbiasOptional.size() * sizeof(resultDataMeanbiasOptional[0]),
        outputMeanDeviceAddrbiasOptional, outputMeanSizebiasOptional * sizeof(resultDataMeanbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: mean output");
    for (int64_t i = 0; i < outputMeanSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataMeanbiasOptional[i]);
    }

    auto outputRstdSizebiasOptional = GetShapeSize(outputRstdShape);
    std::vector<float> resultDataRstdbiasOptional(outputRstdSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataRstdbiasOptional.data(), resultDataRstdbiasOptional.size() * sizeof(resultDataRstdbiasOptional[0]),
        outputRstdDeviceAddrbiasOptional, outputRstdSizebiasOptional * sizeof(resultDataRstdbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: rstd output");
    for (int64_t i = 0; i < outputRstdSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstdbiasOptional[i]);
    }

    auto outputXSizebiasOptional = GetShapeSize(outputXShape);
    std::vector<float> resultDataXbiasOptional(outputXSizebiasOptional, 0);
    ret = aclrtMemcpy(
        resultDataXbiasOptional.data(), resultDataXbiasOptional.size() * sizeof(resultDataXbiasOptional[0]),
        outputXDeviceAddrbiasOptional, outputXSizebiasOptional * sizeof(resultDataXbiasOptional[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("==== AddLayerNorm biasOptional: x output");
    for (int64_t i = 0; i < outputXSizebiasOptional; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataXbiasOptional[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(beta);
    aclDestroyTensor(gamma);
    aclDestroyTensor(biasOptional);

    aclDestroyTensor(outputY);
    aclDestroyTensor(outputMean);
    aclDestroyTensor(outputRstd);
    aclDestroyTensor(outputX);

    aclDestroyTensor(outputYbiasOptional);
    aclDestroyTensor(outputMeanbiasOptional);
    aclDestroyTensor(outputRstdbiasOptional);
    aclDestroyTensor(outputXbiasOptional);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(betaDeviceAddr);
    aclrtFree(biasOptionalDeviceAddr);

    aclrtFree(outputYDeviceAddr);
    aclrtFree(outputMeanDeviceAddr);
    aclrtFree(outputRstdDeviceAddr);
    aclrtFree(outputXDeviceAddr);

    aclrtFree(outputYDeviceAddrbiasOptional);
    aclrtFree(outputMeanDeviceAddrbiasOptional);
    aclrtFree(outputRstdDeviceAddrbiasOptional);
    aclrtFree(outputXDeviceAddrbiasOptional);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    if (workspaceSizebiasOptional > 0) {
        aclrtFree(workspaceAddrbiasOptional);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
