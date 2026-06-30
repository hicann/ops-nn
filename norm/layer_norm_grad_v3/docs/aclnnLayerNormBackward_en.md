# aclnnLayerNormBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_grad_v3)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √   |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √   |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×   |
|  <term>Atlas inference series products</term>   |     √   |
|  <term>Atlas training series products</term>   |     √   |

## Function

- Description: Performs backpropagation of [aclnnLayerNorm](../../layer_norm_v4/docs/aclnnLayerNorm&aclnnLayerNormWithImplMode_en.md). It is used to calculate the gradient of the input tensor so that the model parameters can be updated during backpropagation.
- Formula:
  
  $$
  res\_for\_gamma = (input - mean) \times rstd
  $$
  
  $$
  dy\_g = gradOut \times weightOptional
  $$
  
  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_1} gradOut \times weightOptional
  $$
  
  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_1}(gradOut \times weightOptional \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weightOptional - (temp_1 + temp_2)) \times rstd
  $$
  
  $$
  gradWeightOut =  \sum_{reduce\_axis\_0}gradOut \times (input - mean) \times rstd
  $$
  
  $$
  gradBiasOut = \sum_{reduce\_axis\_0}gradOut
  $$

  N indicates the dimension of the axis on which normalization computation is performed, that is, the size of the normalized axis dimension.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnLayerNormBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnLayerNormBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(
  const aclTensor    *gradOut,
  const aclTensor    *input,
  const aclIntArray  *normalizedShape,
  const aclTensor    *mean,
  const aclTensor    *rstd,
  const aclTensor    *weightOptional,
  const aclTensor    *biasOptional,
  const aclBoolArray *outputMask,
  aclTensor          *gradInputOut,
  aclTensor          *gradWeightOut,
  aclTensor          *gradBiasOut,
  uint64_t           *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnLayerNormBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLayerNormBackwardGetWorkspaceSize

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
      <th>Usage Notes</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradOut</td>
      <td>Input</td>
      <td>Gradient tensor for backpropagation, corresponding to `gradOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. <li>The data type is the same as that of input. <li>The shape is the same as that of input, that is, [A1,...,Ai,R1,...,Rj], and has a length that is greater than or equal to that of normalizedShape.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>First input of forward computation, corresponding to `input` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. <li>The data type is the same as that of gradOut. <li>The shape is the same as that of gradOut, that is, [A1,...,Ai,R1,...,Rj], and has a length that is greater than or equal to that of normalizedShape.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape</td>
      <td>Input</td>
      <td>Dimension for normalization computation, corresponding to reduce_axis_1 in the formula.</td>
      <td><ul><li>reduce_axis_0 in the formula indicates the dimension that does not require normalization computation. <li>The value is [R1,...,Rj]. The length is less than or equal to the length of the input shape. The value cannot be empty.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>Input</td>
      <td>Second output of forward computation, indicating the mean value of the input. It corresponds to mean in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type is the same as that of rstd, and the bit width is not less than that of input. </li><li>The shape is the same as that of rstd, which is [A1,...,Ai,1,...,1] with j 1s after Ai and is the same as the length of the axis that requires normalization.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>Input</td>
      <td>Third output of forward computation, indicating the reciprocal of the standard deviation of input. It corresponds to rstd in the formula.</td>
      <td><ul><li>Empty tensors are not supported. <li>The data type is the same as that of mean, and the bit width is not less than that of input. <li>The shape is the same as that of mean, which is [A1,...,Ai,1,...,1] with j 1s after Ai and is the same as the length of the axis that requires normalization.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>Input</td>
      <td>(Optional) Weight, corresponding to weightOptional in the formula.</td>
      <td><ul><li>Empty tensors are not supported. <li>When weightOptional is not empty, the data type is the same as that of input or is FLOAT. When biasOptional exists, the data type is the same as that of biasOptional. <li>When weightOptional is empty, you need to construct a tensor with shape [R1,...,Rj], data type being the same as that of input, and all data being 1. <li>The shape is the same as that of normalizedShape, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>(Optional) Bias.</td>
      <td><ul><li>Empty tensors are not supported. <li>When biasOptional is not empty, the data type is the same as that of input or is FLOAT. When weightOptional exists, the data type is the same as that of weightOptional. <li>When biasOptional is empty, no processing is performed. <li>The shape is the same as that of normalizedShape, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputMask</td>
      <td>Input</td>
      <td>Output mask.</td>
      <td><ul><li>The length is fixed at 3. <li>If the value is True, the output at the corresponding position is not empty.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInputOut</td>
      <td>Output</td>
      <td>(Optional) Output gradient of backpropagation, corresponding to `gradInputOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The 0th element of outputMask controls whether to output. If the 0th element of outputMask is True, there is output. The data type is the same as that of input. </li><li>The shape is the same as that of input, that is, [A1,...,Ai,R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOut</td>
      <td>Output</td>
      <td>(Optional) Gradient of the weight in backpropagation, corresponding to `gradWeightOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The 1st element of outputMask controls whether to output. If the 1st element of outputMask is True, there is output. The data type is the same as that of weightOptional. </li><li>The shape is the same as that of gradBiasOut, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut</td>
      <td>Output</td>
      <td>(Optional) Gradient of the bias in backpropagation, corresponding to `gradBiasOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The 2nd element of outputMask controls whether to output. If the 2nd element of outputMask is True, there is output. The data type is the same as that of weightOptional. </li><li>The shape is the same as that of gradWeightOut, that is, [R1,...,Rj].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
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

  - <term>Atlas inference series products</term> and <term>Atlas training series products</term>:
  
    The data type of `gradOut`, `input`, `mean`, `rstd`, `weightOptional`, `biasOptional`, `gradInputOut`, `gradWeightOut`, and `gradBiasOut` does not support BFLOAT16.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  
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
      <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="4">161001</td>
      <td>The passed gradOut, input, normalizedShape, mean, rstd, or outputMask is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[0] is True, and gradInputOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[1] is True, and gradWeightOut is a null pointer.</td>
    </tr>
    <tr>
      <td>outputMask[2] is True, and gradBiasOut is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="11">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="11">161002</td>
      <td>The data type of gradOut, input, mean, rstd, weightOptional (non-empty), or biasOptional (non-empty) is not supported.</td>
    </tr>
    <tr>
      <td>The shape of gradOut is different from that of input.</td>
    </tr>
    <tr>
      <td>normalizedShape has less than one dimension.</td>
    </tr>
    <tr>
      <td>The shape product of mean is not equal to the product of axis 0 to axis (len(input) - len(normalizedShape)) of input.</td>
    </tr>
    <tr>
      <td>The shape product of rstd is not equal to the product of axis 0 to axis (len(input) - len(normalizedShape)) of input.</td>
    </tr>
    <tr>
      <td>weightOptional is specified and the shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>biasOptional is specified and the shape is different from normalizedShape.</td>
    </tr>
    <tr>
      <td>The number of dimensions of input is less than that of normalizedShape.</td>
    </tr>
    <tr>
      <td>The shape of input is different from the shape of the corresponding dimension when right aligned with normalizedShape.</td>
    </tr>
    <tr>
      <td>The length of outputMask is not 3.</td>
    </tr>
    <tr>
      <td>The shape of gradOut, input, mean, rstd, weightOptional (non-empty), biasOptional (non-empty), gradInputOut (non-empty), gradWeightOut (non-empty), or gradBiasOut (non-empty) has more than eight dimensions or fewer than one dimension.</td>
    </tr>
  </tbody></table>

## aclnnLayerNormBackward

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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnLayerNormBackwardGetWorkspaceSize.</td>
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

- Shape constraints:

  The shape of gradOut, input, mean, rstd, weightOptional (non-empty), biasOptional (non-empty), gradInputOut (non-empty), gradWeightOut (non-empty), or gradBiasOut (non-empty) supports one to eight dimensions.

- Deterministic compute:
  
  **aclnnLayerNormBackward** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_layer_norm_backward.h"

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

int main()
{
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> xShape = {2, 2};
    std::vector<int64_t> meanShape = {2, 1};
    std::vector<int64_t> normShape = {2};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dwDeviceAddr = nullptr;
    void* dbDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclBoolArray* mask = nullptr;
    aclTensor* out = nullptr;
    aclTensor* dw = nullptr;
    aclTensor* db = nullptr;
    std::vector<float> dyHostData = {2, 3, 4, 5};
    std::vector<float> xHostData = {2, 3, 4, 5};
    std::vector<int64_t> normData = {2};
    std::vector<float> meanHostData = {2, 3};
    std::vector<float> rstdHostData = {4, 5};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> outHostData(4, 0);
    std::vector<float> dwHostData(2, 0);
    std::vector<float> dbHostData(2, 0);

    // Create a dy aclTensor.
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a normalizedShape aclIntArray.
    norm = aclCreateIntArray(normData.data(), 1);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    // Create a mean aclTensor.
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a rstd aclTensor.
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a bias aclTensor.
    ret = CreateAclTensor(biasHostData, normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an outputMask aclBoolArray.
    bool maskData[3] = {true, true, true};
    mask = aclCreateBoolArray(&(maskData[0]), 3);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a dw aclTensor.
    ret = CreateAclTensor(dwHostData, normShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a db aclTensor.
    ret = CreateAclTensor(dbHostData, normShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnLayerNormBackward.
    ret = aclnnLayerNormBackwardGetWorkspaceSize(
        dy, x, norm, mean, rstd, weight, bias, mask, out, dw, db, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnLayerNormBackward.
    ret = aclnnLayerNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(normShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dwDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("dw result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(normShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dbDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("db result[%ld] is: %f\n", i, resultData2[i]);
    }

    // 6. Release aclTensor, aclIntArray, and aclBoolArray. Modify the configuration based on the API definition.
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyBoolArray(mask);
    aclDestroyTensor(out);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);

    // 7. Release device resources.
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(dbDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
