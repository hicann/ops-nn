# aclnnConvDepthwise2d

 [📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                      |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>      |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>      |    √     |
|  <term>Atlas 200I/500 A2 inference products</term>                     |     ×    |
|  <term>Atlas inference products</term>                             |     ×    |
|  <term>Atlas training products</term>                             |     ×    |

## Function

- Function: DepthwiseConv2D is a two-dimensional depthwise convolution operation. In this operation, each input channel is convolved with an independent kernel (called a depthwise kernel).

- Formula:

  Assume that the shape of the input self is $(N, C_{\text{in}}, H, W)$, and the shape of the output out is $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$. The output of each convolution kernel is represented as follows:

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{weight}(C_{\text{out}_j}, C_{\text{in}_j}) \star \text{self}(N_i, C_{\text{in}_j})
  $$

  Where $\star$ denotes the convolution operation, $N$ is the batch size, $C$ is the number of channels, and $W$ and $H$ represent the width and height, respectively.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. First, aclnnConvDepthwise2dGetWorkspaceSize is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, aclnnConvDepthwise2d is called to perform computation.

```cpp
aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclIntArray     *kernelSize,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    aclTensor             *out,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvDepthwise2d(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvDepthwise2dGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:280px">Description</th>
  <th style="width:350px">Usage</th>
  <th style="width:200px">Data Type</th>
  <th style="width:120px">Format</th>
  <th style="width:130px">Dimension (shape) </th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>self (aclTensor*) </td>
  <td>Input</td>
  <td>self in the formula, indicating the convolution input.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). </li><li>The shape is <b>(N,C<sub>in</sub>,H<sub>in</sub>,W<sub>in</sub>)</b>. </li><li>N≥0, C≥1, H≥0, W≥0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*) </td>
  <td>Input</td>
  <td>weight in the formula, indicating the convolution weight.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of self must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). </li><li>The shape is C<sub>out</sub>,C<sub>in</sub>/groups,K<sub>H</sub>,K<sub>W</sub>. </li><li>The first dimension of weight must be an integer multiple of the number of channels in self, and the second dimension must be 1. </li><li>All dimensions must be greater than or equal to 1. The H and W dimensions must be smaller than those of self.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>kernelSize (aclIntArray*) </td>
  <td>Input</td>
  <td>Convolution kernel size.</td>
  <td><ul>
  <li>Tuple of type (INT64, INT64). </li><li>The values correspond to the H and W dimension sizes of weight.</li></ul>
  </td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>bias (aclTensor*) </td>
  <td>Input</td>
  <td>bias in the formula, indicating the convolution bias.</td>
  <td><ul><li>The shape is (C<sub>out</sub>). </li><li>bias is a 1D tensor whose length equals the first dimension of weight.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride (aclIntArray*) </td>
  <td>Input</td>
  <td>Convolution stride.</td>
  <td><ul><li>The array length must be equal to the dimension of self minus two. </li><li>strideH and strideW must be in the range [1, 63].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding (aclIntArray*) </td>
  <td>Input</td>
  <td>Padding added to self.</td>
  <td><ul><li>The array length must be equal to the dimension of self minus two. </li><li>paddingH and paddingW must be in the range [0, 255].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation (aclIntArray*) </td>
  <td>Input</td>
  <td>Spacing between elements in the convolution kernel.</td>
  <td><ul><li>The array length must be equal to the dimension of self minus two. </li><li>dilationH and dilationW must be in the range [1, 255].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>out (aclTensor*) </td>
  <td>Output</td>
  <td>out in the formula, indicating the convolution output.</td>
  <td><ul><li>Empty tensors are supported. </li><li>shape is (N,C<sub>out</sub>,H<sub>out</sub>,W<sub>out</sub>). </li><li>The number of channels is equal to the first dimension of weight. The H and W dimensions are no less than 0, and all other dimensions are no less than 1.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType (int8_t) </td>
  <td>Input</td>
  <td>Computation logic to be used by the Cube unit.</td>
  <td><ul><li>0 (KEEP_DTYPE): The input data type is retained for computation. </li></ul><ul><li>1 (ALLOW_FP32_DOWN_PRECISION): The FLOAT precision can be reduced for computation, improving performance. </li></ul><ul><li>2 (USE_FP16): The FLOAT16 precision is used for computation. </li></ul><ul><li>3 (USE_HF32): The HFLOAT32 (mixed precision) is used for computation.</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize (uint64_t*) </td>
  <td>Output</td>
  <td>Size of the workspace required to be allocated on the device.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor (aclOpExecutor**) </td>
  <td>Output</td>
  <td>Operator executor, containing the operator computation flow.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Return Value**

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

  The first-phase API implements input parameter verification. The following errors may be thrown.
  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
  <tr>
  <td><strong>Return Value</strong></td>
  <td>Error Code</td>
  <td><strong>Description</strong></td>
  </tr></thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The input pointer is null.</td>
  </tr>
  <tr>
  <td rowspan="6" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="6" align="left">161002</td>
  <td align="left">The data types and formats of self, weight, bias, and out are not supported.</td>
  </tr>
  <tr><td align="left">The data types of self, weight, and out are inconsistent.</td></tr>
  <tr><td align="left">The input shape of stride, padding, or dilation is incorrect.</td></tr>
  <tr><td align="left">The number of channels in weight and self does not meet the requirements.</td></tr>
  <tr><td align="left">The shape of out does not meet the infer_shape result.</td></tr>
  <tr><td align="left">self, weight, bias, and out are empty tensor inputs or outputs that are not supported.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">Internal verification error of the API. This is usually caused by the fact that the specifications of the input data or attributes are not supported.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">The API fails to call the NPU runtime interface, for example, soc_version is not supported.</td>
  </tr>
  </table>

## aclnnConvDepthwise2d

- **Parameters**

  <table>
  <tr>
  <th style="width:120px">Parameter</th>
  <th style="width:80px">Input/Output</th>
  <th>Description</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>Input</td>
  <td>Address of the workspace to be allocated on the device.</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Input</td>
  <td>Workspace size allocated on the device, which is obtained by the first API aclnnConvDepthwise2dGetWorkspaceSize.</td>
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
  </table>

- **Return Value**

  aclnnStatus: status code. For details, see <a href="../../../docs/en/context/aclnn_return_code.md">aclnn Return Code</a>.

## Constraints

- Deterministic computation:
  - **aclnnConvDepthwise2d** defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width:150px">
  <col style="width:500px">
  <col style="width:500px">
  </colgroup>
  <thead>
  <tr>
    <th>Restriction Type</th>
    <th><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term></th>
    <th>Ascend 950PR/Ascend 950DT</th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <th scope="row">self, weight</th>
    <td>
      <ul>
        <li>The data type of self or weight cannot be HIFLOAT8.</li>
        <li>The number of channels in self must be less than or equal to 65535.</li>
      </ul>
    </td>
    <td> - </td>
  </tr>
  <tr>
    <th scope="row">bias</th>
    <td>
      The data type of bias cannot be HIFLOAT8 or FLOAT8_E4M3FN. The data type is the same as that of input and weight.
    </td>
    <td>
      When the self data type is HIFLOAT8, the bias data type will be converted to FLOAT for computation.
    </td>
  </tr>
  <tr>
    <th scope="row">cubeMathType</th>
    <td>
      <ul>
        <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), the input of type FLOAT can be converted to HFLOAT32 for computation.</li>
        <li>If the value is 2 (USE_FP16), this option is not supported when the input is of type BFLOAT16.</li>
        <li>If the value is 3 (USE_HF32), the input of type FLOAT is converted to HFLOAT32 for computation.</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), the input of type FLOAT can be converted to HFLOAT32 for computation.</li>
        <li>If the value is 2 (USE_FP16), this option is not supported when the input is of type BFLOAT16.</li>
        <li>If the value is 3 (USE_HF32), the input of type FLOAT can be converted to HFLOAT32 for computation.</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th scope="row">kernelSize constraints</th>
    <td colspan="2">
      The value corresponds to the sizes of the H and W dimensions of weight.
    </td>
  </tr>
  <tr>
    <th scope="row">Other constraints</th>
    <td colspan="2">
      For tensors in self, weight, and bias, the size of each dimension must not exceed 1,000,000.
    </td>
  </tr>
  </tbody>
</table>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
    shape_size *= i;
  }
  return shape_size;
}

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

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvDepthwise2dTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Customize error handling based on your requirements.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> shapeSelf = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {2, 1, 1, 1};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 2, 2};

  void* deviceDataSelf = nullptr;
  void* deviceDataWeight = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* stride = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* dilation = nullptr;

  std::vector<float> selfData(GetShapeSize(shapeSelf), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outData(GetShapeSize(shapeResult), 1);
  std::vector<int64_t> kernelSizeData = {1, 1};
  std::vector<int64_t> strideData = {1, 1};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};

  // Create a self aclTensor.
  ret = CreateAclTensor(selfData, shapeSelf, &deviceDataSelf, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataSelfPtr(deviceDataSelf, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataWeight, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataWeightPtr(deviceDataWeight, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a bias aclTensor.
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataBias, aclDataType::ACL_FLOAT, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  stride = aclCreateIntArray(strideData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridePtr(stride, aclDestroyIntArray);
  CHECK_FREE_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  padding = aclCreateIntArray(paddingData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingPtr(padding, aclDestroyIntArray);
  CHECK_FREE_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationPtr(dilation, aclDestroyIntArray);
  CHECK_FREE_RET(dilation != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvDepthwise2d.
  ret = aclnnConvDepthwise2dGetWorkspaceSize(self, weight, kernelSize, bias, stride, padding, dilation, result, 1,
                                             &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvDepthwise2d.
  ret = aclnnConvDepthwise2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2d failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvDepthwise2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
