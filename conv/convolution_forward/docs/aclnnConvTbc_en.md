# aclnnConvTbc

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

- Description: Implements 1D convolution where the input and output dimensions are T (temporal or spatial), B (batch), and C (channel).
- Formula:
  Assume the input **self** has shape ($H_{\text{in}},N,C_{\text{in}}$) and the output **out** has shape ($H_{\text{out}},N,C_{\text{out}}$). The output is computed as:

  $$
  out_{N_i,C_{out j}} = bias(C_{out j}) + \sum_{k = 0}^{C_{in} - 1} weight(k, C_{out j}) \cdot self(N_i, k)
  $$

  Where $N$ denotes the batch size, $C$ denotes the number of channels, and $H$ denotes the temporal or spatial dimension.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnConvTbcGetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator execution process, and then call aclnnConvTbc to perform the computation.

```cpp
aclnnStatus aclnnConvTbcGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const int64_t          pad,
    aclTensor             *out,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvTbc(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnConvTbcGetWorkspaceSize

- **Parameters**

  <table>
  <tr>
  <th style="width:220px">Parameter Name</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:400px">Usage Notes</th>
  <th style="width:212px">Data Type</th>
  <th style="width:100px">Data Format</th>
  <th style="width:100px">Shape</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <tr>
  <td>self (aclTensor*) </td>
  <td>Input</td>
  <td>self in the formula, indicating the convolution input.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). <li>The shape is N,C<sub>in</sub>,H<sub>in</sub>. </li></li><li>N≥0, C≥1, H≥0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight (aclTensor*) </td>
  <td>Input</td>
  <td>weight in the formula, indicating the convolution weight.</td>
  <td><ul><li>Empty tensors are supported. </li><li>Its data type and the data type of self must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). </li><li>The shape is C<sub>out</sub>,C<sub>in</sub>,K. </li><li>All dimensions must be greater than or equal to 1.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias (aclTensor*) </td>
  <td>Input</td>
  <td>bias in the formula, indicating the convolution bias.</td>
  <td><ul><li> The data type is the same as that of self and weight. </li><li> one-dimensional and equal to the first dimension of the weight. A null pointer is not allowed.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>pad (int64_t) </td>
  <td>Input</td>
  <td>Number of padding elements to add on both sides of the H dimension.</td>
  <td>The value must be in the range [0, 255].</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>out (aclTensor*) </td>
  <td>Output</td>
  <td>out in the formula, indicating the convolution output.</td>
  <td><ul><li>Empty tensors are supported. </li><li>The data type must be the same as that of self. </li><li>shape is (N,C<sub>out</sub>,H<sub>out</sub>). </li><li>The number of channels equals the first dimension of weight, and all other dimensions must be greater than or equal to 0.</li></ul></td>
  <td>FLOAT, FLOAT16, BFLOAT16, HIFLOAT8</td>
  <td>ND, NCL</td>
  <td>3</td>
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
  <td>Error code</td>
  <td><strong>Description</strong></td>
  </tr></thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">The input pointer is null.</td>
  </tr>
  <tr>
  <td rowspan="9" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="9" align="left">161002</td>
  <td align="left">The data types and formats of self, weight, bias, and out are not supported.</td>
  </tr>
  <tr><td align="left">The data types of self, weight, and out are inconsistent.</td></tr>
  <tr><td align="left">The shape of out does not meet the inference shape result.</td></tr>
  <tr><td align="left">The shape of out contains a value less than 0.</td></tr>
  <tr><td align="left">The dimension of self is not 3.</td></tr>
  <tr><td align="left">The dimension of weight is not 3.</td></tr>
  <tr><td align="left">The dimension of bias is not 1.</td></tr>
  <tr><td align="left">The second dimension of self is not equal to that of weight.</td></tr>
  <tr><td align="left">The value of bias is not equal to the first dimension of weight.</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">Internal API verification error. This is usually caused by the fact that the specifications of the input data or attributes are not supported.</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">An exception occurred when the API calls the NPU runtime API. For example, the SoC version is not supported.</td>
  </tr>
  </table>

## aclnnConvTbc

- **Parameters**

  <table>
  <tr>
  <th style="width:120px">Parameter Name</th>
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
  <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnConvTbcGetWorkspaceSize API.</td>
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
  - **aclnnConvTbc** defaults to a deterministic implementation.

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
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
          <li>The data type of self or weight cannot be HIFLOAT8. The N, C, and L dimensions must be greater than or equal to 0.</li>
          <li>The N and C dimensions of weight must be greater than or equal to 0.</li>
        </ul>
     </td>
     <td>
        <ul>
           <li>The self and weight support the N and C dimensions greater than or equal to 0, and the L dimension greater than or equal to 0. (The scenario where the L dimension is 0 is supported only when the L dimension derived from out is also 0.)</li>
           <li>The N dimension of weight must be greater than or equal to 0. (The scenario where the N dimension is 0 is supported only when the N dimension of bias and the C dimension of out are also 0.) The size of the C dimension is the same as that of the C dimension of self. The size of the L dimension must be within the range of [1, 255].</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>-</td>
     <td>
        When the data type of self is HIFLOAT8, the data type of bias will be converted to FLOAT for computation.
     </td>
   </tr>
   <tr>
     <th scope="row">out</th>
     <td>
        The N, C, and L dimensions of out must be greater than or equal to 0. (The value 0 is supported only when the N, C, or L dimension of self is 0.)
     </td>
     <td>
        <ul>
          <li>The N dimension can be greater than or equal to 0, and the C dimension can be greater than or equal to 0. (The scenario where the C dimension is 0 is supported only when the N dimension of weight is 0.)</li>
          <li>The L dimension can be greater than or equal to 0. (The scenario where the L dimension is 0 is supported only when the L dimension of self is 0.)</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>
        <ul>
          <li>If the value is 0 (KEEP_DTYPE), the input is FLOAT, which is not supported currently.</li>
          <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), the input can be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2 (USE_FP16), this option is not supported when the input is BFLOAT16.</li>
          <li>If the value is 3 (USE_HF32), the input is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
     <td>
        <ul>
          <li>If the value is 1 (ALLOW_FP32_DOWN_PRECISION), the input of type FLOAT can be converted to HFLOAT32 for computation.</li>
          <li>If the value is 2 (USE_FP16), this option is not supported when the input is of type BFLOAT16.</li>
          <li>If the value is 3 (USE_HF32), the input of type FLOAT is converted to HFLOAT32 for computation.</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
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
                    aclDataType dataType, aclTensor** tensor, aclFormat dataFormat) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvTbcTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // Customize error handling based on your requirements.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct inputs and outputs based on API definitions.
  std::vector<int64_t> shapeInput = {2, 2, 3};
  std::vector<int64_t> shapeWeight = {3, 3, 4};
  std::vector<int64_t> shapeBias = {4};
  std::vector<int64_t> shapeResult = {2, 2, 4};

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataC = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);

  // Create an input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a bias aclTensor.
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataC, aclDataType::ACL_FLOAT, &bias, aclFormat::ACL_FORMAT_ND);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataCPtr(deviceDataC, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a result aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvTbc.
  ret = aclnnConvTbcGetWorkspaceSize(input, weight, bias, 1, result, 1, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvTbc.
  ret = aclnnConvTbc(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbc failed. ERROR: %d\n", ret); return ret);

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
  auto ret = aclnnConvTbcTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
