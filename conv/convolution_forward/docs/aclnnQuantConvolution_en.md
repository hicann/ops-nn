# aclnnQuantConvolution

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Performs two-dimensional and three-dimensional convolution for per-channel quantization. The convolution process is the same as that of **aclnnConvolution**.

- Formula:
  Assume that the **input** shape is $(N, C_{\text{in}}, D, H, W)$, the **weight** shape is $(C_{\text{out}}, C_{\text{in}}, K_d, K_h, K_w)$, the **scale** shape is $(C_{\text{out}})$, the **bias** shape is $C_{\text{out}}$, and the **output** shape is $(N, C_{\text{out}}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}})$, where $N$ indicates the batch size, $C$ indicates the number of channels, $D$, $H$, and $W$ indicate the depth, height, and width of the sample, respectively, and $K_d$, $K_h$, and $K_w$ indicate the depth, height, and width of the convolution kernel, respectively. Then, the output is expressed as follows:

  $$
  \text{output}(N_i, C_{\text{out}_j}, D_{\text{out}}, H_{\text{out}}, W_{\text{out}}) = \left[\sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)\right] \times \text{scale}(C_{\text{out}_j}) + \text{bias}(C_{\text{out}_j})
  $$

  where $\star$ indicates convolution compute, which is based on dimension of the convolution input and the convolution type (atrous convolution or group convolution). $N$ indicates the batch size, $C$ indicates the number of channels, and $D$, $H$, and $W$ indicate the depth, height, and width, respectively. The formulas for computing the corresponding output dimensions are as follows:

  $$
  D_{\text{out}}=[(D + 2 \times padding[0] - dilation[0] \times (K_d - 1) - 1 ) / stride[0]] + 1 \\
  H_{\text{out}}=[(H + 2 \times padding[1] - dilation[1] \times (K_h - 1) - 1 ) / stride[1]] + 1 \\
  W_{\text{out}}=[(W + 2 \times padding[2] - dilation[2] \times (K_w - 1) - 1 ) / stride[2]] + 1
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantConvolutionGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantConvolution** is called to perform computation.

```cpp
aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(
    const aclTensor       *input,
    const aclTensor       *weight,
    const aclTensor       *bias,
    const aclTensor       *scale,
    const aclTensor       *offset,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    bool                   transposed,
    const aclIntArray     *outputPadding,
    int64_t                groups,
    int32_t                offsetx,
    const char            *roundMode,
    aclTensor             *output,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnQuantConvolution(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnQuantConvolutionGetWorkspaceSize

- **Parameters:**

  <table>
  <tr>
  <th style="width:170px">Name</th>
  <th style="width:120px">Input/Output</th>
  <th style="width:300px">Description</th>
  <th style="width:400px">Usage Notes</th>
  <th style="width:212px">Data Type</th>
  <th style="width:100px">Data Format</th>
  <th style="width:100px">Dimension (Shape)</th>
  <th style="width:145px">Non-contiguous Tensor</th>
  </tr>
  <td>input</td>
  <td>Input</td>
  <td>input in the formula, indicating convolution input.</td>
  <td><ul><li>The dimensions of input, weight, and output must be the same. </li><li>Empty tensors are supported. </li><li>Its data type and the data type of weight must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). </li><li>N ≥ 0, C ≥ 1, D ≥ 0, H ≥ 0, W ≥ 0.</li></ul></td>
  <td>INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight</td>
  <td>Input</td>
  <td>weight in the formula, indicating the convolution weight.</td>
  <td><ul><li>The C dimension of the shape must be the same as that of the input. </li><li>Empty tensors are supported. </li><li>Its data type and the data type of input must meet the type deduction rules (see <a href="../../../docs/en/context/deduction_relationship.md">Deduction Relationship</a>). </li><li>All dimensions must be greater than or equal to 1.</li></ul></td>
  <td>INT8, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias</td>
  <td>Input</td>
  <td>bias in the formula, indicating the convolution bias.</td>
  <td>One-dimensional and equal to the first dimension of weight.</td>
  <td>BFLOAT16, FLOAT16, FLOAT, INT32</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>scale</td>
  <td>Input</td>
  <td>scale in the formula, indicating the quantization parameter.</td>
  <td>One-dimensional and equal to the first dimension of weight.</td>
  <td>FLOAT, INT64, UINT64</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>offset</td>
  <td>Input</td>
  <td>Reserved quantization parameter.</td>
  <td>This parameter is not supported in the current version, and nullptr can be passed.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>stride</td>
  <td>Input</td>
  <td>Convolutional scanning stride.</td>
  <td><ul><li>In the two-dimensional scenario, the array length is 2. In the three-dimensional scenario, the array length is 3. </li><li>strideH and strideW must be in the range of [1, 63]. </li><li>In the conv3d scenario, strideD must be in the range of [1,1000000].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding</td>
  <td>Input</td>
  <td>Padding added to input.</td>
  <td><ul><li>The value must be greater than or equal to 0. </li><li>paddingH and paddingW must be in the range of [0,255]. </li><li>In the conv3d scenario, paddingD must be in the range of [0,1000000].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation</td>
  <td>Input</td>
  <td>Dilation between elements in the convolution kernel.</td>
  <td><ul><li>In the two-dimensional scenario, the array length is 2. In the three-dimensional scenario, the array length is 3. </li><li>The value must be greater than 0. </li><li>dilationH and dilationW must be in the range of [1,255]. </li><li>In the conv3d scenario, dilationD must be in the range of [1,1000000].</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>transposed</td>
  <td>Input</td>
  <td>Reserved parameter, indicating whether the convolution is quantized and then transposed.</td>
  <td>This parameter is not supported in the current version, and false can be passed.</td>
  <td>BOOL</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>outputPadding</td>
  <td>Input</td>
  <td>Reserved parameter, indicating the padding for all output edges in the case of transposed convolution.</td>
  <td>For non-transposed convolution, ignore this attribute. This parameter is not supported in the current version, and nullptr can be passed.</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>groups</td>
  <td>Input</td>
  <td>Number of block links from the input channel to the output channel.</td>
  <td>The value must be greater than or equal to 1 and meet the following condition: C dimension of groups × weight = C dimension of input.</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>offsetx</td>
  <td>Input</td>
  <td>Quantization factor.</td>
  <td>[-128,127] or 0.</td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>roundMode</td>
  <td>Input</td>
  <td>Rounding mode.</td>
  <td>rint, round, or nullptr.</td>
  <td>CHAR*</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>output</td>
  <td>Output</td>
  <td>out in the formula, indicating convolution output.</td>
  <td><ul><li>Its shape complies with the convolution derivation rules. </li><li>Empty tensor output is not supported. </li><li>Number of channels is equal to the first dimension of the weight, and other dimensions are greater than or equal to 0.</li></ul></td>
  <td>BFLOAT16, FLOAT16, FLOAT, FLOAT8_E4M3FN, HIFLOAT8</td>
  <td>NCHW, NCDHW</td>
  <td>4-5</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>Output</td>
  <td>Size of the workspace required to be allocated on the device.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>Output</td>
  <td>Operator executor, containing the operator computation process.</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed; width: 1122px"><colgroup>
  <col style="width: 286px">
  <col style="width: 125px">
  <col style="width: 711px">
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
      <td>The input parameter is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>The data type or format of input, weight, bias, scale, offset, or output is not supported.</td>
    </tr>
    <tr>
      <td>The input shape of stride, padding, or dilation is incorrect.</td>
    </tr>
    <tr>
      <td>The input of groups is incorrect.</td>
    </tr>
    <tr>
      <td>The input shapes of scale and bias are incorrect.</td>
    </tr>
    <tr>
      <td>The shape of output does not meet the infershape result.</td>
    </tr>
    <tr>
      <td>Any dimension whose value is 0 in a tensor passed by input does not meet requirements.</td>
    </tr>
    <tr>
      <td>The shape of input after padding is less than that of weight after dilation (if dilation is greater than 1).</td>
    </tr>
    <tr>
      <td>The numbers of channels for weight and input do not meet the requirements.</td>
    </tr>
    <tr>
      <td>When the value of stride or dilation is less than 0, the requirements are not met.</td>
    </tr>
    <tr>
      <td>The current processor does not support convolution.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>Internal API verification error, usually caused by unsupported input data or attribute specifications.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>An error occurred when the NPU runtime API is called, for example, due to unsupported SocVersion.</td>
    </tr>
  </tbody></table>

## aclnnQuantConvolution

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 123px">
  <col style="width: 855px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by the first-phase API aclnnQuantConvolutionGetWorkspaceSize.</td>
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

- Deterministic compute
  - **aclnnQuantConvolution** defaults to a deterministic implementation.

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width:150px">
    <col style="width:700px">
    </colgroup>
   <thead>
    <tr>
     <th>Constraint Type</th>
     <th><term>Atlas A2 Training Series Products, Atlas A2 Inference Series Products</term>, <term>Atlas A3 Training Series Products, and Atlas A3 Inference Series Products</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">input, weight</th>
     <td>
        <ul>The data types of input and weight cannot be FLOAT8_E4M3FN or HIFLOAT8, and their data formats cannot be NCHW.</ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>
          The bias data type cannot be INT32, which will be converted to FLOAT for computation.
     </td>
   </tr>
   <tr>
     <th scope="row">scale</th>
     <td>
          The scale data type cannot be INT64 or UINT64.
     </td>
   </tr>
   <tr>
     <th scope="row">padding</th>
     <td>
          The length of the padding array must be 3.
     </td>
   </tr>
   <tr>
     <th scope="row">groups</th>
     <td>
          The value of groups must be 1.
     </td>
   </tr>
   <tr>
     <th scope="row">offsetx</th>
     <td>
          offsetx is not supported in the current version. 0 can be passed.
     </td>
   </tr>
   <tr>
     <th scope="row">roundMode</th>
     <td>
        <ul>
          roundMode is not supported in the current version. nullptr can be passed.
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">output</th>
     <td>
          The output data type can be BFLOAT16 or FLOAT16, and the data format can only be NCDHW.
     </td>
   </tr>
   <tr>
     <th scope="row">Other Constraints</th>
     <td>
        <ul>
          <li>The operator can be called only in inference scenarios.</li>
          <li>Only forward three-dimensional convolution is supported.</li>
          <li>Each dimension of each group of tensors in input, weight, bias, and scale must be less than 1000000.</li>
        </ul>
     </td>
     </td>
   </tr>
   </tbody>
  </table>

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
For different product models, use different main functions.

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_convolution.h"

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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCDHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnQuantConvolutionTest(int32_t deviceId, aclrtStream& stream, std::vector<aclDataType> dtypesInfo)
{
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API definition.
  std::vector<int64_t> shapeInput = {2, 2, 32, 32, 32};
  std::vector<int64_t> shapeWeight = {2, 2, 3, 3, 3};
  std::vector<int64_t> shapeScale = {2};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 32, 32, 32};
  std::vector<int64_t> convStrides;
  std::vector<int64_t> convPads;
  std::vector<int64_t> convOutPads;
  std::vector<int64_t> convDilations;

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataScale = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* scale= nullptr;
  aclTensor* bias= nullptr;
  aclTensor* result = nullptr;
  std::vector<int8_t> inputData(GetShapeSize(shapeInput), 1);
  std::vector<int8_t> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> scaleData(GetShapeSize(shapeScale), 1);
  std::vector<uint16_t> outputData(GetShapeSize(shapeResult), 1);
  convStrides = {1, 1, 1};
  convPads = {1, 1, 1};
  convOutPads = {1, 1, 1};
  convDilations = {1, 1, 1};
  aclDataType inputDtype = dtypesInfo[0];
  aclDataType weightDtype = dtypesInfo[1];
  aclDataType biasDtype = dtypesInfo[2];
  aclDataType scaleDtype = dtypesInfo[3];
  aclDataType outputDtype = dtypesInfo[4];
  // Create an input aclTensor.
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, inputDtype, &input);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create a weight aclTensor.
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, weightDtype, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a scale.
  ret = CreateAclTensorND(scaleData, shapeScale, &deviceDataScale, scaleDtype, &scale);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(scale, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataScalePtr(deviceDataScale, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // Create a bias.
  ret = CreateAclTensorND(biasData, shapeBias, &deviceDataBias, biasDtype, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // Create an out aclTensor.
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, outputDtype, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  aclIntArray *strides = aclCreateIntArray(convStrides.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridesPtr(strides, aclDestroyIntArray);
  CHECK_FREE_RET(strides != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *pads = aclCreateIntArray(convPads.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> padsPtr(pads, aclDestroyIntArray);
  CHECK_FREE_RET(pads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *outPads = aclCreateIntArray(convOutPads.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> outPadsPtr(outPads, aclDestroyIntArray);
  CHECK_FREE_RET(outPads != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  aclIntArray *dilations = aclCreateIntArray(convDilations.data(), 3);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationsPtr(dilations, aclDestroyIntArray);
  CHECK_FREE_RET(dilations != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first-phase API of aclnnConvolution.
  ret = aclnnQuantConvolutionGetWorkspaceSize(input, weight, bias, scale, nullptr, strides, pads, dilations,
                                              false, outPads, 1, 0, nullptr, result, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // Call the second-phase API of aclnnConvolution.
  ret = aclnnQuantConvolution(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolution failed. ERROR: %d\n", ret); return ret);

  // 4. (Fixed writing) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the configuration based on the API definition.
  auto size = GetShapeSize(shapeResult);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}
```

- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
```
int main() {
  // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  std::vector<aclDataType> dtypesInfo = {aclDataType::ACL_INT8, aclDataType::ACL_INT8, aclDataType::ACL_FLOAT,
    aclDataType::ACL_FLOAT, aclDataType::ACL_BF16}; // Data types of input, weight, bias, scale, and output, respectively.
  auto ret = aclnnQuantConvolutionTest(deviceId, stream, dtypesInfo);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantConvolutionTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
