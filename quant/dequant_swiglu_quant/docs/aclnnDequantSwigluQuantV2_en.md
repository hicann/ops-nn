# aclnnDequantSwigluQuantV2

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     √    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     ×    |
|  <term>Atlas training products</term>   |     ×    |

## Function

- API function: Adds dequant and quant operations before and after the Swish gating linear unit activation function to implement the DequantSwigluQuant computation of x. Compared with [aclnnDequantSwigluQuant](aclnnDequantSwigluQuant_en.md), this API has two types of new parameters: (1) Three input parameters used by the Ascend 950 chip are added: dstType, roundModeOptional, and activateDim. (2) On the Atlas A2 and Atlas A3 chips, four parameters are added for the variant SwiGLU used by GPT-OSS: swigluMode, clampLimit, gluAlpha, and gluBias. When this API is used on the Ascend 950 chip, default values need to be set for these four parameters. Select the appropriate API as required.
- The formula for swigluMode = 0 is as follows: 

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  where A<sub>i</sub> indicates the first half of dequantOut<sub>i</sub>, and B<sub>i</sub> indicates the second half of dequantOut<sub>i</sub>.

- The formula for swigluMode = 1 is as follows: 

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$
  
  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  x\_glu indicates the even-indexed part of dequantOut<sub>i</sub>, and x\_linear indicates the odd-indexed part of dequantOut<sub>i</sub>.

## Prototype

Each operator has <a href="../../../docs/en/context/two_phase_api.md">two-phase API calls</a>. You must call aclnnDequantSwigluQuantV2GetWorkspaceSize to obtain the workspace size required for computation and the executor that contains the operator computation process, and then call aclnnDequantSwigluQuantV2 to perform the computation.

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightScaleOptional,
  const aclTensor *activationScaleOptional,
  const aclTensor *biasOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  int64_t          dstType,
  char            *roundModeOptional,
  int64_t          activateDim,
  int64_t          swigluMode,
  double           clampLimit,
  double           gluAlpha,
  double           gluBias,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDequantSwigluQuantV2GetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 104px">
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
      <td>x (aclTensor*) </td>
      <td>Input</td>
      <td>Input data to be processed, corresponding to x in the formula.</td>
      <td><ul><li>The shape is [X1, X2,..., Xn, 2H]. The shape must be at least 2-dimensional and cannot exceed 8 dimensions. </li><li>The dimension of activateDim corresponding to x must be a multiple of 2. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Only 2D input is supported, and the data type can be INT32 or BFLOAT16.</li></ul></td>
      <td>FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
     <tr>
      <td>weightScaleOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Dequantization scale of the weight.</td>
      <td><ul><li>The shape can be 1D or 2D, and is represented as [2H] or [groupNum, 2H]. The value 2H must be the same as the last dimension of x. </li><li>Optional. A null pointer can be passed. When groupIndexOptional is a null pointer, the shape is [2H]. When groupIndexOptional is not a null pointer, the shape is [groupNum, 2H].</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
     <tr>
      <td>activationScaleOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Dequantization scale of the activation function.</td>
      <td><ul><li>Dequantization scale of the activation function. </li><li>The shape is [X1,X2,...Xn]. The shape must be a 1D to 7D tensor, and the number of dimensions is one less than that of x. In addition, the shape must be consistent with that of x in the corresponding dimension. </li><li>Optional. A null pointer can be passed.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
    </tr>
      <tr>
      <td>biasOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Bias of Matmul, corresponding to biasOptional in the formula.</td>
      <td><ul><li>The shape can be 1D or 2D, and is represented as [2H] or [groupNum, 2H]. The value 2H must be consistent with the last dimension of x. If groupIndexOptional is a null pointer, the shape is [2H]. Otherwise, the shape is [groupNum, 2H]. </li><li>Optional. A null pointer can be passed.</li></ul></td>
      <td>FLOAT, FLOAT16, BFLOAT16, INT32</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantScaleOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Quantization scale, corresponding to quantScaleOptional in the formula.</td>
      <td><ul><li>When quantModeOptional is static, the shape is 1-dimensional and the value is 1. The shape is represented as shape[1]. </li><li>When quantModeOptional is dynamic, the shape is 1-dimensional or 2-dimensional. The shape is represented as [H], [2H], or [groupNum, H]. </li><li>When groupIndexOptional is a null pointer and activateDim is the last axis, the shape is [H]. </li><li>When groupIndexOptional is not a null pointer and activateDim is the last axis, the shape is [groupNum, H]. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When quantModeOptional is static, the shape is represented as [groupNum] or [groupNum, H]. When quantModeOptional is dynamic, the shape is represented as [groupNum] or [groupNum, H].</li></ul></td>
      <td>FLOAT, FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
       <tr>
      <td>quantOffsetOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Quantization offset.</td>
      <td><ul><li>If quant_mode is set to dynamic, quantOffset is not required. In static quantization, quantOffset is required, and its data type must be the same as that of quantScale.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>x</td>
    </tr>
      <tr>
      <td>groupIndexOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Group index required for MoE grouping.</td>
      <td><ul><li>The shape supports 1D or 2D tensors. The shape is [groupNum] or [groupNum, 2], and groupNum is greater than or equal to 1. </li><li>Optional. A null pointer can be passed. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Only 1D [groupNum] is supported. Null pointers are not supported.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1 or 2</td>
      <td>x</td>
    </tr>
      <tr>
      <td>activateLeft (bool) </td>
      <td>Input</td>
      <td>Whether to apply SwiGLU activation to the left half of the input.</td>
      <td><ul><li>If the value is false, activation is performed on the right part of the input. If swigluMode is set to 1, activateLeft must be set to true.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>quantModeOptional (char*) </td>
      <td>Input</td>
      <td>Indicates whether to use dynamic quantization or static quantization.</td>
      <td><ul><li>The value can be "dynamic" or "static". </li><li>A null pointer can be passed. If a null pointer is passed, "static" is used by default.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType (int64_t) </td>
      <td>Input</td>
      <td>Indicates the data type of the output y.</td>
      <td><ul><li>The value of dstType can be 2, 34, 35, 36, 40, or 41, corresponding to INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1 and FLOAT4_E1M2 respectively. </li><li>For <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>, only 2-INT8 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>roundModeOptional (char*) </td>
      <td>Input</td>
      <td> indicates the rounding mode of the output y result.</td>
      <td><ul><li>Value range: ["rint", "round", "floor", "ceil", "trunc"]. </li><li>When the data type of output y is INT8, FLOAT8_E5M2, or FLOAT8_E4M3FN, only the "rint" mode is supported. </li><li>When the data type of output y is HIFLOAT8, only the "round" mode is supported. </li><li>Null pointers are supported. If a null pointer is passed, "rint" is used by default. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Only the null pointer or "rint" is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateDim (int64_t) </td>
      <td>Input</td>
      <td> indicates the split axis selected for swish calculation.</td>
      <td><ul><li>The value range of activateDim is [-xDim, xDim - 1], where xDim indicates the dimension of input x. </li><li> When activateDim does not correspond to the tail axis of x, groupIndexOptional is not allowed. </li><li> When activateDim does not correspond to the tail axis of x, quantModeOptional supports only static. </li><li><term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: Only -1 is supported.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>swigluMode (int64_t) </td>
      <td>Input</td>
      <td>Computing mode of swiglu.</td>
      <td><ul><li>Value range: [0, 1]. </li><li>0: indicates the traditional swiglu computing mode. </li><li>1: indicates the swiglu variant, which uses the odd-even block partitioning mode and supports clamp_limit, activation coefficient, and bias. The value 0 indicates that the swiglu variant is not used, and the value 1 indicates that the swiglu variant is used. The default value is 0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clampLimit (double) </td>
      <td>Input</td>
      <td>Threshold used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional. </li><li>It is used to clip the input. The value must be greater than 0 and less than infinity to prevent the swiglu computing stability from being affected by excessively large values. The default value is 7.0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluAlpha (double) </td>
      <td>Input</td>
      <td>Indicates the parameters used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional. </li><li>It is used to adjust the scaling of the linear part in the glu activation function. The default value is 1.702.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluBias (double) </td>
      <td>Input</td>
      <td>Indicates the bias parameter used by the swiglu variant.</td>
      <td><ul><li>This parameter is optional. </li><li>It is used to add an offset to the linear calculation of swiglu. The default value is 1.0.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>yOut (aclTensor*) </td>
      <td>Output</td>
      <td>-</td>
      <td><ul><li>When activateDim corresponds to the last axis of x, the shape is [X1,X2,...Xn,H]. </li><li>When activateDim does not correspond to the last axis of x, the shape is [X1,X2,...,XactivateDim / 2,...,2H]. </li><li>When the data type of yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2. </li><li>When activateDim is not the last axis of x, the last axis of yOut must be less than 5120.</li></ul></td>
      <td>INT8, HIFLOAT8, FLOAT8_E5M2, FLOAT8_E4M3FN, FLOAT4_E2M1, FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>scaleOut (aclTensor*) </td>
      <td>Output</td>
      <td>-</td>
      <td><ul><li>When activateDim is the last axis of x, the shape is [X1, X2,..., Xn]. </li><li>When activateDim is not the last axis of x, the shape is [X1, X2,..., XactivateDim/2,..., Xn]. </li><li>When quantModeOptional is static, scaleOut is not calculated.</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**) </td>
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

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td><ul><li>The input x, yOut, or scaleOut is a null pointer. </li><li>When the data type of x is int32, weightScaleOptional is a null pointer. </li><li> When quantModeOptional is set to static, quantScaleOptional is a null pointer.</li></ul></td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>The input or output data type is not supported.</td>
    </tr>
    <tr>
      <td>The input or output parameter dimension is not supported.</td>
    </tr>
    <tr>
      <td>quantModeOptional is not within the specified range.</td>
    </tr>
    <tr>
      <td>dstType is not within the specified range.</td>
    </tr>
     <tr>
      <td>The value of roundModeOptional is not within the specified range.</td>
    </tr>
    <tr>
      <td>The value of activateDim is not within the specified range.</td>
    </tr>
    <tr>
      <td>weightScaleOptional, activationScaleOptional, biasOptional, quantScaleOptional,
                                           The shape of groupIndexOptional and x do not meet the constraints.</td>
    </tr>
  </tbody></table>

## aclnnDequantSwigluQuantV2

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
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnDequantSwigluQuantV2GetWorkspaceSize API.</td>
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

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The default deterministic implementation of aclnnDequantSwigluQuantV2 is used.

- The dimension of activateDim corresponding to the input x must be a multiple of 2, and the number of dimensions of x must be greater than 1.
- If the data type of the input x is INT32, weightScaleOptional cannot be null. If the data type of the input x is not INT32, weightScaleOptional cannot be input; a null pointer must be passed.
- If the data type of the input x is not INT32, activationScaleOptional cannot be input and a null pointer is passed.
- If the data type of the input x is not INT32, biasOptional cannot be input and a null pointer is passed.
- When the output data type of yOut is FLOAT4_E2M1 or FLOAT4_E1M2, the last dimension of yOut must be a multiple of 2.
- If the dimension corresponding to activateDim is not the last axis of x, the last axis of the output yOut cannot exceed 5120.
- The sum of all elements in groupIndexOptional cannot be greater than the product of the remaining axes of the input x except the last axis.
- The parts of the output yOut and scaleOut that exceed the sum of all elements in groupIndexOptional are not cleared. The memory of this part is junk data.
- <term>Atlas A2 training products/Atlas A2 inference products</term> and <term>Atlas A3 training products/Atlas A3 inference products</term>: When groupIndexOptional is input, the maximum size of the input tensor supported by the operator is limited. The last axis of x cannot exceed 7232.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant_v2.h"

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> weightScaleShape = {1, 64};
  std::vector<int64_t> activationScaleShape = {2};
  std::vector<int64_t> biasShape = {1, 64};
  std::vector<int64_t> scaleShape = {1, 32}; // quantscale
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> groupIndexShape = {1};
  std::vector<int64_t> outShape = {2, 32};
  std::vector<int64_t> scaleOutShape = {2};

  void* xDeviceAddr = nullptr;
  void* weightScaleDeviceAddr = nullptr;
  void* activationScaleDeviceAddr = nullptr;
  void* biasDeviceAddr = nullptr;

  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* weightScale = nullptr;
  aclTensor* activationScale= nullptr;
  aclTensor* bias = nullptr;

  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;

  std::vector<int32_t> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> weightScaleData = {1.0};
  std::vector<float> activationScaleData = {1.0};
  std::vector<float> biasData = {1.0};
  std::vector<int64_t> groupIndexData = {1};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};

  bool activateLeft = true;
  int64_t dstType = 2;
  int64_t activateDim = -1;
  int64_t swigluMode = 1;
  float clampLimit = 7.0;
  float gluAlpha = 1.0;
  float gluBias = 1.702;

  // Create an x aclTensor.
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT32, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create weightScale.
  ret = CreateAclTensor(weightScaleData, weightScaleShape, &weightScaleDeviceAddr, aclDataType::ACL_FLOAT, &weightScale);
  // Create activationScale.
  ret = CreateAclTensor(activationScaleData, activationScaleShape, &activationScaleDeviceAddr, aclDataType::ACL_FLOAT, &activationScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a bias.
  ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a scale aclTensor.
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an offset aclTensor.
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a groupIndex aclTensor.
  ret = CreateAclTensor(groupIndexData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create an out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create a scaleOut aclTensor.
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first part of the aclnnDequantSwigluQuantV2 API.
  ret = aclnnDequantSwigluQuantV2GetWorkspaceSize(x, weightScale, activationScale, bias, scale, nullptr, groupIndex, activateLeft, "dynamic", dstType, "rint", activateDim, swigluMode, clampLimit, gluAlpha, gluBias, out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second segment of the aclnnDequantSwigluQuantV2 API.
  ret = aclnnDequantSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
