# aclnnDeformableConv2dBackward

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-20T17:08:49.292Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Implements backward propagation of deformable convolution and computes the gradient of the input, the weight, the bias, and the offset.

- Formula:

  The relationship between the input input($N,C_{in},H_{in},W_{in}$), the output out($N,C_{out},H_{out},W_{out}$), the convolution stride ($stride$), the kernel size ($kernelSize,K_H,K_W$), the dilation ($dilation$), and the padding ($padding$) is:

  $$
    H_{out}=\lfloor \frac{H_{in}+padding[0]+padding[1]-((K_H - 1) * dilation[2] + 1)}{stride[2]}+1 \rfloor
  $$

  $$
    W_{out}=\lfloor \frac{W_{in}+padding[2]+padding[3]-((K_W - 1) * dilation[3] + 1)}{stride[3]}+1 \rfloor
  $$

  The backward propagation of deformable convolution requires computing the gradients of the input tensor $x$ of the forward convolution (input in the prototype), the convolution kernel weight tensor $w$ (weight in the prototype), the bias $b$ (bias in the prototype), and the offset $o$ (offset in the prototype).

  For deformable convolution, the sampling position of each output position is dynamically determined by the offset. Let the output position be $(n, c_{out}, i, j)$ and the corresponding offset be $\Delta p_{n,c_{out},i,j}$. The actual sampling position is $p_{n,c_{out},i,j} = (i \cdot s_H, j \cdot s_W) + \Delta p_{n,c_{out},i,j}$, where $s_H$ and $s_W$ are the strides in the height and width directions, respectively.

  - For the gradient of $x$, $\frac{\partial L}{\partial x}$ (corresponding to the gradInput parameter in the function prototype):

    $$
    \frac{\partial L}{\partial x_{n, c_{in}, h, w}} = \sum_{c_{out}=1}^{C_{out}} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}} \cdot w_{c_{out}, c_{in}, p, q} \cdot \delta(p_{n,c_{out},i,j} = (h, w))
    $$

    where $L$ is the loss function, $\frac{\partial L}{\partial y}$ is the gradient of the output tensor $y$ with respect to $L$ (the gradOutput parameter in the prototype), $w_{c_{out}, c_{in}, p, q}$ denotes the $(p, q)$ element of the convolution kernel weight tensor $w$ (the weight parameter in the prototype, with shape $(C_{out}, C_{in}/groups, K_H, K_W)$), where $p$ and $q$ are the coordinate indexes of the kernel in the height and width directions, respectively, $p \in [1, K_H]$, $q \in [1, K_W]$, and $\delta(\cdot)$ is the indicator function, which is 1 when the sampling position coincides with the input position $(h,w)$ and 0 otherwise. In the actual implementation, the gradient is distributed to neighboring pixel positions through bilinear interpolation.

  - For the gradient of $w$, $\frac{\partial L}{\partial w}$ (corresponding to the gradWeight parameter in the function prototype):

    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, p_{n,c_{out},i,j}.h, p_{n,c_{out},i,j}.w} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

    where $p_{n,c_{out},i,j}.h$ and $p_{n,c_{out},i,j}.w$ indicate the height and width coordinates of the sampling position respectively, obtained by sampling from the input tensor $x$ through bilinear interpolation.

  - For the gradient of $b$, $\frac{\partial L}{\partial b}$ (corresponding to the gradBias parameter in the function prototype):

    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$

  - For the gradient of $o$ $\frac{\partial L}{\partial o}$ (corresponding to the gradOffset parameter in the prototype):

    $$
    \frac{\partial L}{\partial o_{n, c_{out}, i, j}} = \sum_{c_{in}=1}^{C_{in}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}} \cdot w_{c_{out}, c_{in}, p, q} \cdot \frac{\partial x_{n, c_{in}, p_{n,c_{out},i,j}.h, p_{n,c_{out},i,j}.w}}{\partial o_{n, c_{out}, i, j}}
    $$

    where the computation of the gradient of the offset involves the backward propagation of bilinear interpolation. The partial derivative of the input value with respect to the offset must be computed. This partial derivative reflects the impact of offset changes on the sampling point position.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnDeformableConv2dBackwardGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor containing the operator computation flow, and then call the "aclnnDeformableConv2dBackward" API to execute the computation.

```Cpp
aclnnStatus aclnnDeformableConv2dBackwardGetWorkspaceSize(
  const aclTensor*   input,
  const aclTensor*   gradOutput,
  const aclTensor*   offsetOut,
  const aclTensor*   weight,
  const aclTensor*   offset,
  const aclIntArray* kernelSize,
  const aclIntArray* stride,
  const aclIntArray* padding,
  const aclIntArray* dilation,
  int64_t            groups,
  int64_t            deformableGroups,
  bool               modulated,
  aclTensor*         gradInput,
  aclTensor*         gradWeight,
  aclTensor*         gradOffset,
  aclTensor*         gradBias,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnDeformableConv2dBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  const aclrtStream    stream)
```

## aclnnDeformableConv2dBackwardGetWorkspaceSize

- **Parameters**:

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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>Input</td>
      <td>Input</td>
      <td>Indicates the input data for forward propagation.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The shape is [N, C<sub>in</sub>, H<sub>in</sub>, W<sub>in</sub>].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput</td>
      <td>Input</td>
      <td>Indicates the gradient of the forward propagation output.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format keep consistent with input.</li><li>The shape is [N, C<sub>out</sub>, H<sub>out</sub>, W<sub>out</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOut</td>
      <td>Input</td>
      <td>Indicates the output sampling points of deformable convolution in forward propagation.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format keep consistent with input.</li><li>The shape is [N, C<sub>in</sub>, H<sub>out</sub> * K<sub>H</sub>, W<sub>out</sub> * K<sub>W</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>Indicates the convolution kernel weight in forward propagation.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format keep consistent with those of input.</li><li>The shape is [C<sub>out</sub>, C<sub>in</sub>/groups, K<sub>H</sub>, K<sub>W</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>Indicates the x-y coordinate offset and mask for forward propagation.</td>
      <td><ul><li>Does not support empty Tensor.</li><li>The data type and data format keep consistent with input.</li><li>If modulated is True, the shape is [N, 3 * deformableGroups * K<sub>H</sub> * K<sub>W</sub>, H<sub>out</sub>, W<sub>out</sub>].</li></ul></td>
      <td>Consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Indicates the kernel size, corresponding to K<sub>H</sub> and K<sub>W</sub> in the formula.</td>
      <td>size is 2(K<sub>H</sub>, K<sub>W</sub>), each element is greater than zero, K<sub>H</sub> * K<sub>W</sub> cannot exceed 2048, and K<sub>H</sub> * K<sub>W</sub> * inC/groups cannot exceed 65535.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Indicates the sliding window stride in each input dimension, corresponding to stride in the formula.</td>
      <td>Size is 4, each element is greater than zero, and the dimension order is interpreted based on the data format of input. The N and C dimensions must be set to 1.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Indicates the number of pixels to be added to each side (top, bottom, left, and right) of the input, corresponding to padding in the formula.</td>
      <td>Size is 4.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Indicates the dilation coefficient of each input dimension, corresponding to dilation in the formula.</td>
      <td>size is 4, each element is greater than zero, and the dimension order is interpreted based on the data format of input. The N and C dimensions must be set to 1.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Input</td>
      <td>Indicates the number of groups in grouped convolution.</td>
      <td>inC and outC must both be divisible by groups, and groups must be greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformableGroups</td>
      <td>Input</td>
      <td>Indicates the number of deformable group partitions.</td>
      <td>inC must be divisible by deformableGroups, and deformableGroups must be greater than zero.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Input</td>
      <td>Indicates whether offset contains a mask. If true, offset contains a mask; if false, it does not.</td>
      <td>Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>Output</td>
      <td>Indicates the gradient of the input data.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format must keep consistent with input.</li><li>The shape is [N, C<sub>in</sub>, H<sub>in</sub>, W<sub>in</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>Output</td>
      <td>Indicates the gradient of the convolution kernel weight.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format keep consistent with those of input.</li><li>The shape is [C<sub>out</sub>, C<sub>in</sub>/groups, K<sub>H</sub>, K<sub>W</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOffset</td>
      <td>Output</td>
      <td>Indicates the gradient of the offset.</td>
      <td><ul><li>Does not support empty tensors.</li><li>The data type and data format keep consistent with those of input.</li><li>If modulated is True, the shape is [N, 3 * deformableGroups * K<sub>H</sub> * K<sub>W</sub>, H<sub>out</sub>, W<sub>out</sub>]; if modulated is False, the shape is [N, 2 * deformableGroups * K<sub>H</sub> * K<sub>W</sub>, H<sub>out</sub>, W<sub>out</sub>].</li></ul></td>
      <td>Consistent with input</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>Output</td>
      <td>Gradient of the bias.</td>
      <td><ul><li>Does not support empty tensors.</li><li>shape is [C<sub>out</sub>].</li></ul></td>
      <td>Keep consistent with input</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Return value**:

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Error Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The input, gradOutput, offsetOut, weight, offset, gradInput, gradWeight, gradBias, or gradOffset passed in is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or data format of input, gradOutput, offsetOut, weight, offset, gradInput, gradWeight, gradBias, and gradOffset is not in the supported range.</td>
    </tr>
    <tr>
      <td>The shapes of input, gradOutput, offsetOut, weight, offset, gradInput, gradWeight, gradBias, and gradOffset are inconsistent with those described in Parameters.</td>
    </tr>
    <tr>
      <td>The sizes of kernelSize, stride, padding, and dilation are inconsistent with those described in Parameters.</td>
    </tr>
    <tr>
      <td>K<sub>H</sub>*K<sub>W</sub> is greater than 2048, or K<sub>H</sub>*K<sub>W</sub>*inC/groups is greater than 65535.</td>
    </tr>
    <tr>
      <td>The groups or deformableGroups parameter does not meet the constraints.</td>
    </tr>
    <tr>
      <td>The modulated parameter is not true.</td>
    </tr>
    <tr>
     <td>ACLNN_ERR_INNER_NULLPTR</td>
     <td>561103</td>
     <td>API internal verification error. Usually caused by input data or attributes whose specifications are out of the supported range.</td>
   </tr>
  </tbody></table>

## aclnnDeformableConv2dBackward

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace applied by the Device side, obtained through the first-phase API aclnnDeformableConv2dBackwardGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, including the operator computing process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnDeformableConv2dBackward defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
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

int Init(int32_t deviceId, aclrtStream* stream)
{
    // Boilerplate: initialize resources.
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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // Call aclrtMemcpy to copy host-side data to device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Compute the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto format = shape.size() == 1 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnDeformableConv2dBackwardTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // Handle the check as needed.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct the input and output. Customize the construction based on the API.
    std::vector<int64_t> inputShape = {1, 6, 2, 4};
    std::vector<int64_t> gradOutputShape = {1, 4, 2, 4};
    std::vector<int64_t> offsetOutShape = {1, 6, 10, 20};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> gradInputShape = {1, 6, 2, 4};
    std::vector<int64_t> gradWeightShape = {4, 3, 5, 5};
    std::vector<int64_t> gradBiasShape = {4};
    std::vector<int64_t> gradOffsetShape = {1, 75, 2, 4};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* inputDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* offsetOutDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    void* gradOffsetDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* offsetOut = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;
    aclTensor* gradOffset = nullptr;
    std::vector<float> inputHostData(1 * 6 * 2 * 4, 1);
    std::vector<float> gradOutputHostData(1 * 4 * 2 * 4, 1);
    std::vector<float> offsetOutHostData(1 * 6 * 10 * 20, 1);
    std::vector<float> weightHostData(4 * 3 * 5 * 5, 1);
    std::vector<float> offsetHostData(1 * 75 * 2 * 4, 1);
    std::vector<float> gradInputHostData(1 * 6 * 2 * 4, 0);
    std::vector<float> gradWeightHostData(4 * 3 * 5 * 5, 0);
    std::vector<float> gradBiasHostData(4, 0);
    std::vector<float> gradOffsetHostData(1 * 75 * 2 * 4, 0);
    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the gradOutput aclTensor.
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the offsetOut aclTensor.
    ret = CreateAclTensor(offsetOutHostData, offsetOutShape, &offsetOutDeviceAddr, aclDataType::ACL_FLOAT, &offsetOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradInput aclTensor.
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradWeight aclTensor.
    ret = CreateAclTensor(
        gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradBias aclTensor.
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create gradOffset aclTensor.
    ret = CreateAclTensor(
        gradOffsetHostData, gradOffsetShape, &gradOffsetDeviceAddr, aclDataType::ACL_FLOAT, &gradOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create kernelSize aclIntArray.
    const aclIntArray* kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // Create stride aclIntArray.
    const aclIntArray* strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // Create padding aclIntArray.
    const aclIntArray* paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // Create dilation aclIntArray.
    const aclIntArray* dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. Call the CANN operator library API. Modify it to the specific API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnDeformableConv2dBackward.
    ret = aclnnDeformableConv2dBackwardGetWorkspaceSize(
        input, gradOutput, offsetOut, weight, offset, kernelSizeArray, strideArray, paddingArray, dilationArray, groups,
        deformableGroups, true, gradInput, gradWeight, gradOffset, gradBias, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnDeformableConv2dBackward.
    ret = aclnnDeformableConv2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize and wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from device memory to the host. Modify it based on the specific API definition.
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightShapeData(size, 0);
    ret = aclrtMemcpy(
        gradWeightShapeData.data(), gradWeightShapeData.size() * sizeof(gradWeightShapeData[0]), gradWeightDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradWeightShapeData[i]);
    }

    size = GetShapeSize(gradOffsetShape);
    std::vector<float> gradOffsetShapeData(size, 0);
    ret = aclrtMemcpy(
        gradOffsetShapeData.data(), gradOffsetShapeData.size() * sizeof(gradOffsetShapeData[0]), gradOffsetDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradOffsetShapeData[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasShapeData(size, 0);
    ret = aclrtMemcpy(
        gradBiasShapeData.data(), gradBiasShapeData.size() * sizeof(gradBiasShapeData[0]), gradBiasDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradBiasShapeData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(offsetOut);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);
    aclDestroyTensor(gradOffset);

    // 7. Release device resources. Modify it based on the specific API definition.
    aclrtFree(inputDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(offsetOutDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    aclrtFree(gradOffsetDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize device/stream.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDeformableConv2dBackwardTest(deviceId, stream);
    CHECK_FREE_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
