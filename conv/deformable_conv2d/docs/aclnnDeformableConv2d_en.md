# aclnnDeformableConv2d

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/conv/deformable_conv2d)

## Product Support

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Implements convolution operations, supporting 2D, deformable, and grouped convolutions.

- Formula:
  
  Assume that the input tensor (**input**) has shape **[N, inC, inH, inW]** and the output tensor (**out**) has shape **[N, outC, outH, outW]**. The output height (**outH**) and width (**outW**) are computed as:
  
  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$
  
  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$
  
  For standard convolution, the sampling coordinates are:
  
  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw ∈ (0, K\_W – 1)
  $$
  
  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh ∈ (0, K\_H – 1)
  $$
  
  For deformable convolution with the given offset, the deformed coordinates are:
  
  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  The value at the deformed location is computed using bilinear interpolation:
  
  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$
  
  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\ 
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\ 
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\ 
  $$
  
  $$
  deformOut(x, y) = weight_{00} * input(x0, y0) + weight_{01} * input(x0,y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1,y1)
  $$
  
  The final convolution output is given by:
  
  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnDeformableConv2dGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnDeformableConv2d** is called to perform computation.

```Cpp
aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(
  const aclTensor*   x,
  const aclTensor*   weight,
  const aclTensor*   offset,
  const aclTensor*   biasOptional,
  const aclIntArray* kernelSize,
  const aclIntArray* stride,
  const aclIntArray* padding,
  const aclIntArray* dilation,
  int64_t            groups,
  int64_t            deformableGroups,
  bool               modulated,
  aclTensor*         out,
  aclTensor*         deformOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```
```Cpp
aclnnStatus aclnnDeformableConv2d(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnDeformableConv2dGetWorkspaceSize

- **Parameters**

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
      <th>Shape</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input data, corresponding to `input` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape is [N, inC, inH, inW], where inH × inW must not exceed 2,147,483,647.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>Input</td>
      <td>4D learnable filter tensor, corresponding to `weight` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type and format must be the same as those of `x`. </li><li>The shape is [outC, inC/groups, K_H, K_W].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>Input</td>
      <td>4D tensor for x-y coordinate offsets and mask, corresponding to `offset` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type and format must be the same as those of <idp:inline displayname="code" id="code15295625171515">x</idp:inline>. </li><li>When `modulated` is True, the shape is [N, 3 * deformableGroups * K_H * K_W, outH, outW]. When `modulated` is False, the shape is [N, 2 * deformableGroups * K_H * K_W, outH, outW].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>Input</td>
      <td>(Optional) 1D tensor for the bias added to the filter output, corresponding to `bias` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type must be the same as that of `x`. </li><li>Pass a null pointer when this parameter is not required; the shape is [outC] if it is provided.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the convolution kernel, corresponding to `K_H` and `K_W` in the formula.</td>
      <td>The size is 2(K_H, K_W). Each element must be greater than 0. K_H * K_W must not exceed 2048. K_H * K_W * inC/groups must not exceed 65535.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Stride of the sliding window for each input dimension, corresponding to `stride` in the formula.</td>
      <td>The size is 4. Each element must be greater than 0. The order of dimensions is interpreted according to the data format of `x`. The N and C dimensions must be set to 1.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of pixels padded to each side of the input (top, bottom, left, and right), corresponding to `padding` in the formula.</td>
      <td>The size is 4.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Dilation factor for each input dimension, corresponding to `dilation` in the formula.</td>
      <td>The size is 4. Each element must be greater than 0. The order of dimensions is interpreted according to the data format of x. The N and C dimensions must be set to 1.</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>Input</td>
      <td>Number of grouped connections from input channels to output channels.</td>
      <td>Both inC and outC must be divisible by groups, and groups must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformableGroups</td>
      <td>Input</td>
      <td>Number of deformable group partitions.</td>
      <td>inC must be divisible by deformableGroups, and deformableGroups must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>Input</td>
      <td>Reserved parameter, indicating whether the offset contains a mask. If true, `offset` contains a mask; if false, it does not.</td>
      <td>Currently, only true is supported.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output data, corresponding to `out` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type and format must be the same as those of `x`. </li><li>The shape is [N, outC, outH, outW].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>deformOutOptional</td>
      <td>Output</td>
      <td>(Optional) Deformable convolution sampling points, corresponding to `deformOut` in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The data type and format must be the same as those of `x`. </li><li>The shape is [N, inC, outH * K_H, outW * K_W].</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND, NCHW</td>
      <td>4</td>
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
      <td>Operator executor, containing the operator computation flow.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Returns**

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
      <td>The passed x, weight, offset, or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The data type or format of x, weight, offset, or out is not supported.</td>
    </tr>
    <tr>
      <td>When deformOutOptional is not a null pointer, the data type or format is not supported.</td>
    </tr>
    <tr>
      <td>When biasOptional is not a null pointer, the data type or format is not supported.</td>
    </tr>
    <tr>
      <td>The shape of x, weight, offset, biasOptional, out, or deformOutOptional does not match that described in the parameter description.</td>
    </tr>
    <tr>
      <td>The size of kernelSize, stride, padding, or dilation does not match that described in the parameter description.</td>
    </tr>
    <tr>
      <td>K_H * K_W exceeds 2048, or K_H * K_W * inC/groups exceeds 65535.</td>
    </tr>
  </tbody></table>

## aclnnDeformableConv2d

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
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnDeformableConv2dGetWorkspaceSize.</td>
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

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Deterministic computation:
  - **aclnnDeformableConv2d** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deformable_conv2d.h"
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
    int64_t shape_size = 1;
    for (auto i : shape) {
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

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto format = shape.size() == 1 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
    std::vector<int64_t> xShape = {1, 6, 2, 4};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> biasShape = {4};
    std::vector<int64_t> outShape = {1, 4, 2, 4};
    std::vector<int64_t> deformOutShape = {1, 6, 10, 20};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* deformOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* deformOut = nullptr;
    std::vector<float> xHostData(1*6*2*4,1);
    std::vector<float> weightHostData(4*3*5*5,1);
    std::vector<float> offsetHostData(1*75*2*4,1);
    std::vector<float> biasHostData(4,0);
    std::vector<float> outHostData(1*4*2*4,0);
    std::vector<float> deformOutHostData(1*6*10*20,0);
    // Create an x aclTensor.
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // Create a weight aclTensor.
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an offset aclTensor.
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a bias aclTensor.
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a deformOut aclTensor.
    ret = CreateAclTensor(deformOutHostData, deformOutShape, &deformOutDeviceAddr, aclDataType::ACL_FLOAT, &deformOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a kernelSize aclIntArray.
    const aclIntArray *kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // Create a stride aclIntArray.
    const aclIntArray *strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // Create a padding aclIntArray.
    const aclIntArray *paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // Create a dilation aclIntArray.
    const aclIntArray *dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnDeformableConv2d.
    ret = aclnnDeformableConv2dGetWorkspaceSize(x, weight, offset, bias, kernelSizeArray, strideArray, paddingArray, dilationArray, groups, deformableGroups, true, out, deformOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnDeformableConv2d.
    ret = aclnnDeformableConv2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2d failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(deformOut);

    // 7. Release device resources. Modify the code based on the API definition.
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(deformOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
