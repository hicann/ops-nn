# aclnnMaxPool3dWithArgmax

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_with_argmax_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √    |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    √    |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference products</term>                            |    ×    |
| <term>Atlas training products</term>                             |    ×    |

## Function

* Description:
  * Applies three-dimensional max pooling over the input channels of an input signal, and outputs the value **out** and **indices** after pooling.
  * In the input **dims**, N represents the batch size, C represents the channel, D represents the depth, W represents the width, and H represents the height.
  * When the D *H* W exceeds int32, it is recommended that the D axis be split in the model size.

* Formula:
  
  * Calculation formula of each element of the output tensor:
    
    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$

  * Formula for deducing the output tensor shape (with ceilMode set to false by default, that is, rounding down)
    
$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\\
\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \\
\lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
$$

  * Formula for deducing the output tensor shape (with ceilMode set to true by default, that is, rounding up)
    
$$
[N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\\
\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \\
\lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxPool3dWithArgmaxGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxPool3dWithArgmax** is called to perform computation.

```Cpp
aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(
  const aclTensor   *self,
  const aclIntArray *kernelSize, 
  const aclIntArray *stride,
  const aclIntArray *padding,
  const aclIntArray *dilation,
  bool               ceilMode,
  aclTensor         *out,
  aclTensor         *indices,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnMaxPool3dWithArgmax(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMaxPool3dWithArgmaxGetWorkspaceSize

* **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>self</td>
      <td>Input</td>
      <td>Tensor to be computed.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>Input</td>
      <td>Size of the max pooling window.</td>
      <td>The array length must be 1 or 3, and all elements in the array must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>Input</td>
      <td>Step for moving the window.</td>
      <td>The array length must be 0, 1, or 3, and all elements in the array must be greater than 0. When the array length is 0, the value of **kernelSize** is used as **strides**.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>Input</td>
      <td>Number of layers added to each side.</td>
      <td>The position to be added is filled with "negative infinity". The array length must be 1 or 3, and all array elements must be greater than or equal to 0 and less than or equal to kernelSize divided by 2.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>Input</td>
      <td>Stride of the elements in the control window.</td>
      <td>The array length must be 1 or 3, and all elements in the array must be greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>Input</td>
      <td>Rounding method used to calculate the output shape.</td>
      <td>If the value is True, the output shape is rounded up. If the value is False, the output shape is rounded down.</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Output tensor, which is the result of pooling.</td>
      <td>The data type is the same as that of self. The shape is deduced from the preceding formula.</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Output</td>
      <td>Tensor consisting of the index positions of the maximum values.</td>
      <td>The shape is the same as that of out.</td>
      <td>INT32 or INT64</td>
      <td>NCDHW, NDHWC, ND</td>
      <td>4-5</td>
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
  </tbody></table>

   - For the <term>Atlas A3 training products/Atlas A3 inference products</term> and <term>Atlas A2 training products/Atlas A2 inference products</term>, the value of the `dilation` element can only be 1. The `indices` data type does not support INT64. The input data format does not support NDHWC. The depth*height*width cannot be greater than the maximum int32.

* **Returns:**
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
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
      <td>The passed `self` or `out` is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="13">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="13">161002</td>
      <td>The data type of self is not supported.</td>
    </tr>
    <tr>
      <td>The data format of self is not supported.</td>
    </tr>
    <tr>
      <td>The shape of self does not meet the requirements.</td>
    </tr>
    <tr>
      <td>The axis of the shape of out derived from the formula is 0.</td>
    </tr>
    <tr>
      <td>The element values of kernelSize, stride, padding, and dilation do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The lengths of kernelSize, stride, padding, and dilation do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The platform is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnMaxPool3dWithArgmax

- **Parameters:**
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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnMaxPool3dWithArgmaxGetWorkspaceSize.</td>
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

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computing: The aclnnMaxPool3dWithArgmax is implemented in deterministic mode by default.

- The **kernelSize**, **stride**, **padding**, **dilation**, and **ceilMode** parameters must ensure that no axis in the out shape is less than 1.

- If **ceilMode** is set to **True** and sliding windows are all in the right padded region, the output result will be ignored.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <cstdio>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool3d_with_argmax.h"

#define CHECK_RET(cond, return_expr)  \
    do {                              \
        if (!(cond)) {                \
            return_expr;              \
        }                             \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
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
    // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Use CHECK as required.
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> selfShape = {1, 1, 2, 2, 2};
    std::vector<int64_t> outShape = {1, 1, 1, 1, 1};
    std::vector<int64_t> indicesShape = {1, 1, 1, 1, 1};
    std::vector<int64_t> kernelSizeData = {2, 2, 2};
    std::vector<int64_t> strideData = {2, 2, 2};
    std::vector<int64_t> paddingData = {0, 0, 0};
    std::vector<int64_t> dilationData = {1, 1, 1};
    void* selfDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* out = nullptr;
    aclTensor* indices = nullptr;
    std::vector<float> selfHostData = {1, 6, 2, 8, 4, 5, 7, 3};
    std::vector<float> outHostData = {0};
    std::vector<int32_t> indicesHostData = {0};

    // Create a self aclTensor.
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an out aclTensor.
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an indices aclTensor.
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create an input array.
    aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 3);
    aclIntArray* stride = aclCreateIntArray(strideData.data(), 3);
    aclIntArray* padding = aclCreateIntArray(paddingData.data(), 3);
    aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 3);
    const bool ceilMode = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnMaxPool3dWithArgmax API call example
    // 3. Call the first-phase API of aclnnMaxPool3dWithArgmax.
    ret = aclnnMaxPool3dWithArgmaxGetWorkspaceSize(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmaxGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnMaxPool3dWithArgmax.
    ret = aclnnMaxPool3dWithArgmax(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool3dWithArgmax failed. ERROR: %d\n", ret); return ret);

    // 4. Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy output result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("output[%ld] is: %f\n", i, resultData[i]);
    }

    size = GetShapeSize(indicesShape);
    std::vector<int> indicesResultData(size, 0);
    ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesDeviceAddr,
                      size * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("indices[%ld] is: %d\n", i, indicesResultData[i]);
    }

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(self);
    aclDestroyTensor(out);
    aclDestroyTensor(indices);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(selfDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
