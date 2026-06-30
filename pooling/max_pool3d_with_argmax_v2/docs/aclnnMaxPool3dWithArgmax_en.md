# aclnnMaxPool3dWithArgmax

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/pooling/max_pool3d_with_argmax_v2)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

* Description:
  * Applies three-dimensional max pooling over the input channels of an input signal, and outputs the value **out** and **indices** after pooling.
  * In the input **dims**, N represents the batch size, C represents the channel, D represents the depth, W represents the width, and H represents the height.
  * If the product of D × H × W exceeds int32, it is recommended that the D axis be split based on the model size.
* Formula:
  
  * Calculation formula of each element of the output tensor:
    
    $$
    out(N_i, C_j, d, h, w) = \max\limits_{{k\in[0,k_{D}-1],m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times d + k, stride[1]\times h + m, stride[2]\times w + n)
    $$

  * Formula for deducing the output tensor shape (with ceilMode set to false by default, that is, rounding down)
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lfloor{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1, \lfloor{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rfloor + 1]
    $$
    
  * Formula for deducing the output tensor shape (with ceilMode set to true by default, that is, rounding up)
    
    $$
    [N, C, D_{out}, H_{out}, W_{out}]=[N,C,\lceil{\frac{D_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rceil + 1,\lceil{\frac{H_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rceil + 1, \lceil{\frac{W_{in}+2 \times {padding[2] - dilation[2] \times(kernelSize[2] - 1) - 1}}{stride[2]}}\rceil + 1]
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnMaxPool3dWithArgmaxGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnMaxPool3dWithArgmax** is called to perform computation.

* `aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnMaxPool3dWithArgmax(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaxPool3dWithArgmaxGetWorkspaceSize

* **Parameters:**
  
  * **self** (aclTensor*, compute input): input tensor, aclTensor on the device. The data type can only be FLOAT32, FLOAT16, or BFLOAT16. The shape can be four-dimensional or five-dimensional. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **kernelSize** (aclIntArray*, compute input): maximum pooling window size. The array length must be 1 or 3, and all array elements must be greater than 0.
  * **stride** (aclIntArray*, compute input): stride of the window. The array length must be 0, 1, or 3, and all array elements must be greater than 0. When the array length is 0, the value of **kernelSize** is used as **strides**.
  * **padding** (aclIntArray*, compute input): the number of layers of padding to be applied on each side, with negative infinity values used for padding. The array length must be 1 or 3, and all array elements must be greater than or equal to 0 and less than or equal to kernelSize divided by 2.
  * **dilation** (aclIntArray*, compute input): controls the stride between elements in the window. The array length must be 1 or 3, and the value can only be 1.
  * **ceilMode** (bool, compute input): mode of computing the output shape, that is, rounding up (**True**) or rounding down (**False**).
  * **out** (aclTensor \*, compute output): output tensor, aclTensor on the device. It is the pooled result. The data type is the same as that of **self**. The shape is deduced from the preceding formula. The [data format](../../../docs/en/context/data_formats.md) can be ND and must be the same as that of **self**.
  * **indices** (aclTensor \*, compute output): output tensor, aclTensor on the device. This tensor consists of the index positions of maximum values. The data type can only be INT32. The shape is the same as that of **out**. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **workspaceSize** (uint64_t \*, output): size of the workspace to be allocated on the device.
  * **executor** (aclOpExecutor \*\*, output): operator executor, containing the operator computation process.
* **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

```
  The first-phase API implements input parameter verification. The following errors may be thrown:
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The passed self or out is a null pointer.
  161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type of self is not supported.
                                   2. The data format of self is not supported.
                                   3. The shape of self is not four-dimensional or five-dimensional.
                                   4. An axis of the shape of out deduced from the formula is 0.
                                   5. kernelSize has values less than or equal to 0.
                                   6. The length of kernelSize is not 1 or 3.
                                   7. stride has values less than or equal to 0.
                                   8. The length of stride is not 0, 1, or 3. (When the length of stride is 0, the stride value is the kernelSize value.)
                                   9. padding has values less than 0 or greater than kernelSize divided by 2.
                                   10. The length of padding is not 1 or 3.
                                   11. dilation has values not equal to 1.
                                   12. This operator is not supported by the platform.
                                   13. depth × height × width > max int32, which exceeds the expression range of indices.


  561103 (ACLNN_ERR_INNER_NULLPTR): 1. The intermediate result is null.
  561101 (ACLNN_ERR_INNER_CREATE_EXECUTOR): 1. The executor is null.
```

## aclnnMaxPool3dWithArgmax

- **Parameters:**
  
  * **workspace** (void \*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by the first-phase API **aclnnMaxPool3dWithArgmaxGetWorkspaceSize**.
  * **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
- Deterministic compute:
  - **aclnnMaxPool3dWithArgmax** defaults to a deterministic implementation.

- The data type of the input tensor can only be FLOAT32, FLOAT16, or BFLOAT16.

- The input data format does not support NDHWC.

- The **kernelSize**, **stride**, **padding**, **dilation**, and **ceilMode** parameters must ensure that no axis in the out shape is less than 1.

- If **ceilMode** is set to **True** and sliding windows are all in the right padded region, the output result will be ignored.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
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
