# aclnnModulateBackward

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/vfusion/modulate_grad)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description: Calculates parameters in ModulateBackward backpropagation and updates gradients.
- Formula:
  
    If the shape of the input **self** is [B, L, D], the calculation formula is as follows:
    
    $$
    \begin{cases}
    \text{grad\_input} = \text{grad\_output} \odot \text{scale}^{\uparrow L} \\
    \text{grad\_scale} = \sum_{l=1}^{L} (\text{grad\_output} \odot \text{input})_{b,l,d} \\
    \text{grad\_shift} = \sum_{l=1}^{L} \text{grad\_output}_{b,l,d}
    \end{cases}
    $$
    
Symbol description:
    - $\odot$: element-wise multiplication
    - $\sum_{l=1}^{L}$: summation operation, performed along the sequence dimension $L$ (that is, dim=1)
    -  $b,l,d$: subscripts, indicating the tensor dimension indices (commonly Batch, Length, and Dimension)
    - $\text{scale}^{\uparrow L}$: broadcasting (expanding) the scale tensor along the sequence dimension $L$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnModulateBackwardGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnModulateBackward** is called to perform computation.

```Cpp
aclnnStatus aclnnModulateBackwardGetWorkspaceSize(
    const aclTensor* grad_output, 
    const aclTensor* input,
    const aclTensor* scale, 
    const aclTensor* shift, 
    const aclTensor* grad_input, 
    const aclTensor* grad_scale,
    const aclTensor* grad_shift,
    uint64_t*        workspaceSize, 
    aclOpExecutor**  executor)
```
```Cpp
aclnnStatus aclnnModulateBackward(
    void*          workspaceAddr, 
    uint64_t       workspaceSize, 
    aclOpExecutor* executor, 
    aclrtStream    stream)
```

## aclnnModulateBackwardGetWorkspaceSize

- **Parameters:**
  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <td>grad_output</td>
      <td>Input</td>
      <td>Passed feature tensor, corresponding to grad_output in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, seq_len, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input</td>
      <td>Input</td>
      <td>Feature tensor for forward propagation, corresponding to input in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, seq_len, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale</td>
      <td>Input</td>
      <td>Scaling coefficient (optional), corresponding to scale in the formula.</td>
      <td>The data type must be the same as that of grad_output and only two-dimensional input is supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift</td>
      <td>Input</td>
      <td>Shifting coefficient (optional), which determines whether there is a corresponding output.</td>
      <td>The data type must be the same as that of grad_output and only two-dimensional input is supported.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>(batch_size, hidden_dim)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_input</td>
      <td>Output</td>
      <td>Gradient of the input tensor, corresponding to grad_input in the formula.</td>
      <td>The data type and shape must be the same as those of input.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>grad_scale</td>
      <td>Output</td>
      <td>Scaling gradient, corresponding to grad_scale in the formula.</td>
      <td>The data type and shape must be the same as those of scale.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
        <tr>
      <td>grad_shift</td>
      <td>Output</td>
      <td>Shifting gradient, corresponding to grad_shift in the formula.</td>
      <td>The data type and shape must be the same as those of shift.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>-</td>
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
  </tbody></table>
- **Returns:**
  
  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The passed self or out is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type or format of self, scaleOptional, or shiftOptional is not supported.</td>
    </tr>
    <tr>
      <td>The shapes of self, scaleOptional, and shiftOptional do not meet the requirements.</td>
    </tr>
    <tr>
      <td>self is an empty tensor, and scale or shift is not an empty tensor.</td>
    </tr>
  </tbody>
  </table>


## aclnnModulateBackward

- **Parameters:**
  
  <table><thead>
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize.</td>
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
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
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
      <td>The passed grad_output or input is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The input and output data types and formats are not supported.</td>
    </tr>
    <tr>
      <td>The shapes of input, scale, and shift do not meet the requirements.</td>
    </tr>
    <tr>
      <td>The input dimension exceeds three dimensions.</td>
    </tr>
  </tbody>
  </table>

## Constraints

- Deterministic compute:
  - **aclnnModulateBackward** defaults to a deterministic implementation.

- **scale** and **shift** are two-dimensional vectors. The first dimension must have the same size as the first dimension of the input's shape, and the second dimension must have the same size as the third dimension of the input's shape.
- The shape of the input gradoutput must be the same as that of the input.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_modulate_backward.h"

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

void PrintOutResult(const std::vector<int64_t>& shape, void* deviceAddr, const std::string& name){
    auto size = GetShapeSize(shape);
    std::vector<float>resultData(size,0);
    auto ret = aclrtMemcpy(resultData.data(),resultData.size() * sizeof(float),deviceAddr,
                            size * sizeof(float),ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR:%d\n",name.c_str(),ret); return);

    int print_count = std::min(static_cast<int64_t>(10),size);
    LOG_PRINT("%s (first %d elements):\n",name.c_str(),print_count);
    for (int64_t i = 0; i < print_count; i++){
        LOG_PRINT(" [%ld]: %f\n",i,resultData[i]);
    }
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
    std::vector<int64_t> grad_outputShape = {2, 1, 1};
    std::vector<int64_t> inputShape = {2, 1, 1};
    std::vector<int64_t> scaleShape = {2, 1};
    std::vector<int64_t> shiftShape = {2, 1};
    std::vector<int64_t> grad_inputShape = {2, 1, 1};
    std::vector<int64_t> grad_scaleShape = {2, 1};
    std::vector<int64_t> grad_shiftShape = {2, 1};
    void* grad_outputDeviceAddr = nullptr;
    void* inputDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* grad_inputDeviceAddr = nullptr;
    void* grad_scaleDeviceAddr = nullptr;
    void* grad_shiftDeviceAddr = nullptr;
    aclTensor* grad_output = nullptr;
    aclTensor* input = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* grad_input = nullptr;
    aclTensor* grad_scale = nullptr;
    aclTensor* grad_shift = nullptr;
    std::vector<float> grad_outputHostData{10, 20};
    std::vector<float> inputHostData{10, 20};
    std::vector<float> scaleHostData{20, 30};
    std::vector<float> shiftHostData{30, 40};
    std::vector<float> grad_inputHostData{0, 0};
    std::vector<float> grad_scaleHostData{0, 0};
    std::vector<float> grad_shiftHostData{0, 0};
    // Create a grad_output aclTensor.
    ret = CreateAclTensor(grad_outputHostData, grad_outputShape, &grad_outputDeviceAddr, aclDataType::ACL_FLOAT, &grad_output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create an input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a scale aclTensor.
    ret = CreateAclTensor(
        scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a shift aclTensor.
    ret = CreateAclTensor(
        shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a grad_input aclTensor.
    ret = CreateAclTensor(grad_inputHostData, grad_inputShape, &grad_inputDeviceAddr, aclDataType::ACL_FLOAT, &grad_input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a grad_scale aclTensor.
    ret = CreateAclTensor(grad_scaleHostData, grad_scaleShape, &grad_scaleDeviceAddr, aclDataType::ACL_FLOAT, &grad_scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a grad_shift aclTensor.
    ret = CreateAclTensor(grad_shiftHostData, grad_shiftShape, &grad_shiftDeviceAddr, aclDataType::ACL_FLOAT, &grad_shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // Call the first-phase API of aclnnModulate.
    ret = aclnnModulateBackwardGetWorkspaceSize(grad_output, input, scale, shift, grad_input, grad_scale, grad_shift, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulaBackwardteGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // Call the second-phase API of aclnnModulate.
    ret = aclnnModulateBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnModulateBackward failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
   PrintOutResult(grad_inputShape, grad_inputDeviceAddr,"grad_input");
   PrintOutResult(grad_scaleShape, grad_scaleDeviceAddr,"grad_scale");
   PrintOutResult(grad_shiftShape, grad_shiftDeviceAddr,"grad_shift");

    // 6. Release aclTensor and aclScalar. Modify the configuration based on the API definition.
    aclDestroyTensor(grad_output);
    aclDestroyTensor(input);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(grad_input);
    aclDestroyTensor(grad_scale);
    aclDestroyTensor(grad_shift);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(grad_outputDeviceAddr);
    aclrtFree(inputDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(grad_inputDeviceAddr);
    aclrtFree(grad_scaleDeviceAddr);
    aclrtFree(grad_shiftDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    LOG_PRINT("Program completed successfully.\n");
    return 0;
}
```
