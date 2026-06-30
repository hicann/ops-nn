# aclnnApplyAdamWQuant

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam_w_quant)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description: The m and v inputs of the optimizer are used as indexes to obtain their values in the qmap, and the values are multiplied by the absmax corresponding to each blockSize for dequantization. Then, the AdamW optimizer is implemented. The maximum value of the updated m and v is selected from each blockSize. Each blockSize of m and v corresponds to an absmax, and normalization is performed once. The indexes in the qmap corresponding to m and v are found using the binary search method as the output. The absmax is also used as the input for the next round of quantization.

- **Optimizer calculation formula:**

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$
  
  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## Prototype
Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnApplyAdamWQuantGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnApplyAdamWQuant** is called to perform computation.

* `aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor *varRef, const aclTensor *grad, aclTensor *mRef, aclTensor *vRef, const aclTensor *qmapM, const aclTensor *qmapV, aclTensor *absmaxMRef, aclTensor *absmaxVRef, const aclTensor *step, double lr, double beta1, double beta2, double weightDecay, double eps, double gnormScale, char *quantModeOptional, int64_t blockSize, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnApplyAdamWQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);`

## aclnnApplyAdamWQuantGetWorkspaceSize

- **Parameters:**

  * **varRef** (const aclTensor*, compute input/compute output): weight input and output (theta in the formula), which is an aclTensor on the device. The data type can be FLOAT, FLOAT16, or BFLOAT16. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **grad** (aclTensor*, compute input): gradient (gt in the formula), which is an aclTensor on the device. The data type can be FLOAT16, BFLOAT16, or FLOAT32, which are the same as that of **varRef**. The shape must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **mRef** (aclTensor*, compute input/compute output): index value of the m parameter in the AdamW optimizer formula before quantization, which is an aclTensor on the device. The specific value in qmapM is exported based on the index. The data type can be UINT8. The shape must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **vRef** (aclTensor*, compute input/compute output): index value of the v parameter in the AdamW optimizer formula before quantization, which is an aclTensor on the device. The specific value in qmapV is exported based on the index. The data type can be UINT8. The shape must be the same as that of **varRef**. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **qmapM** (aclTensor*, compute input): aclTensor on the device. The quantization mapping table is sorted in ascending order. The data type can be FLOAT32. The data format must be ND. The shape must be [256,]. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **qmapV** (aclTensor*, compute input): aclTensor on the device. The quantization mapping table is sorted in ascending order. The data type can be FLOAT32. The data format must be ND. The shape must be [256,]. [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **absmaxMRef** (aclTensor*, compute input/compute output): aclTensor on the device. The compute input is percentile denormalization (multiplied by **absmaxMRef**) in the current dequantization phase. The compute output is the parameters for the current quantization and the next round of dequantization. The data type can be FLOAT32. The shape requirement is that every 256 m values correspond to one maximum value. The shape requirement is "absmaxMRef.size = m.size/blockSize". [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **absmaxVRef** (aclTensor*, compute input/compute output): aclTensor on the device. The compute input is percentile denormalization (multiplied by **absmaxVRef**) in the current dequantization phase. The compute output is the parameters for the current quantization and the next round of dequantization. The data type can be FLOAT32. The shape requirement is that every 256 v values correspond to one maximum value. The shape requirement is "absmaxVRef.size = m.size/blockSize". [Non-contiguous tensors](../../../docs/en/context/non_contiguous_tensors.md) are not supported. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **step** (aclTensor*): number of iterations (t in the formula), which is an aclTensor on the device. The data type can be INT64. The shape is [1,]. The [data format](../../../docs/en/context/data_formats.md) can be ND.
  * **lr** (float\*, compute input): learning rate (eta in the formula). The recommended value is 1e-3, 1e-5, or 1e-8. The value ranges from 0 to 1. The data type can be float.
  * **beta1** (float\*, compute input): beta1 parameter in the AdamW optimizer formula. The recommended value is **0.9**. The value ranges from 0 to 1. The data type can be float.
  * **beta2** (float\*, compute input): beta2 parameter in the AdamW optimizer formula. The recommended value is **0.99**. The value ranges from 0 to 1. The data type can be float.
  * **weightDecay** (float\*, compute input): weight decay coefficient (lambda in the AdamW optimizer formula). The recommended value is **0.999**. The value ranges from 0 to 1. The data type can be float.
  * **eps** (float\*, compute input): epsilon parameter in the AdamW optimizer formula, which is added to the denominator to prevent division by zero. The recommended value is **1e-8**. The data type can be float.
  * **gnormScale** (float\*, compute input): parameter for scaling the input parameter **grad**. The recommended value is **0.999**. The value ranges from 0 to 1. The data type can be float.
  * **blockSize** (int64\*, compute input): size of each block involved in computation. The value is fixed at **256**. The data type can be int64.
  * **quantModeOptional** (char\*, compute input): reserved parameter.
  * **workspaceSize** (uint64_t\*, output): size of the workspace to be allocated on the NPU device.
  * **executor** (aclOpExecutor\*\*, output): operator executor, containing the operator computation process.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  ``` 
    The first-phase API implements input parameter verification. The following errors may be thrown.
    161001 (ACLNN_ERR_PARAM_NULLPTR): 1. The input or output tensor is a null pointer.
    161002 (ACLNN_ERR_PARAM_INVALID): 1. The data type or format of the input or output is not supported.
  ```

## aclnnApplyAdamWQuant

- **Parameters:**

  * **workspace** (void\*, input): address of the workspace to be allocated on the device.
  * **workspaceSize** (uint64_t, input): size of the workspace to be allocated on the device, which is obtained by calling the first-phase API **aclnnApplyAdamWQuantGetWorkspaceSize**.
  * **executor** (aclOpExecutor \*, input): operator executor, containing the operator computation process.
  * **stream** (aclrtStream, input): stream for executing the task.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints
  The shape of varRef must meet the following constraints:
  - varRef.shape = grad.shape
  - varRef.shape = mRef.shape
  - varRef.shape = vRef.shape
  - varRef.size/blockSize = absmaxMRef.size
  - varRef.size/blockSize = absmaxVRef.size
  
  - Deterministic compute:
    - **aclnnApplyAdamWQuant** defaults to a deterministic implementation.

## Example
The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_quant.h"
#define FAILED 1

#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)

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
// Call aclrtMemcpy to copy the data from the host to the device.
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
// Handle the check as required.
CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 2. Construct the input and output based on the API.
std::vector<int64_t> VarRefShape = {1,256};
std::vector<int64_t> GradShape = {1,256};
std::vector<int64_t> mRefShape = {1,256};
std::vector<int64_t> vRefShape = {1,256};
std::vector<int64_t> mapMShape = {256,};
std::vector<int64_t> mapVShape = {256,};
std::vector<int64_t> absmaxMRefShape = {1,};
std::vector<int64_t> absmaxVRefShape = {1,};
std::vector<int64_t> stepShape = {1,};

void *VarRefDeviceAddr = nullptr;
void *GradDeviceAddr = nullptr;
void *mRefDeviceAddr = nullptr;
void *vRefDeviceAddr = nullptr;
void *qmapMDeviceAddr = nullptr;
void *qmapVDeviceAddr = nullptr;
void *absmaxMRefDeviceAddr = nullptr;
void *absmaxVRefDeviceAddr = nullptr;
void *stepDeviceAddr = nullptr;

aclTensor *varRef = nullptr;
aclTensor *grad = nullptr;
aclTensor *mRef = nullptr;
aclTensor *vRef = nullptr;
aclTensor *qmapM = nullptr;
aclTensor *qmapV = nullptr;
aclTensor *absmaxMRef = nullptr;
aclTensor *absmaxVRef = nullptr;
aclTensor *step = nullptr;

std::vector<float> inputVarHostData(256);
std::vector<float> inputGradHostData(256);
std::vector<uint8_t> inputMHostData(256);
std::vector<uint8_t> inputVHostData(256);
std::vector<float> inputmapMHostData(256);
std::vector<float> inputmapVHostData(256);
std::vector<float> inputabsmaxMHostData = {5};
std::vector<float> inputabsmaxVHostData = {3};
std::vector<int64_t> inputstepHostData(1);

const float lr = 0.1;
const float beta1 = 0.1;
const float beta2 = 0.1;
const float weightDecay = 0.1;
const float eps = 0.01;
const float gnormScale = 0.1;
const int64_t blockSize = 256;
char* quantModeOptional = "BLOCKWISE";

// Create a gradOutput aclTensor.
ret = CreateAclTensor(inputVarHostData, VarRefShape, &VarRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputGradHostData, GradShape, &GradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputMHostData, mRefShape, &mRefDeviceAddr, aclDataType::ACL_UINT8, &mRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputVHostData, vRefShape, &vRefDeviceAddr, aclDataType::ACL_UINT8, &vRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapMHostData, mapMShape, &qmapMDeviceAddr, aclDataType::ACL_FLOAT, &qmapM);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapVHostData, mapVShape, &qmapVDeviceAddr, aclDataType::ACL_FLOAT, &qmapV);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxMHostData, absmaxMRefShape, &absmaxMRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxMRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxVHostData, absmaxVRefShape, &absmaxVRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxVRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputstepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

// 3. Call the CANN operator library API, which needs to be replaced with the actual API.
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
// Call the first-phase API aclnnApplyAdamWQuantGetWorkspaceSize.
ret = aclnnApplyAdamWQuantGetWorkspaceSize(varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, lr, beta1, beta2, weightDecay, eps, gnormScale, quantModeOptional, blockSize, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
// Allocate device memory based on workspaceSize calculated by the first-phase API.
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
}
// Call the second-phase API aclnnApplyAdamWQuant.
ret = aclnnApplyAdamWQuant(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuant failed. ERROR: %d\n", ret); return ret);
// 4. (Fixed writing) Synchronously wait until the task execution is complete.
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

// 5. Obtain the output value and copy the result from the device to the host.
auto size = GetShapeSize(VarRefShape);
std::vector<float> resultData(size, 0);
ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), VarRefDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
for (int64_t i = 0; i < size; i++) {
LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
// 6. Release aclTensor. Modify the configuration based on the API definition.
aclDestroyTensor(varRef);
aclDestroyTensor(grad);
aclDestroyTensor(mRef);
aclDestroyTensor(vRef);
aclDestroyTensor(qmapM);
aclDestroyTensor(qmapV);
aclDestroyTensor(absmaxMRef);
aclDestroyTensor(absmaxVRef);
aclDestroyTensor(step);

// 7. Release device resources. Modify the configuration based on the API definition.
aclrtFree(VarRefDeviceAddr);
aclrtFree(GradDeviceAddr);
aclrtFree(mRefDeviceAddr);
aclrtFree(vRefDeviceAddr);
aclrtFree(qmapMDeviceAddr);
aclrtFree(qmapVDeviceAddr);
aclrtFree(absmaxMRefDeviceAddr);
aclrtFree(absmaxVRefDeviceAddr);
aclrtFree(stepDeviceAddr);
if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
return 0;
}
```
