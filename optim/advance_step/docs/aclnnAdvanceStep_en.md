# aclnnAdvanceStep

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/optim/advance_step)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description:
  
  vLLM is a high-performance LLM inference and service framework that focuses on optimizing the inference efficiency of large-scale language models. Its core features include PageAttention and efficient memory management. The main function of the advance_step operator is to advance the inference step, that is, update the model status and generate new inputTokens, inputPositions, seqLens, and slotMapping in each generation step, improving the inference efficiency of vLLM.

- Formula:
  
  $$
  blockIdx is the index of the core where the current code is executed.
  $$
  
  $$
  blockTablesStride = blockTables.stride(0)
  $$
  
  $$
  inputTokens[blockIdx] = sampledTokenIds[blockIdx]
  $$
  
  $$
  inputPositions[blockIdx] = seqLens[blockIdx]
  $$
  
  $$
  seqLens[blockIdx] = seqLens[blockIdx] + 1
  $$
  
  $$
  slotMapping[blockIdx] = (blockTables[blockIdx] + blockTablesStride * blockIdx) * blockSize + (seqLens[blockIdx] \% blockSize)
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnAdvanceStepGetWorkspaceSize** is called to obtain the input parameters and compute the required workspace size based on the process. Then, **aclnnAdvanceStep** is called to perform computation.

```Cpp
aclnnStatus aclnnAdvanceStepGetWorkspaceSize(
  const aclTensor *inputTokens,
  const aclTensor *sampledTokenIds,
  const aclTensor *inputPositions,
  const aclTensor *seqLens,
  const aclTensor *slotMapping,
  const aclTensor *blockTables,
  int64_t          numSeqs,
  int64_t          numQueries,
  int64_t          blockSize,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAdvanceStep(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnAdvanceStepGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 171px">
  <col style="width: 115px">
  <col style="width: 250px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Precaution</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>inputTokens</td>
      <td>Input</td>
      <td>Input/Output parameter for the AdvanceStep computation, which is the output inputTokens in the formula. It is used to update the token value in the vLLM model.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape length is the same as that of numSeqs. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sampledTokenIds</td>
      <td>Input</td>
      <td>Input parameter for the AdvanceStep computation, which is used to store tokenIDs and corresponds to input sampledTokenIds in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The length of the first dimension of the shape is the same as that of numQueries, and the length of the second dimension is 1. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>inputPositions</td>
      <td>Input/Output</td>
      <td>Input/Output parameter for the AdvanceStep computation, which is the output inputPositions in the formula. It is used to record the token index.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape length is the same as that of numSeqs. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>seqLens</td>
      <td>Input/Output</td>
      <td>Input/Output parameter for the AdvanceStep computation, which is used to record the seq length under different blockIdx. It corresponds to the input/output seqLens in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape length is the same as that of numSeqs. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>slotMapping</td>
      <td>Input/Output</td>
      <td>Input/Output parameter for the AdvanceStep computation, which is the output slotMapping in the formula. It is used to map the position of the token value in the sequence to the physical position.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The shape length is the same as that of numSeqs. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>blockTables</td>
      <td>Input</td>
      <td>Input parameter for the AdvanceStep computation, which is used to record the block size under different blockIdx. It corresponds to the input blockTables in the formula.</td>
      <td><ul><li>Empty tensors are not supported. </li><li>The length of the first dimension of the shape is the same as that of numSeqs, and the length of the second dimension is greater than (maximum value in seqLens)/blockSize. </li><li>The value is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr> 
      <tr>
      <td>numSeqs</td>
      <td>Input</td>
      <td>Number of input seqs. The size is the same as the length of seqLens.</td>
      <td><ul><li>The value is a positive integer greater than 0. </li><li>The value of numSeqs is greater than that of input numQueries.</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr> 
      <tr>
      <td>numQueries</td>
      <td>Input</td>
      <td>Number of input queries. The size is the same as the length of the first dimension of sampledTokenIds.</td>
      <td>The value is a positive integer greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr> 
     <tr>
      <td>blockSize</td>
      <td>Input</td>
      <td>Size of each block, which corresponds to blockSize in the formula.</td>
      <td>The value is a positive integer greater than 0.</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
  </tbody>
  </table>
  
- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
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
      <td>The passed inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, or blockTables is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, or blockTables is not supported.</td>
    </tr>
    <tr>
      <td rowspan="3">aclnnAdvanceStepGetWorkspaceSize failed</td>
      <td rowspan="3">561002</td>
      <td>The length of the first dimension of the shape of inputTokens, inputPositions, seqLens, slotMapping, or blockTables is inconsistent with that of numSeqs.</td>
    </tr>
    <tr>
      <td>The length of the first dimension of the shape of sampledTokenIds is inconsistent with that of numQueries, or the length of the second dimension of the shape is not 1.</td>
    </tr>
    <tr>
      <td>The value of numSeqs is less than or equal to that of numQueries.</td>
    </tr>
  </tbody>
  </table>

  
## aclnnAdvanceStep

- **Parameters:**

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
      <td>Size of the workspace to be allocated on the device, obtained by calling the first-phase API aclnnAdvanceStepGetWorkspaceSize.</td>
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

- Deterministic compute:
  - **aclnnAdvanceStep** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_advance_step.h"
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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %ld\n", i, resultData[i]);
    }
}

int Init(int64_t deviceId, aclrtStream* stream) {
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
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output based on the API.
    std::vector<int64_t> inputShape = {8,1}; 
    std::vector<int64_t> input2Shape = {4,1}; 
    std::vector<int64_t> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<int64_t> input2HostData = {0, 1, 2, 3};

    void* input1DeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    void* input2DeviceAddr = nullptr;
    aclTensor* input2 = nullptr;
    void* input3DeviceAddr = nullptr;
    aclTensor* input3 = nullptr;
    void* input4DeviceAddr = nullptr;
    aclTensor* input4 = nullptr;
    void* input5DeviceAddr = nullptr;
    aclTensor* input5 = nullptr;
    void* input6DeviceAddr = nullptr;
    aclTensor* input6 = nullptr;
    // Create an input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &input1DeviceAddr, aclDataType::ACL_INT64, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(input2HostData, input2Shape, &input2DeviceAddr, aclDataType::ACL_INT64, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input3DeviceAddr, aclDataType::ACL_INT64, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input4DeviceAddr, aclDataType::ACL_INT64, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input5DeviceAddr, aclDataType::ACL_INT64, &input5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(inputHostData, inputShape, &input6DeviceAddr, aclDataType::ACL_INT64, &input6);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numseq = 8;
    int64_t numqueries = 4;
    int64_t blocksize = 2;

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnAdvanceStep.
    ret = aclnnAdvanceStepGetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,
    numseq,numqueries,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnAdvanceStep.
    ret = aclnnAdvanceStep(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStep failed. ERROR: %d\n", ret); return ret);

    // 4. (Fixed writing) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
    PrintOutResult(inputShape, &input1DeviceAddr);
    PrintOutResult(inputShape, &input3DeviceAddr);
    PrintOutResult(inputShape, &input4DeviceAddr);
    PrintOutResult(inputShape, &input5DeviceAddr);

    // 6. Release aclTensors. Modify the configuration based on the API definition.
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);

    // 7. Release device resources. Modify the configuration based on the API definition.
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(input5DeviceAddr);
    aclrtFree(input6DeviceAddr);
    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
