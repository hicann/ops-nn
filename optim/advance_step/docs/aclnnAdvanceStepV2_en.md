# aclnnAdvanceStepV2

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:42:15.249Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
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

- Description:

  vLLM is a high-performance LLM inference and serving framework focused on optimizing the inference efficiency of large-scale language models. Its core features include PageAttention and efficient memory management. The primary function of the advance_step operator is to advance the inference step, i.e., updating the model state at each generation step and generating new inputTokens, inputPositions, seqLens, and slotMapping, thereby improving efficiency for vLLM inference.

- Formula:

  $
  blockIdx is the index of the kernel where the current code is executed.
  $

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

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnAdvanceStepV2GetWorkspaceSize" API first to obtain the input parameters and calculate the required workspace size based on the calculation flow, and then call the "aclnnAdvanceStepV2" API to perform the calculation.

```Cpp
aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(
  const aclTensor *inputTokens,
  const aclTensor *sampledTokenIds,
  const aclTensor *inputPositions,
  const aclTensor *seqLens,
  const aclTensor *slotMapping,
  const aclTensor *blockTables,
  const aclTensor *specToken,
  const aclTensor *acceptedNum,
  int64_t          numSeqs,
  int64_t          numQueries,
  int64_t          blockSize,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAdvanceStepV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnAdvanceStepV2GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 250px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>inputTokens(aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input/output parameter for the AdvanceStepV2 calculation to be performed, output inputTokens in the formula, used to update the token values in the vLLM model.</td>
      <td><ul><li>Empty tensor not supported.</li><li>The shape is [numSeqs * (1+specNum)].</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sampledTokenIds (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the AdvanceStepV2 calculation to be performed, used to store token IDs, input sampledTokenIds in the formula.</td>
      <td><ul><li>Empty tensors not supported.</li><li>The length of the first dimension of shape is consistent with numSeqs, and the length of the second dimension is 1+specNum.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
     <tr>
      <td>inputPositions (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input/output parameter of the AdvanceStepV2 computation to be performed. It is the output inputPositions in the formula, used to record the token index.</td>
      <td><ul><li>Empty tensors not supported.</li><li>The shape is [numSeqs * (1+specNum)].</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqLens (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input/output parameter for the AdvanceStepV2 calculation to be performed, used to record the length of seq for different blockIdx, that is, the input/output seqLens in the formula.</td>
      <td><ul><li>Empty Tensor not supported.</li><li>The shape is [numSeqs * (1+specNum)].</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>slotMapping (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input for calculation/output parameter of the AdvanceStepV2 calculation to be performed. It is the output slotMapping in the formula, used to map the position of a token in the sequence to the physical position.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape is [numSeqs * (1+specNum)].</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockTables (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the AdvanceStepV2 calculation to be performed, used to record the physical block IDs of blocks under different blockIdx. It is the input blockTables in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The length of the first dimension of shape is consistent with numSeqs, and the second dimension is greater than (max value in seqLens)/blockSize.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>specToken (aclTensor*)</td>
      <td>Input</td>
      <td>Input for the AdvanceStepV2 calculation to be performed, used to record the token index of the current speculative model.</td>
      <td><ul><li>Optional input; passing an empty tensor is regarded as not passing it.</li><li>When it is not empty, the length of the first dimension of the shape is consistent with numSeqs, and the length of the second dimension is specNum, which must be greater than 0.</li><li>The value range is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>acceptedNum (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for the AdvanceStepV2 calculation to be performed, used to record the number of speculative tokens accepted by each request.</td>
      <td><ul><li>Empty tensors not supported.</li><li>The shape length is consistent with numSeqs.</li><li>The value range is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>numSeqs(int64_t)</td>
      <td>Input</td>
      <td>Records the number of input sequences. The size must be consistent with the length of seqLens.</td>
      <td><ul><li>Value range: positive integers greater than 0.</li><li>The value of numSeqs is equal to the value of input numQueries.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>numQueries (int64_t)</td>
      <td>Input</td>
      <td>Records the number of input queries. Its size is consistent with the length of the first dimension of sampledTokenIds.</td>
      <td>Value range: positive integers greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>blockSize (int64_t)</td>
      <td>Input</td>
      <td>Size of each block, corresponding to blockSize in the formula.</td>
      <td>Value range: positive integers greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace to be applied by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
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

- **Return value**

aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

The first-phase API performs input parameter validation and returns an error in the following scenarios:

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
</colgroup>
<thead>
  <tr>
    <th>Error Code</th>
    <th>Error Code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>The input inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, and blockTables are null pointers (specToken and acceptedNum are optional inputs and may be null pointers).</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data types of inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, blockTables, specToken, and acceptedNum are not within the supported range.</td>
  </tr>
  <tr>
    <td rowspan="5">aclnnAdvanceStepV2GetWorkspaceSize failed</td>
    <td rowspan="5">561002</td>
    <td>The shape of input inputTokens, inputPositions, seqLens, or slotMapping is not [numSeqs * (1+specNum)], or the length of the first dimension of the shape of blockTables, specToken, or acceptedNum is inconsistent with numSeqs.</td>
  </tr>
  <tr>
    <td>The length of the first dimension of the shape of input sampledTokenIds is inconsistent with numSeqs, or the length of the second dimension of the shape is not 1+specNum.</td>
  </tr>
  <tr>
    <td>The length of the shape of input acceptedNum is inconsistent with numSeqs.</td>
  </tr>
    <tr>
    <td>When input specToken is not empty, the length of the first dimension of the shape is inconsistent with numSeqs, or the number of dimensions is not 2.</td>
  </tr>
    <tr>
    <td>The value of input numSeqs is not equal to the value of input numQueries.</td>
  </tr>
</tbody>
</table>

## aclnnAdvanceStepV2

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
      <td>Workspace memory address applied by the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size applied by the Device side, obtained by the first-phase API aclnnAdvanceStepV2GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, which contains the operator calculation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Interface path constraints:
  - The V2 path requires specToken and acceptedNum to be passed together and to be non-empty, specNum (the length of the second dimension of specToken) to be greater than 0, and numSeqs to be equal to numQueries.
  - When only one of specToken and acceptedNum is passed, or neither is passed (including passing an empty tensor; when specNum is 0, specToken is an empty tensor), the V1 path is used. The V1 path requires numSeqs to be greater than numQueries; otherwise, 561002 is returned.
  - For the no-speculative-decoding scenario, evaluate aclnnAdvanceStep, but the numSeqs and numQueries constraints of the V1 path differ from those of the V2 path, so it is not a direct equivalent replacement.
- deterministic computation:
  - aclnnAdvanceStepV2 defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_advance_step_v2.h"
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
    LOG_PRINT("advanceStepV2 result[%ld] is: %ld\n", i, resultData[i]);
    }
}

int Init(int64_t deviceId, aclrtStream* stream) {
    // Boilerplate for resource initialization.
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
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to the device-side memory.
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // Calculate the strides of the contiguous tensor.
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
    }

    // Call the aclCreateTensor API to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> input1Shape = {72};
    std::vector<int64_t> input2Shape = {8,9};
    std::vector<int64_t> input3Shape = {72};
    std::vector<int64_t> input4Shape = {72};
    std::vector<int64_t> input5Shape = {72};
    std::vector<int64_t> input6Shape = {8,1000};
    std::vector<int64_t> input7Shape = {8,8};
    std::vector<int64_t> input8Shape = {8};
    std::vector<int64_t> input1HostData(8 * 9, 1);
    std::vector<int64_t> input2HostData(8 * 9, 1);
    std::vector<int64_t> input3HostData(8 * 9, 1);
    std::vector<int64_t> input4HostData(8 * 9, 8);
    std::vector<int64_t> input5HostData(8 * 9, 1);
    std::vector<int64_t> input6HostData(8 * 1000, 1);
    std::vector<int64_t> input7HostData(8 * 8, 1);
    std::vector<int64_t> input8HostData(8, 1);

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
    void* input7DeviceAddr = nullptr;
    aclTensor* input7 = nullptr;
    void* input8DeviceAddr = nullptr;
    aclTensor* input8 = nullptr;
    // Create the input aclTensor.
    ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, aclDataType::ACL_INT64, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input2HostData, input2Shape, &input2DeviceAddr, aclDataType::ACL_INT64, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input3HostData, input3Shape, &input3DeviceAddr, aclDataType::ACL_INT64, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input4HostData, input4Shape, &input4DeviceAddr, aclDataType::ACL_INT64, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input5HostData, input5Shape, &input5DeviceAddr, aclDataType::ACL_INT64, &input5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input6HostData, input6Shape, &input6DeviceAddr, aclDataType::ACL_INT64, &input6);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input7HostData, input7Shape, &input7DeviceAddr, aclDataType::ACL_INT64, &input7);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input8HostData, input8Shape, &input8DeviceAddr, aclDataType::ACL_INT64, &input8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numseq = 8;
    int64_t specnum = 8;
    int64_t blocksize = 8;

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnAdvanceStepV2.
    ret = aclnnAdvanceStepV2GetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,input7,input8,
    numseq,numseq,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second-phase API of aclnnAdvanceStepV2.
    ret = aclnnAdvanceStepV2(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2 failed. ERROR: %d\n", ret); return ret);

    // 4. (boilerplate) Synchronously wait for the task execution to complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device side to the host side. The code needs to be modified based on the specific API definition.
    PrintOutResult(input1Shape, &input1DeviceAddr);
    PrintOutResult(input2Shape, &input2DeviceAddr);
    PrintOutResult(input3Shape, &input3DeviceAddr);
    PrintOutResult(input4Shape, &input4DeviceAddr);
    PrintOutResult(input5Shape, &input5DeviceAddr);

    // 6. Release aclTensor. The code needs to be modified based on the specific API definition.
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);
    aclDestroyTensor(input7);
    aclDestroyTensor(input8);

    // 7. Release device resources. The code needs to be modified based on the specific API definition.
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(input5DeviceAddr);
    aclrtFree(input6DeviceAddr);
    aclrtFree(input7DeviceAddr);
    aclrtFree(input8DeviceAddr);

    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
