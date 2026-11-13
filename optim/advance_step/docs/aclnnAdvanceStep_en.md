# aclnnAdvanceStep

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T20:40:59.332Z -->

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

  Let $blockTablesFlat$ be the row-flattened one-dimensional view of $blockTables$:

  $$
  blockTablesStride = blockTables.stride(0)
  $$

  Let $i \in [0, numQueries)$ be the query row index, $pos_i$ the pre-update $seqLens[i]$:

  $$
  inputTokens[i] = sampledTokenIds[i, 0]
  $$

  $$
  inputPositions[i] = pos_i
  $$

  $$
  seqLens[i] = pos_i + 1
  $$

  $$
  slotMapping[i] = blockTablesFlat[blockTablesStride * i + \lfloor pos_i / blockSize \rfloor] * blockSize + (pos_i \% blockSize)
  $$

  Padding behavior: when numSeqs is greater than numQueries, the kernel writes inputTokens[j]=0, inputPositions[j]=0, and slotMapping[j]=-1 for all padding rows $j \in [numQueries, numSeqs)$, while the padding rows of seqLens keep their input values. This write-back acts on the corresponding regions of the caller's input tensors, and the caller must ensure they are writable.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must call "aclnnAdvanceStepGetWorkspaceSize" first to obtain the input parameters and calculate the required workspace size based on the calculation process, and then call "aclnnAdvanceStep" to perform the calculation.

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
      <td>inputTokens (aclTensor*)</td>
      <td>Input</td>
      <td>Input/output parameter for the AdvanceStep computation. It is output inputTokens in the formula, used to update the token values in the vLLM model.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape length and numSeqs are consistent.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sampledTokenIds (aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for the AdvanceStep computation, used to store token IDs. It is input sampledTokenIds in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The length of the first dimension of the shape is consistent with numQueries, and the length of the second dimension is 1.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>inputPositions (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input parameter/output parameter for the AdvanceStep computation to be performed. It is the output inputPositions in the formula and is used to record the index of tokens.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The length of the shape is consistent with numSeqs.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>seqLens (aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input/output parameter for the AdvanceStep computation to be performed, used to record the length of seq for different blockIdx values, corresponding to the input/output seqLens in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape length and numSeqs are consistent.</li><li>The value range is a positive integer greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>slotMapping(aclTensor*)</td>
      <td>Input/Output</td>
      <td>Input/output parameter for the AdvanceStep computation to be performed. It is the output slotMapping in the formula, used to map the position of a token value in a sequence to a physical location.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The shape length and numSeqs must be consistent.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>blockTables(aclTensor*)</td>
      <td>Input</td>
      <td>Input parameter for the AdvanceStep computation to be performed, used to record the physical block IDs of blocks under different blockIdx values. It is the input blockTables in the formula.</td>
      <td><ul><li>Empty tensors are not supported.</li><li>The first dimension length of shape must be consistent with numSeqs, and the second dimension must be greater than (the maximum value in seqLens)/blockSize.</li><li>The value range is positive integers greater than 0.</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>numSeqs (int64_t)</td>
      <td>Input</td>
      <td>Records the number of input sequences. The size is consistent with the length of seqLens.</td>
      <td><ul><li>The value range is positive integers less than 200000000.</li><li>The value of numSeqs is greater than the value of the input numQueries.</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>numQueries (int64_t)</td>
      <td>Input</td>
      <td>Records the number of queries in the input. The size is consistent with the length of the first dimension of sampledTokenIds.</td>
      <td>The value range is positive integers greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>blockSize (int64_t)</td>
      <td>Input</td>
      <td>Size of each block, corresponding to blockSize in the formula.</td>
      <td>The value range is positive integers greater than 0.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the workspace size that needs to be requested by the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>Output</td>
      <td>Returns the op executor containing the operator computation flow.</td>
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
    <td>The input inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, and blockTables are null pointers.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>The data types of inputTokens, sampledTokenIds, inputPositions, seqLens, slotMapping, and blockTables are not within the supported range.</td>
  </tr>
  <tr>
    <td rowspan="3">aclnnAdvanceStepGetWorkspaceSize failed</td>
    <td rowspan="3">561002</td>
    <td>The first dimension length and numSeqs of the shapes of inputs inputTokens, inputPositions, seqLens, slotMapping, and blockTables are inconsistent.</td>
  </tr>
  <tr>
    <td>The length of the first dimension of the shape of input sampledTokenIds is inconsistent with numQueries, or the length of the second dimension of the shape is not 1.</td>
  </tr>
  <tr>
    <td>The value of input numSeqs is less than or equal to the value of input numQueries.</td>
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
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace memory requested by the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace requested by the Device, obtained by calling the first-phase API aclnnAdvanceStepGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the computation flow of the operator.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream where the task is executed.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Out-of-range inputs (such as negative positions or positions beyond the blockTables coverage) are out-of-contract inputs: when the lookup column index falls outside the row-local blockTables coverage ($q = pos/blockSize \notin [0, blockTablesStride)$), the kernel defensively skips the GM read and writes slotMapping=0 for that row (both the block number and the remainder are masked). This behavior is defined as the public contract, and golden mirrors it.
- Deterministic computation:
  - aclnnAdvanceStep defaults to a deterministic implementation.

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

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
    LOG_PRINT("advanceStep result[%ld] is: %ld\n", i, resultData[i]);
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
    // Call aclrtMalloc to request device-side memory.
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
    // Fill in deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the input and output. Custom construction is required based on the API.
    std::vector<int64_t> inputShape = {8};
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
    // Create the input aclTensor.
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

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // Call the first-phase API of aclnnAdvanceStep.
    ret = aclnnAdvanceStepGetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,
    numseq,numqueries,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Request device memory based on the workspaceSize calculated by the first-phase API.
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

    // 4. (boilerplate) Synchronously wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host memory. Modify it based on the actual API definition.
    PrintOutResult(inputShape, &input1DeviceAddr);
    PrintOutResult(inputShape, &input3DeviceAddr);
    PrintOutResult(inputShape, &input4DeviceAddr);
    PrintOutResult(inputShape, &input5DeviceAddr);

    // 6. Release aclTensor and aclTensor. Modify it based on the actual API definition.
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);

    // 7. Release device resources. Modify it based on the actual API definition.
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
