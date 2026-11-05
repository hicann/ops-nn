# aclnnExpandIntoJaggedPermute

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:14:32.375Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
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

- **Description**: Expands sparse data permutation indices from the table dimension to the batch dimension. Applicable to use cases where sparse features have different batch sizes across ranks.
- **Formula**:

$$
len = outputOffsets[i+1] - outputOffsets[i]
$$

$$
out[outputOffsets[i]:outputOffsets[i+1]] = arange(inputOffsets[permute[i]],inputOffsets[permute[i]]+len)
$$

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). First call "aclnnExpandIntoJaggedPermuteGetWorkspaceSize" to obtain the workspace size required for the computation and the executor containing the operator computation process, and then call "aclnnExpandIntoJaggedPermute" to perform the computation.

```c++
aclnnStatus aclnnExpandIntoJaggedPermuteGetWorkspaceSize(
    const aclTensor *permute,
    const aclTensor *inputOffsets,
    const aclTensor *outputOffsets,
    int64_t          outputSize,
    const aclTensor *out,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnExpandIntoJaggedPermute(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnExpandIntoJaggedPermuteGetWorkspaceSize

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 187px">
    <col style="width: 121px">
    <col style="width: 287px">
    <col style="width: 387px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 187px">
    <col style="width: 146px">
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
        <td>permute</td>
        <td>Input</td>
        <td>The input permute in the formula, representing the table-level permutation index.</td>
        <td>Required to be a one-dimensional tensor, with a value range of [0,inputLen-1].</td>
        <td>INT32</td>
        <td>ND</td>
        <td>inputLen</td>
        <td>√</td>
    </tr>
    <tr>
        <td>inputOffsets</td>
        <td>Input</td>
        <td>The inputOffsets in the formula, representing the mutually exclusive offset of table-level lengths.</td>
        <td>Required to be a one-dimensional tensor starting at 0. When permute is non-empty, its values must be strictly increasing.</td>
        <td>Consistent with permute</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputOffsets</td>
        <td>Input</td>
        <td>outputOffsets in the formula, indicating the target offset in the flattened output space.</td>
        <td>Required to be a one-dimensional tensor starting at 0. When permute is non-empty, its values must be strictly increasing, and the last value must equal outputSize.</td>
        <td>Consistent with permute</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputSize</td>
        <td>Input</td>
        <td>Represents the length of the output.</td>
        <td>-</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>out</td>
        <td>Output</td>
        <td>Output in the formula.</td>
        <td>The shape is (outputSize).</td>
        <td>Consistent with permute.</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
    </tr>
    <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be applied on the Device side.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor that contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    </tbody></table>

- **Return value**

    aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API performs input parameter validation and returns an error in the following scenarios:

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 267px">
        <col style="width: 124px">
        <col style="width: 775px">
        </colgroup>
        <thead>
            <tr>
            <th>Return Value</th>
            <th>Error Code</th>
            <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
            <td> ACLNN_ERR_PARAM_NULLPTR </td>
            <td> 161001 </td>
            <td>A required input, required output, or required attribute is a null pointer.</td>
            </tr>
            <tr>
            <td> ACLNN_ERR_PARAM_INVALID </td>
            <td> 161002 </td>
            <td>The data types and data formats of inputs and outputs are not supported.</td>
            </tr>
            <tr>
            <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
            <td rowspan="2"> 561002 </td>
            <td>The shape information of multiple input tensors does not match.</td>
            </tr>
            <tr>
            <td>The shape information between input attributes and input tensors does not match.</td>
            </tr>
        </tbody></table>

## aclnnExpandIntoJaggedPermute

- **Parameters:**

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 173px">
        <col style="width: 133px">
        <col style="width: 860px">
        </colgroup>
            <thead>
                <tr>
                <th>Parameter</th>
                <th>Input/Output</th>
                <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                <td>workspace</td>
                <td>Input</td>
                <td>The workspace memory address applied on the Device side.</td>
                </tr>
                <tr>
                <td>workspaceSize</td>
                <td>Input</td>
                <td>The workspace size applied on the Device side, obtained by the first-phase API aclnnExpandIntoJaggedPermuteGetWorkspaceSize.</td>
                </tr>
                <tr>
                <td>executor</td>
                <td>Input</td><td> op executor containing the operator computation flow.</td>
                </tr>
                <tr>
                <td>stream</td>
                <td>Input</td>
                <td> Stream that specifies the task to be executed.</td>
                </tr>
            </tbody>
        </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnExpandIntoJaggedPermute defaults to a deterministic implementation.

1. The shapes of inputOffsets and outputOffsets must be the same.

2. permute, inputOffsets, outputOffsets, and out must have the same data type.

3. inputOffsets and outputOffsets must start at 0 and, when permute is non-empty, must be strictly increasing. The last
   value of outputOffsets must equal outputSize.

4. Let P be the length of permute. When P is 0, outputSize must be 0. When P is greater than 0, outputSize must be
   greater than or equal to P.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include "aclnnop/aclnn_expand_into_jagged_permute.h"
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cassert>
#include <iomanip>
#include <unistd.h>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

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

template <typename T>
bool ReadFile(const std::string &filePath, std::vector<int64_t> shape, std::vector<T>& hostData)
{
    size_t fileSize = 1;
    for (int64_t i : shape){
        fileSize *= i;
    }
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open the file." << std::endl;
        return 1;
    }
    // Obtain the file size.
    file.seekg(0, std::ios::end);
    file.seekg(0, std::ios::beg);
    hostData.resize(fileSize);
    if (file.read(reinterpret_cast<char*>(hostData.data()), fileSize * sizeof(T))) {
    } else {
        std::cerr << "Failed to read the file." << std::endl;
        return 1;
    }
    file.close();
    return true;
}

template <typename T>
bool WriteFile(const std::string &filePath, int64_t size, std::vector<T>& hostData)
{
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd < 0) {
        LOG_PRINT("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    size_t writeSize = write(fd, reinterpret_cast<char*>(hostData.data()), size * sizeof(T));
    (void)close(fd);
    if (writeSize != size * sizeof(T)) {
        LOG_PRINT("Write file Failed.");
        return false;
    }

    return true;
}
void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < 10; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // Call aclrtMalloc to apply for device-side memory.
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // Call aclrtMemcpy to copy host-side data to device-side memory.
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
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> permuteShape = {3};
  std::vector<int64_t>inputOffsetsShape = {4};
  std::vector<int64_t> outputOffsetsShape = {4};
  std::vector<int64_t> outputPermuteShape= {6};
  void* permuteDeviceAddr = nullptr;
  void* inputOffsetsDeviceAddr = nullptr;
  void* outputOffsetsDeviceAddr = nullptr;
  void* outputPermuteDeviceAddr = nullptr;

  aclTensor* permute = nullptr;
  aclTensor* inputOffsets = nullptr;
  aclTensor* outputOffsets = nullptr;
  aclTensor* outputPermute = nullptr;
  int64_t outputSize = 6;

  std::vector<int32_t> permuteHostData = {1, 0, 2};
  std::vector<int32_t> inputOffsetsHostData = {0, 3, 5, 8};
  std::vector<int32_t>outputOffsetsHostData = {0, 2, 4, 6};
  std::vector<int32_t> outputPermuteHostData = {3, 4, 0, 1, 5, 6};

  ret = CreateAclTensor(permuteHostData, permuteShape,
                        &permuteDeviceAddr, aclDataType::ACL_INT32,
                        &permute);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(inputOffsetsHostData, inputOffsetsShape, &inputOffsetsDeviceAddr,
                      aclDataType::ACL_INT32, &inputOffsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outputOffsetsHostData, outputOffsetsShape, &outputOffsetsDeviceAddr,
                      aclDataType::ACL_INT32, &outputOffsets);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outputPermuteHostData , outputPermuteShape , &outputPermuteDeviceAddr,
                      aclDataType::ACL_INT32, &outputPermute);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Change it to the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the aclnnExpandIntoJaggedPermute first-phase API.
  ret = aclnnExpandIntoJaggedPermuteGetWorkspaceSize(permute, inputOffsets, outputOffsets,
                                               outputSize, outputPermute, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnExpandIntoJaggedPermuteGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on the workspaceSize calculated by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the aclnnExpandIntoJaggedPermute second-phase API.
  ret = aclnnExpandIntoJaggedPermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnExpandIntoJaggedPermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Wait for the task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value. Copy the result from device memory to host memory. Modify it based on the specific API definition.
  PrintOutResult(outputPermuteShape, &outputPermuteDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code according to the specific API definition.
  aclDestroyTensor(permute);
  aclDestroyTensor(inputOffsets);
  aclDestroyTensor(outputOffsets);
  aclDestroyTensor(outputPermute);

  // 7. Release device resources.
  aclrtFree(permuteDeviceAddr);
  aclrtFree(inputOffsetsDeviceAddr);
  aclrtFree(outputOffsetsDeviceAddr);
  aclrtFree(outputPermuteDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
