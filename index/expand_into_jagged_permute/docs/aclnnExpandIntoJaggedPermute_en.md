# aclnnExpandIntoJaggedPermute

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/index/expand_into_jagged_permute)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |


## Function

- **Description**: Extends the permutation indices of sparse data from the table dimension to the batch dimension, applicable to scenarios where sparse features have different batch sizes across ranks.
- **Formula**:

$$
len = outputOffset[i+1] - outputOffset[i]
$$

$$
outputPermuteOut[outputOffset[i]:outputOffset[i+1]] = arange(inputOffset[permute[i]],inputOffset[permute[i]]+len)
$$


## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnExpandIntoJaggedPermuteGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnExpandIntoJaggedPermute** is called to perform computation.

```c++
aclnnStatus aclnnExpandIntoJaggedPermuteGetWorkspaceSize(
    const aclTensor *permute,
    const aclTensor *inputOffset,
    const aclTensor *outputOffset,
    int64_t          outputSize,
    const aclTensor *outputPermuteOutOut,
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

- **Parameters**

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
        <td>permute</td>
        <td>Input</td>
        <td>Table-level permutation indices, corresponding to permute in the formula.</td>
        <td>The tensor must be one-dimensional, with values in the range [0, inputLen – 1].</td>
        <td>INT32</td>
        <td>ND</td>
        <td>inputLen</td>
        <td>√</td>
    </tr>
    <tr>
        <td>inputOffset</td>
        <td>Input</td>
        <td>Mutually exclusive offsets of table-level lengths, corresponding to inputOffset in the formula.</td>
        <td>The tensor must be one-dimensional, with strictly monotonically increasing values, and the last value equals outputSize.</td>
        <td>Same as permute.</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputOffset</td>
        <td>Input</td>
        <td>First shared expert, corresponding to outputOffset in the formula.</td>
        <td>The tensor must be one-dimensional, with strictly monotonically increasing values, and the last value equals outputSize.</td>
        <td>Same as permute.</td>
        <td>ND</td>
        <td>inputLen+1</td>
        <td>√</td>
    </tr>
    <tr>
        <td>outputSize</td>
        <td>Input</td>
        <td>Length of the output result.</td>
        <td>-</td>
        <td>INT32, INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>outputPermute</td>
        <td>Output</td>
        <td>Output tensor, corresponding to the output in the formula.</td>
        <td>The shape is (outputSize).</td>
        <td>Same as permute.</td>
        <td>ND</td>
        <td>-</td>
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
    </tbody></table>

- **Returns**

    **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

    The first-phase API implements input parameter verification. The following errors may be thrown.

    <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
        <col style="width: 267px">
        <col style="width: 124px">
        <col style="width: 775px">
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
            <td> ACLNN_ERR_PARAM_NULLPTR </td>
            <td> 161001 </td>
            <td>The passed mandatory input, output, or attribute is a null pointer.</td>
            </tr>
            <tr>
            <td> ACLNN_ERR_PARAM_INVALID </td>
            <td> 161002 </td>
            <td>The data type or format of the input or output is not supported.</td>
            </tr>
            <tr>
            <td rowspan="2"> ACLNN_ERR_INNER_TILING_ERROR </td>
            <td rowspan="2"> 561002 </td>
            <td>The shapes of multiple input tensors do not match.</td>
            </tr>
            <tr>
            <td>The shape of the input attribute does not match that of the input tensor.</td>
            </tr>
        </tbody></table>

## aclnnExpandIntoJaggedPermute

- **Parameters**

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
                </tr>
            </thead>
            <tbody>
                <tr>
                <td>workspace</td>
                <td>Input</td>
                <td>Address of the workspace to be allocated on the device.</td>
                </tr>
                <tr>
                <td>workspaceSize</td>
                <td>Input</td>
                <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnExpandIntoJaggedPermuteGetWorkspaceSize.</td>
                </tr>
                <tr>
                <td>executor</td>
                <td>Input</td><td>Operator executor, containing the operator computation flow.</td>
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
  - **aclnnExpandIntoJaggedPermute** defaults to a deterministic implementation.

1. The shapes of **inputOffset** and **outputOffset** must be the same.

2. The data types of **permute**, **inputOffset**, **outputOffset**, and **outputPermuteOut** must be the same.

3. The values of **outputOffset** are strictly monotonically increasing, and the last value equals **outputSize**.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
    hostData.reserve(fileSize);
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
    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
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
                    aclDataType dataType, aclTensor** tensor)
{
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

    // Call aclCreateTensor to create an aclTensor.
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID (deviceId) based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. Construct inputs and outputs based on API definitions.
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

  // 3. Call the CANN operator library API. Modify the API as required.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // Call the first-phase API of aclnnExpandIntoJaggedPermute.
  ret = aclnnExpandIntoJaggedPermuteGetWorkspaceSize(permute, inputOffsets, outputOffsets, 
                                               outputSize, outputPermute, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnExpandIntoJaggedPermuteGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // Call the second-phase API of aclnnExpandIntoJaggedPermute.
  ret = aclnnExpandIntoJaggedPermute(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnExpandIntoJaggedPermute failed. ERROR: %d\n", ret);
            return ret);

  // 4. (Boilerplate) Synchronize the stream and wait for task completion.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.
  PrintOutResult(outputPermuteShape, &outputPermuteDeviceAddr);

  // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
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
