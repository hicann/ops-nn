# aclnnCrossEntropyLoss

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T17:12:21.402Z -->

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

- Description: Computes the cross-entropy loss of the input.
- Formula:

  When reductionOptional = mean, the cross-entropy loss is calculated as:

  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$

  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reductionOptional = 'mean'} \\\sum_{n=1}^Nl_n,&\text {if reductionOptional = ‘sum’}\\\{l_0,l_1,...,l_n\},&\text{if reductionOptional = ‘None’}\end{cases}
  $$

  log\_prob formula:

  $$
  lse_n = log\sum_{c=1}^{C}exp(x_{n,c})
  $$

  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$

  The zloss formula is:

  $$
  zloss_n = lseSquareScaleForZloss * (lse_n)^2
  $$

  Where N is the batch size and C is the number of labels.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call the "aclnnCrossEntropyLossGetWorkspaceSize" API first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator. Then, call the "aclnnCrossEntropyLoss" API to perform the computation.

```Cpp
aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(
    const aclTensor *input,
    const aclTensor *target,
    const aclTensor *weightOptional,
    char            *reductionOptional,
    int64_t          ignoreIndex,
    double           labelSmoothing,
    double           lseSquareScaleForZloss,
    bool             returnZloss,
    const aclTensor *lossOut,
    const aclTensor *logProbOut,
    const aclTensor *zlossOut,
    const aclTensor *lseForZlossOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCrossEntropyLoss(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnCrossEntropyLossGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
    <col style="width: 208px">
    <col style="width: 120px">
    <col style="width: 256px">
    <col style="width: 226px">
    <col style="width: 149px">
    <col style="width: 111px">
    <col style="width: 155px">
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
    <td>input (aclTensor*)</td>
    <td>Input</td>
    <td>x in the formula.</td>
    <td>-</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(N,C)<br>N indicates the batch size, and C indicates the number of labels. The value must be greater than 0.</td>
    <td>√</td>
    </tr>
    <tr>
    <td>target (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates the label, y in the formula.</td>
    <td>-</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>(N)<br>N is the same as the 0th dimension of input. The value range is [0, C). When ignoreIndex is specified, the value of target can also be equal to ignoreIndex.</td>
    <td>√</td>
    </tr>
    <tr>
    <td>weightOptional (aclTensor*)</td>
    <td>Input</td>
    <td>Indicates the scaling weight specified for each class, which is the weight in the Formula.</td>
    <td>If not specified, no weight is applied to target.</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>(C)</td>
    <td>√</td>
    </tr>
    <tr>
    <td>reductionOptional (char*)</td>
    <td>Input</td>
    <td>Indicates the reduction mode of the loss.</td>
    <td>Supported values are ["mean", "sum", "none"].</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>ignoreIndex (int64_t)</td>
    <td>Input</td>
    <td>Specifies the label value to be ignored.</td>
    <td>-</td>
    <td>INT64</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>labelSmoothing (double)</td>
    <td>Input</td>
    <td>Indicates the smoothing amount during loss computation.</td>
    <td>The value is in the range [0.0, 1.0].</td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseSquareScaleForZloss (double)</td>
    <td>Input</td>
    <td>Indicates the scale required for zloss computation. It is lse_square_scale_for_zloss in the formula.</td>
    <td>The value is in the range [0, 1). Currently not supported.</td>
    <td>DOUBLE</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnZloss (bool)</td>
    <td>Input</td>
    <td>Controls whether to return the zloss output.</td>
    <td>Pass True to output zLoss; otherwise pass False. Currently not supported.</td>
    <td>BOOL</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lossOut (aclTensor*)</td>
    <td>Output</td>
    <td>Indicates the output loss.</td>
    <td>If reductionOptional is "none", the shape is [N], the same as the zeroth dimension of input; otherwise, the shape is [1].</td>
    <td>Same as input</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>logProbOut (aclTensor*)</td>
    <td>Output</td>
    <td>Output for backward propagation.</td>
    <td>-</td>
    <td>Same as input</td>
    <td>ND</td>
    <td>(N,C)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zlossOut (aclTensor*)</td>
    <td>Output</td>
    <td>Indicates the auxiliary loss.</td>
    <td>Currently not supported. It must be passed in during computation but does not participate in the computation.</td>
    <td>Same as input</td>
    <td>ND</td>
    <td>Same as lossOut</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseForZlossOut (aclTensor*)</td>
    <td>Output</td>
    <td>Indicates the tensor output to backward in the zloss use case. If lseSquareScaleForZloss is 0, the output is None.</td>
    <td>Currently not supported. This parameter must be passed in for computation but does not participate in the computation.</td>
    <td>Same as input</td>
    <td>ND</td>
    <td>(N)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize (uint64_t*)</td>
    <td>Output</td>
    <td>Returns the size of the workspace to be allocated on the Device.</td>
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

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 319px">
    <col style="width: 125px">
    <col style="width: 800px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>input, target, lossOut, logProbOut, zlossOut, or lseForZlossOut is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The data types of input, target, lossOut, logProbOut, zlossOut, and lseForZlossOut are not in the supported range.</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>The shapes of input, target, or weightOptional are not in the supported range, or the shapes of lossOut/zlossOut are inconsistent with reductionOptional (they should match dim 0 of input when reductionOptional is "none", otherwise be [1]).</td>
      </tr>
      <tr>
        <td>The values of reductionOptional and labelSmoothing are not in the supported range.</td>
      </tr>
    </tbody>
    </table>

## aclnnCrossEntropyLoss

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Workspace memory address requested on the device side.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size requested on the device side. It is obtained by the first-phase API aclnnCrossEntropyLossGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor, containing the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the Stream for task execution.</td>
      </tr>
      </tbody>
    </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

  - target supports only class label indices, not probability inputs.
  - zloss-related features are currently not supported. Passing related inputs, i.e., lseSquareScaleForZloss and returnZloss, will not take effect.

  - Deterministic computation:
    - aclnnCrossEntropyLoss defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss.h"

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
  // Boilerplate for initialization.
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
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // Call aclrtMemcpy to copy host data to device memory.
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
    // 1. (Boilerplate) Initialize the device/stream. For details, see the public ACL API list.
    // Set deviceId based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Handle the check as required.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs. Customize the construction based on the API.
    std::vector<int64_t> inputShape = {2, 5};
    std::vector<int64_t> targetShape = {2,};
    std::vector<int64_t> weightShape = {5,};
    std::vector<int64_t> lossOutShape = {1,};
    std::vector<int64_t> logProbOutShape = {2,5};
    std::vector<int64_t> zlossOutShape = {1,};
    std::vector<int64_t> lseForZlossOutShape = {2,};

    void* inputDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;

    void* lossOutDeviceAddr = nullptr;
    void* logProbOutDeviceAddr = nullptr;
    void* zlossDeviceAddr = nullptr;
    void* lseForZlossDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* lossOut = nullptr;
    aclTensor* logProbOut = nullptr;
    aclTensor* zloss = nullptr;
    aclTensor* lseForZloss = nullptr;

    // data
    std::vector<float> inputHostData = {5, 0, 3, 3, 7,
                                            9, 3, 5, 2, 4};
    std::vector<int64_t> targetHostData = {0, 0};
    std::vector<float> lossOutHostData = {1.0937543};
    std::vector<float> logProbOutHostData = {
        -2.159461, -7.159461, -4.159461, -4.159461, -0.159461,
        -0.0280476, -6.0280476, -4.0280476, -7.0280476, -5.0280476};
    std::vector<float> zlossOutHostData = {0};
    std::vector<float> lseForZlossOutHostData = {0, 0};

    // attr
    char* reduction = "mean";
    int64_t ignoreIndex = -100;
    float labelSmoothing = 0.0;
    float lseSquareScaleForZloss = 0.0;
    bool returnZloss = 0;

    // Create the input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the target aclTensor.
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create the lossOut aclTensor.
    ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the logProbOut aclTensor.
    ret = CreateAclTensor(logProbOutHostData, logProbOutShape, &logProbOutDeviceAddr, aclDataType::ACL_FLOAT, &logProbOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create the zloss aclTensor.
    ret = CreateAclTensor(zlossOutHostData, zlossOutShape, &zlossDeviceAddr, aclDataType::ACL_FLOAT, &zloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // lseForZloss aclTensor
    ret = CreateAclTensor(lseForZlossOutHostData, lseForZlossOutShape, &lseForZlossDeviceAddr, aclDataType::ACL_FLOAT, &lseForZloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API. Replace it with the specific API name.
    // Call the first-phase API of aclnnCrossEntropyLoss.
    ret = aclnnCrossEntropyLossGetWorkspaceSize(input, target, weight, reduction, ignoreIndex, labelSmoothing, lseSquareScaleForZloss, returnZloss, lossOut, logProbOut, zloss, lseForZloss, &workspaceSize, &executor);

    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropyLossGetWorkspaceSize failed. ERROR: %d\n",
                    ret);
        return ret);

    // Allocate device memory based on the workspaceSize calculated by the first-phase API.
    void *workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                return ret);
    }

    // Call the second-phase API of aclnnCrossEntropyLoss.
    ret = aclnnCrossEntropyLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnCrossEntropyLoss failed. ERROR: %d\n", ret);
                return ret);

    // 4. (Boilerplate) Synchronize and wait until the task is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ret);

    // 5. Obtain the output value and copy the result from device memory to host memory. Modify it based on the specific API definition.]

    auto size1 = GetShapeSize(lossOutShape);
    auto size2 = GetShapeSize(logProbOutShape);
    std::vector<float> resultData1(size1, 0);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), lossOutDeviceAddr,
                        size1 * sizeof(resultData1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: \n[");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("%f, ", resultData1[i]);
    }
    LOG_PRINT("]\n");

    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), logProbOutDeviceAddr,
                        size2 * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logProb result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("logprob is: \n [");
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("%f,", i, resultData2[i]);
    }
    LOG_PRINT("]\n");

    // 6. Release aclTensor and aclScalar. Modify it based on the specific API definition.
    aclDestroyTensor(input);
    aclDestroyTensor(target);
    aclDestroyTensor(lossOut);
    aclDestroyTensor(logProbOut);

    // 7. Release device resources.
    aclrtFree(inputDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(lossOutDeviceAddr);
    aclrtFree(logProbOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
