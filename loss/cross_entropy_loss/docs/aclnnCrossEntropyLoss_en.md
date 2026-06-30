# aclnnCrossEntropyLoss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/cross_entropy_loss)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    ×     |
| <term>Atlas training series products</term>                             |    ×     |

## Function

- Description: Computes the cross-entropy loss of the input.
- Formula:
  
  When **reductionOptional** is **mean**, the cross-entropy loss is calculated as:

  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$

  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reductionOptional = 'mean'} \\\sum_{n=1}^Nl_n,&\text {if reductionOptional = 'sum' }\\\{l_0,l_1,...,l_n\},&\text{if reductionOptional = 'None' }\end{cases}
  $$

  **log\_prob** is calculated as:

  $$
  lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
  $$

  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$

  **zloss** is calculated as:

  $$
  zloss_n = lseSquareScaleForZloss × (lse_n)^2
  $$

  Where **N** is the batch size and **C** is the number of labels.
  
## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnCrossEntropyLossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnCrossEntropyLoss** is called to perform computation.

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
    <td>input</td>
    <td>Input</td>
    <td>input in the formula.</td>
    <td>-</td>
    <td>FLOAT, FLOAT16, BFLOAT16</td>
    <td>ND</td>
    <td>(N,C)<br>N indicates the batch size, and C indicates the number of labels. The value must be greater than 0.</td>
    <td>-</td>
    </tr>
    <tr>
    <td>target</td>
    <td>Input</td>
    <td>Label, corresponding to y in the formula.</td>
    <td>-</td>
    <td>INT32, INT64</td>
    <td>ND</td>
    <td>(N)<br>N equals the zeroth dimension of input, and the label values are in the range [0, C).</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightOptional</td>
    <td>Input</td>
    <td>Weight specified for each class, corresponding to weight in the formula.</td>
    <td>If the weight is not specified, target is unweighted.</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>(C)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>reductionOptional</td>
    <td>Input</td>
    <td>Reduction mode for the loss.</td>
    <td>"mean", "sum", and "none" are supported.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>ignoreIndex</td>
    <td>Input</td>
    <td>Index of the label to be ignored.</td>
    <td>The value must be less than C. Values less than 0 indicate no label is ignored.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>labelSmoothing</td>
    <td>Input</td>
    <td>Smoothing factor for loss computation.</td>
    <td>The value must be in the range [0.0, 1.0].</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseSquareScaleForZloss</td>
    <td>Input</td>
    <td>Scale factor for zloss computation, corresponding to lse_square_scale_for_zloss in the formula.</td>
    <td>The value must be in the range [0, 1). This parameter is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnZloss</td>
    <td>Input</td>
    <td>Whether to return the zloss output.</td>
    <td>Pass True to output zLoss; otherwise, pass False. This parameter is not supported currently.</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lossOut</td>
    <td>Output</td>
    <td>Output loss.</td>
    <td>If reductionOptional is "None", the shape is [N], consistent with the zeroth dimension of input; otherwise, the shape is [1].</td>
    <td>Same as input.</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>logProbOut</td>
    <td>Output</td>
    <td>Output passed to the backward computation.</td>
    <td>-</td>
    <td>Same as input.</td>
    <td>ND</td>
    <td>(N,C)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zlossOut</td>
    <td>Output</td>
    <td>Auxiliary loss.</td>
    <td>This parameter is not supported currently.</td>
    <td>Same as input.</td>
    <td>ND</td>
    <td>Same as lossOut.</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseForZlossOut</td>
    <td>Output</td>
    <td>Tensor output used for backward computation in zloss scenarios. The output is None if lseSquareScaleForZloss is 0.</td>
    <td>This parameter is not supported currently.</td>
    <td>Same as input.</td>
    <td>ND</td>
    <td>(N)</td>
    <td>-</td>
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
    </tbody>
    </table>

- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1152px"><colgroup>
  <col style="width: 287px">
  <col style="width: 125px">
  <col style="width: 740px">
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
      <td>The passed input, target, lossOut, logProbOut, zlossOut, or lseForZlossOut is a null pointer.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>The data type of input, target, lossOut, logProbOut, zlossOut, or lseForZlossOut is not supported.</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="2">561002</td>
      <td>The shape of input, target, or weightOptional is not supported.</td>
    </tr>
    <tr>
      <td>The value of reductionOptional or labelSmoothing is not supported.</td>
    </tr>
  </tbody>
  </table>

## aclnnCrossEntropyLoss

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1148px"><colgroup>
  <col style="width: 170px">
  <col style="width: 123px">
  <col style="width: 855px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnCrossEntropyLossGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation flow.</td>
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

  - **target** only supports class label indices; probabilistic inputs are not supported.
  - The **zloss**-related features are currently not supported. Parameters **lseSquareScaleForZloss** and **returnZloss** are ignored even if provided.
  - Deterministic computation:
    - **aclnnCrossEntropyLoss** defaults to a deterministic implementation.
## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

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
  // (Boilerplate) Perform initialization.
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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
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

    // Create an input aclTensor.
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a target aclTensor.
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // Create a lossOut aclTensor.
    ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a logProbOut aclTensor.
    ret = CreateAclTensor(logProbOutHostData, logProbOutShape, &logProbOutDeviceAddr, aclDataType::ACL_FLOAT, &logProbOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a zloss aclTensor.
    ret = CreateAclTensor(zlossOutHostData, zlossOutShape, &zlossDeviceAddr, aclDataType::ACL_FLOAT, &zloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // lseForZloss aclTensor
    ret = CreateAclTensor(lseForZlossOutHostData, lseForZlossOutShape, &lseForZlossDeviceAddr, aclDataType::ACL_FLOAT, &lseForZloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. Call the CANN operator library API. Modify the API as required.
    // Call the first-phase API of aclnnCrossEntropyLoss.
    ret = aclnnCrossEntropyLossGetWorkspaceSize(input, target, weight, reduction, ignoreIndex, labelSmoothing, lseSquareScaleForZloss, returnZloss, lossOut, logProbOut, zloss, lseForZloss, &workspaceSize, &executor);

    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropyLossGetWorkspaceSize failed. ERROR: %d\n",
                    ret);
        return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
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

    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ret);

    // 5. Obtain the output value and copy the result from the device to the host. Modify the code based on the API definition.

    auto size1 = GetShapeSize(lossOutShape);
    auto size2 = GetShapeSize(logProbOutShape);
    std::vector<float> resultData1(size1, 0);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), lossOutDeviceAddr,
                        size1 * sizeof(resultData1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: \n[");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("%f, ", i, resultData1[i]);
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

    // 6. Destroy aclTensor and aclScalar. Modify the code based on the API definition.
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
