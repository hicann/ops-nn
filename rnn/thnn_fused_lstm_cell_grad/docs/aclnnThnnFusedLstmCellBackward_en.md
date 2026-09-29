# aclnnThnnFusedLstmCellBackward

## Supported Products

| Product                                                                           | Supported|
| :------------------------------------------------------------------------------ | :------: |
| Ascend 950PR/Ascend 950DT                                               |    ×     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>                         |    √     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>   |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                                         |    ×     |
| <term>Atlas inference products</term>                                                |    ×     |
| <term>Atlas training products</term>                                                 |    ×     |

## Function

- Operator function: performs backpropagation of the remaining computation after matmul in the four gates of LSTMCell, and calculates the gradients of the input cx and bias b and the values of the four gates before activation (gates) in the forward output.
- Formula:

**Variable definition**

* **Input gradients**: $\delta h_t$ (`gradHy`) and $\delta c_t$ (`gradC`)
* **Forward cache**: $i, f, g, o$ (activation values of each gate `storage`), $c_{t-1}$ (`cx`), and $c_t$ (`cy`)
* **Output gradients**: $\delta a_i, \delta a_f, \delta a_g, \delta a_o$ (stored in `gradGatesOut`) and $\delta c_{t-1}$ (`gradCxOut`)

**Phase 1: Intermediate gradient and status backpropagation**

First, the contribution of the hidden state to the cell state is calculated, and the total gradient of the cell state at the current time point is obtained as follows: $\text{grad\_}c_{total}$.

$$
\begin{aligned}
gcx &= \tanh(c_t) \\
\text{grad\_}c_{total} &= \delta h_t \cdot o \cdot (1 - gcx^2) + \delta c_t \\
\delta c_{t-1} &= \text{grad\_}c_{total} \cdot f
\end{aligned}
$$

**Phase 2: Gating component gradient (pre-activation)**

According to the code logic, the gradient $\delta a$ of each gate before entering the activation function is calculated as follows:

$$
\begin{aligned}
\delta a_o &= (\delta h_t \cdot gcx) \cdot o \cdot (1 - o) \\
\delta a_i &= (\text{grad\_}c_{total} \cdot g) \cdot i \cdot (1 - i) \\
\delta a_f &= (\text{grad\_}c_{total} \cdot c_{t-1}) \cdot f \cdot (1 - f) \\
\delta a_g &= (\text{grad\_}c_{total} \cdot i) \cdot (1 - g^2)
\end{aligned}
$$

**Phase 3: Parameter gradient (db)**

**1. Bias gradient (db):** Sum up the values in the batch dimension ($N$).

$$
\delta b = \sum_{n=1}^{N} \begin{bmatrix} \delta a_i \\ \delta a_f \\ \delta a_g \\ \delta a_o \end{bmatrix}_n
$$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnThnnFusedLstmCellBackwardGetWorkspaceSize` is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, `aclnnThnnFusedLstmCellBackward` is called to perform computation.

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(
  const aclTensor     *gradHyOptional,
  const aclTensor     *gradCOptional,
  const aclTensor     *cx,
  const aclTensor     *cy,
  const aclTensor     *storage,
  bool                hasBias,
  aclTensor           *gradGatesOut,
  aclTensor           *gradCxOut,
  aclTensor           *gradBiasOut,
  uint64_t            *workspaceSize,
  aclOpExecutor       **executor)
```

```Cpp
aclnnStatus aclnnThnnFusedLstmCellBackward(
  void            *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream     stream)
```

## aclnnThnnFusedLstmCellBackwardGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>gradHyOptional</td>
      <td>Optional input</td>
      <td>Gradient of the hidden state output by the LSTMCell in the forward direction.</td>
      <td>-</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCOptional</td>
      <td>Optional input</td>
      <td>Gradient of the cell state output by the LSTMCell in the forward direction.</td>
      <td>The data type is the same as that of gradHy.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cx</td>
      <td>Input</td>
      <td>Cell state input by the LSTMCell in the forward direction.</td>
      <td>The data type is the same as that of gradHy.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Input</td>
      <td>Forward output cell state of the LSTMCell.</td>
      <td>The data type is the same as that of gradHy.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storage</td>
      <td>Input</td>
      <td>Activation values of the four gates in the forward output of the LSTMCell.</td>
      <td>The data type must be the same as that of `input`.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, 4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias</td>
      <td>Input</td>
      <td>Whether to calculate the bias gradient.</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradGatesOut</td>
      <td>Output</td>
      <td>Gradient of the pre-activation values of the four gates in the forward output of the LSTMCell.</td>
      <td>The data type must be the same as that of `input`.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, 4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradCxOut</td>
      <td>Output</td>
      <td>Gradient of the cell state input to the LSTMCell forward pass.</td>
      <td>The data type must be the same as that of `input`.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[batch, hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut</td>
      <td>Output</td>
      <td>Gradient of the input bias in the LSTM forward pass.</td>
      <td>The data type must be the same as that of `input`.</td>
      <td>FLOAT32, FLOAT16</td>
      <td>ND</td>
      <td>[4 * hidden_size]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>Output parameter on the host.</td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>Output parameter on the host.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API implements input parameter validation. The following error codes may be returned.
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>Return Code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>If the input parameter is aclTensor and is not an optional input, the pointer is null.</td>
    </tr>
    <tr>
      <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="12">161002</td>
      <td>If the input parameter is aclTensor, the data type is not supported.</td>
    </tr>
    <tr>
      <td>If the input parameter is of type aclTensor, the data types are different.</td>
    </tr>
    <tr>
      <td>If the input parameter is of type aclTensor, the shape does not meet the corresponding requirements.</td>
    </tr>
  </tbody>
  </table>

## aclnnThnnFusedLstmCellBackward

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
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the aclnnThnnFusedLstmCellBackwardGetWorkspaceSize API.</td>
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
  
  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Restrictions

- Deterministic computation:
  - The aclnnThnnFusedLstmCellBackward is implemented in deterministic mode by default.
- Boundary value scenarios:
  - If the input is Inf, the output is NAN.
  - When the input is `NaN`, the output is `NaN`.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_thnn_fused_lstm_cell_backward.h"

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

int Init(int32_t deviceId, aclrtStream* stream) {
  // (Fixed writing) Initialize AscendCL.
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
  // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs based on the API definition.
  // Define variables.
  int64_t n = 1;
  int64_t hiddenSize = 8;

  // Define the shape.
  std::vector<int64_t> bShape = {hiddenSize * 4};
  std::vector<int64_t> dhShape = {n, hiddenSize};
  std::vector<int64_t> gatesShape = {n, 4 * hiddenSize};

  // Pointer to the device address
  void* dhyDeviceAddr = nullptr;
  void* dcDeviceAddr = nullptr;
  void* cxDeviceAddr = nullptr;
  void* cyDeviceAddr = nullptr;
  void* storageDeviceAddr = nullptr;

  // Pointer to the device address of the backpropagation output
  void* dgatesDeviceAddr = nullptr;
  void* dcPrevDeviceAddr = nullptr;
  void* dbDeviceAddr = nullptr;

  // Pointer to the ACL tensor.
  aclTensor* dhy = nullptr;
  aclTensor* dc = nullptr;
  aclTensor* cx = nullptr;
  aclTensor* cy = nullptr;
  aclTensor* storage = nullptr;

  // Output ACL tensor pointer for backpropagation
  aclTensor* dgates = nullptr;
  aclTensor* dcPrev = nullptr;
  aclTensor* db = nullptr;

  std::vector<float> dhyHostData(n * hiddenSize, 1.0f); // 1*1*8 = 8 ones
  std::vector<float> dcHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512 ones
  std::vector<float> cxHostData(n * hiddenSize, 1.0f); // (8+8)*32 = 16*32 = 512 ones
  std::vector<float> cyHostData(n * hiddenSize, 1.0f); // 32 ones
  std::vector<float> storageHostData(n * hiddenSize * 4, 1.0f); // 32 ones

  // Output host data for backpropagation (initialized to 0)
  std::vector<float> dgatesHostData(n * hiddenSize * 4, 0.0f);
  std::vector<float> dcPrevHostData(n * hiddenSize, 0.0f);
  std::vector<float> dbHostData(hiddenSize * 4, 0.0f);

  // Create the dhy aclTensor.
  ret = CreateAclTensor(dhyHostData, dhShape, &dhyDeviceAddr, aclDataType::ACL_FLOAT, &dhy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the params aclTensorList.
  ret = CreateAclTensor(dcHostData, dhShape, &dcDeviceAddr, aclDataType::ACL_FLOAT, &dc);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cxHostData, dhShape, &cxDeviceAddr, aclDataType::ACL_FLOAT, &cx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(cyHostData, dhShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(storageHostData, gatesShape, &storageDeviceAddr, aclDataType::ACL_FLOAT, &storage);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the backpropagation output tensor.
  // Create the dgates aclTensor.
  ret = CreateAclTensor(dgatesHostData, gatesShape, &dgatesDeviceAddr, aclDataType::ACL_FLOAT, &dgates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create the dcPrev aclTensor.
  ret = CreateAclTensor(dcPrevHostData, dhShape, &dcPrevDeviceAddr, aclDataType::ACL_FLOAT, &dcPrev);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create a db aclTensor.
  ret = CreateAclTensor(dbHostData, bShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // Call the first part of the aclnnThnnFusedLstmCellBackward API.
  ret = aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(dhy, dc, cx, cy, storage, true, dgates, dcPrev, db,
    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // Call the second part of the aclnnThnnFusedLstmCellBackward API.
  ret = aclnnThnnFusedLstmCellBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnThnnFusedLstmCellBackward failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.
  // Print the dparams result.
  auto dgatesSize = GetShapeSize(gatesShape);
  std::vector<float> resultDgatesData(dgatesSize, 0);
  ret = aclrtMemcpy(resultDgatesData.data(), resultDgatesData.size() * sizeof(resultDgatesData[0]), dgatesDeviceAddr,
                    dgatesSize * sizeof(resultDgatesData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dgates result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dgatesSize; i++) {
    LOG_PRINT("result dgates[%ld] is: %f\n", i, resultDgatesData[i]);
  }

  auto dbSize = GetShapeSize(bShape);
  std::vector<float> resultDwhData(dbSize, 0);
  ret = aclrtMemcpy(resultDwhData.data(), resultDwhData.size() * sizeof(resultDwhData[0]), dbDeviceAddr,
                    dbSize * sizeof(resultDwhData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy db result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dbSize; i++) {
    LOG_PRINT("result db[%ld] is: %f\n", i, resultDwhData[i]);
  }

  auto dcPrevSize = GetShapeSize(dhShape);
  std::vector<float> resultDcPrevData(dcPrevSize, 0);
  ret = aclrtMemcpy(resultDcPrevData.data(), resultDcPrevData.size() * sizeof(resultDcPrevData[0]), dcPrevDeviceAddr,
                    dcPrevSize * sizeof(resultDcPrevData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy dcPrev result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < dcPrevSize; i++) {
    LOG_PRINT("result dcPrev[%ld] is: %f\n", i, resultDcPrevData[i]);
  }
  // Destroy aclTensor.
  aclDestroyTensor(dhy);
  aclDestroyTensor(dc);
  aclDestroyTensor(cx);
  aclDestroyTensor(cy);
  aclDestroyTensor(storage);
  aclDestroyTensor(dgates);
  aclDestroyTensor(dcPrev);
  aclDestroyTensor(db);

  //Destroy the device resources.
  aclrtFree(dhyDeviceAddr);
  aclrtFree(dcDeviceAddr);
  aclrtFree(cxDeviceAddr);
  aclrtFree(cyDeviceAddr);
  aclrtFree(storageDeviceAddr);
  aclrtFree(dgatesDeviceAddr);
  aclrtFree(dcPrevDeviceAddr);
  aclrtFree(dbDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
