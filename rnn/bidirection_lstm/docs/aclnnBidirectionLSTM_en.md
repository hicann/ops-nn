# aclnnBidirectionLSTM

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:48:03.797Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: The LSTM (Long Short-Term Memory) network is a special type of recurrent neural network (RNN) model. Performs LSTM network computation, receives an input sequence and initial states, and returns an output sequence and final states.
- Formula:

  $$
  f_t =sigm(W_f[h_{t-1}, x_t] + b_f)\\
  i_t =sigm(W_i[h_{t-1}, x_t] + b_i)\\
  o_t =sigm(W_o[h_{t-1}, x_t] + b_o)\\
  \tilde{c}_t =tanh(W_c[h_{t-1}, x_t] + b_c)\\
  c_t =f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t\\
  c_{o}^{t} =tanh(c_t)\\
  h_t =o_t ⊙ c_{o}^{t}\\
  $$

  - $x_t ∈ R^{d}$: Input vector of the LSTM unit.
  - $f_t ∈ (0, 1)^{h}$: forget gate activation vector.
  - $i_t ∈ (0, 1)^{h}$: input gate activation vector.
  - $o_t ∈ (0, 1)^{h}$: output gate activation vector.
  - $h_t ∈ (-1, 1)^{h}$: hidden status vector, also called the LSTM unit output vector.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: cell input activation vector.
  - $c_t ∈ R^{h}$: cell state vector.
  - $W ∈ R^{h×d}，(U ∈ R^{h×h})∩(b ∈ R^{h})$: weight matrices and bias vector parameters learned during training.

## Function Prototype

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call "aclnnBidirectionLSTMGetWorkspaceSize" to obtain the input parameters and calculate the required workspace size, and then call "aclnnBidirectionLSTM" to execute the computation.

```Cpp
aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *initH,
  const aclTensor *initC,
  const aclTensor *wIh,
  const aclTensor *wHh,
  const aclTensor *bIhOptional,
  const aclTensor *bHhOptional,
  const aclTensor *wIhReverseOptional,
  const aclTensor *wHhReverseOptional,
  const aclTensor *bIhReverseOptional,
  const aclTensor *bHhReverseOptional,
  int64_t          numLayers,
  bool             isbias,
  bool             batchFirst,
  bool             bidirection,
  const aclTensor *yOut,
  const aclTensor *outputHOut,
  const aclTensor *outputCOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnBidirectionLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnBidirectionLSTMGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1420px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 240px">
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
      <td>x (aclTensor*)</td>
      <td>Input</td>
      <td>Input vector of the LSTM unit, x in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape supports 3D (time_step, batch_size, input_size). Here, `time_step` indicates the time dimension; `batch_size` indicates the number of batches processed at each time step; `input_size` indicates the number of input features.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initH (aclTensor*)</td>
      <td>Input</td>
      <td>Initial hidden status, h in the formula.</td>
      <td><ul><li>Support empty tensor.</li><li>The shape supports 3D (num_layers, batch_size, hidden_size) or when bidirection is True (2 * num_layers, batch_size, hidden_size). `num_layers` corresponds to the parameter `numLayers` and indicates the number of LSTM layers; `hidden_size` indicates the number of features in the hidden state.</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>initC (aclTensor*)</td>
      <td>Input</td>
      <td>Initial cell state, c in the formula.</td>
      <td><ul><li>Support empty tensor.</li><li>The shape supports 3D (num_layers, batch_size, hidden_size) or when bidirection is True (2 * num_layers, batch_size, hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wIh (aclTensor*)</td>
      <td>Input</td>
      <td>input-hidden weight, which is W in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape supports 2D (4 * hidden_size, input_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHh (aclTensor*)</td>
      <td>Input</td>
      <td>hidden-hidden weight, W in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>shape supports 2D (4 * hidden_size, hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bIhOptional (aclTensor*)</td>
      <td>Input</td>
      <td>input-hidden bias, which is b in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>Supports a 1D shape (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhOptional(aclTensor*)</td>
      <td>Input</td>
      <td>hidden-hidden bias, which is b in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Supports 1D shape (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>wIhReverseOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Reverse input-hidden weight, which is W in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>Supports 2D shape (4 * hidden_size, input_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
      <tr>
      <td>wHhReverseOptional(aclTensor*)</td>
      <td>Input</td>
      <td>Reverse hidden-hidden weight, W in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>The shape supports 2D (4 * hidden_size, hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bIhReverseOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Reverse input-hidden bias, b in the formula.</td>
      <td><ul><li>Supports empty tensors.</li><li>shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>bHhReverseOptional (aclTensor*)</td>
      <td>Input</td>
      <td>Reverse hidden-hidden bias, b in the formula.</td>
      <td><ul><li>Supports empty tensor.</li><li>shape supports 1D (4 * hidden_size).</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers (int64_t)</td>
      <td>Input</td>
      <td>Indicates the number of LSTM layers.</td>
      <td>Currently, only 1 is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>isbias (bool)</td>
      <td>Input</td>
      <td>Indicates whether a bias exists.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>batchFirst (bool)</td>
      <td>Input</td>
      <td>Indicates whether batch is the first dimension.</td>
      <td>Currently, only false is supported.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>bidirection (bool)</td>
      <td>Input</td>
      <td>Indicates whether it is bidirectional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut (aclTensor*)</td>
      <td>Output</td>
      <td>LSTM unit output vector.</td>
      <td>The shape supports 3D (time_step, batch_size, hidden_size) or, when bidirection is True, (time_step, batch_size, 2 * hidden_size).</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputHOut (aclTensor*)</td>
      <td>Output</td>
      <td>Final hidden status, h in the formula.</td>
      <td>The shape supports 3D (num_layers, batch_size, hidden_size) or when bidirection is True, (2 * num_layers, batch_size, hidden_size).</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>outputCOut (aclTensor*)</td>
      <td>Output</td>
      <td>Final cell status, c in the formula.</td>
      <td>The shape supports 3D (num_layers, batch_size, hidden_size) or when bidirection is True, (2 * num_layers, batch_size, hidden_size).</td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>Output</td>
      <td>Returns the size of the workspace that requires Device side application.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor(aclOpExecutor**)</td>
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
    <td>If the passed-in parameter is a required input, output, or required attribute, and is a null pointer.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>If the passed-in parameter is of the aclTensor type and its data type is not within the supported range.</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_INNER_TILING_ERROR</td>
    <td>561002</td>
    <td>If the passed-in parameter is of type aclTensor and its shape does not match the preceding parameter description.</td>
  </tr>
</tbody>
</table>

## aclnnBidirectionLSTM

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
      <td>The workspace memory address applied on the Device side.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>The workspace size applied on the Device side, obtained by the first-phase API aclnnBidirectionLSTMGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream used to execute the task.</td>
    </tr>
  </tbody>
  </table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - aclnnBidirectionLSTM defaults to a deterministic implementation.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_bidirection_lstm.h"

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
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization
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

  // Compute the strides of the contiguous tensor.
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
  // 1. (Boilerplate) Initialize device/stream. For details, see the ACL API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the inputs and outputs. Custom construction is required based on the API.
  int time_step = 2;
  int batch_size = 32;
  int input_size = 32;
  int hidden_size = 32;

  int64_t numLayers = 1;
  bool isbias = true;
  bool batchFirst = false;
  bool bidirection = true;

  std::vector<int64_t> selfShape = {time_step, batch_size, input_size};
  std::vector<int64_t> weightHIShape = {4 * hidden_size, input_size};
  std::vector<int64_t> weightHHShape = {4 * hidden_size, hidden_size};
  std::vector<int64_t> initHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> initCShape = {2, batch_size, hidden_size};
  std::vector<int64_t> biasHIShape = {4 * hidden_size};
  std::vector<int64_t> biasHHShape = {4 * hidden_size};
  std::vector<int64_t> outShape = {time_step, batch_size, 2 * hidden_size};
  std::vector<int64_t> outHShape = {2, batch_size, hidden_size};
  std::vector<int64_t> outCShape = {2, batch_size, hidden_size};

  void* selfDeviceAddr = nullptr;
  void* weightHIDeviceAddr = nullptr;
  void* weightHHDeviceAddr = nullptr;
  void* weightHIReverseDeviceAddr = nullptr;
  void* weightHHReverseDeviceAddr = nullptr;
  void* initHDeviceAddr = nullptr;
  void* initCDeviceAddr = nullptr;
  void* biasHIDeviceAddr = nullptr;
  void* biasHHDeviceAddr = nullptr;
  void* biasHIReverseDeviceAddr = nullptr;
  void* biasHHReverseDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* outHDeviceAddr = nullptr;
  void* outCDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weightHI = nullptr;
  aclTensor* weightHH = nullptr;
  aclTensor* weightHIReverse = nullptr;
  aclTensor* weightHHReverse = nullptr;
  aclTensor* biasHI = nullptr;
  aclTensor* biasHH = nullptr;
  aclTensor* biasHIReverse = nullptr;
  aclTensor* biasHHReverse = nullptr;
  aclTensor* initH = nullptr;
  aclTensor* initC = nullptr;
  aclTensor* out = nullptr;
  aclTensor* outH = nullptr;
  aclTensor* outC = nullptr;

  std::vector<uint16_t> selfHostData(GetShapeSize(selfShape));
  std::vector<uint16_t> weightHIHostData(GetShapeSize(weightHIShape));
  std::vector<uint16_t> weightHHHostData(GetShapeSize(weightHHShape));
  std::vector<uint16_t> biasHIHostData(GetShapeSize(biasHIShape));
  std::vector<uint16_t> biasHHHostData(GetShapeSize(biasHHShape));
  std::vector<uint16_t> initHHostData(GetShapeSize(initHShape));
  std::vector<uint16_t> initCHostData(GetShapeSize(initCShape));
  std::vector<uint16_t> outHostData(GetShapeSize(outShape));
  std::vector<uint16_t> outHHostData(GetShapeSize(outHShape));
  std::vector<uint16_t> outCHostData(GetShapeSize(outCShape));

  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIDeviceAddr, aclDataType::ACL_FLOAT16, &weightHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHDeviceAddr, aclDataType::ACL_FLOAT16, &weightHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(initHHostData, initHShape, &initHDeviceAddr, aclDataType::ACL_FLOAT16, &initH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(initCHostData, initCShape, &initCDeviceAddr, aclDataType::ACL_FLOAT16, &initC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIDeviceAddr, aclDataType::ACL_FLOAT16, &biasHI);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHDeviceAddr, aclDataType::ACL_FLOAT16, &biasHH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHIHostData, weightHIShape, &weightHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHHHostData, weightHHShape, &weightHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &weightHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHIHostData, biasHIShape, &biasHIReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHIReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHHHostData, biasHHShape, &biasHHReverseDeviceAddr, aclDataType::ACL_FLOAT16, &biasHHReverse);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHHostData, outHShape, &outHDeviceAddr, aclDataType::ACL_FLOAT16, &outH);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outCHostData, outCShape, &outCDeviceAddr, aclDataType::ACL_FLOAT16, &outC);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the specific API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // Call the first-phase API of aclnnBidirectionLSTM.
  ret = aclnnBidirectionLSTMGetWorkspaceSize(self, initH, initC, weightHI, weightHH,
                                            biasHI, biasHH, weightHIReverse, weightHHReverse, biasHIReverse, biasHHReverse,
                                            numLayers, isbias, batchFirst, bidirection,
                                            out, outH, outC,
                                            &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // Call the second-phase API of aclnnBidirectionLSTM.
  ret = aclnnBidirectionLSTM(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBidirectionLSTM failed. ERROR: %d\n", ret); return ret);

  // 4. (boilerplate) Synchronously wait for the task execution to end.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output values and copy the result in device-side memory to the host side. Modify the code based on the specific API definition.
  PrintOutResult(outShape, &outDeviceAddr);
  PrintOutResult(outHShape, &outHDeviceAddr);
  PrintOutResult(outCShape, &outCDeviceAddr);

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(weightHI);
  aclDestroyTensor(weightHH);
  aclDestroyTensor(initH);
  aclDestroyTensor(initC);
  aclDestroyTensor(biasHI);
  aclDestroyTensor(biasHH);
  aclDestroyTensor(weightHIReverse);
  aclDestroyTensor(weightHHReverse);
  aclDestroyTensor(biasHIReverse);
  aclDestroyTensor(biasHHReverse);
  aclDestroyTensor(out);
  aclDestroyTensor(outH);
  aclDestroyTensor(outC);

  // 7. Release device resources.
  aclrtFree(selfDeviceAddr);
  aclrtFree(weightHIDeviceAddr);
  aclrtFree(weightHHDeviceAddr);
  aclrtFree(initHDeviceAddr);
  aclrtFree(initCDeviceAddr);
  aclrtFree(biasHIDeviceAddr);
  aclrtFree(biasHHDeviceAddr);
  aclrtFree(weightHIReverseDeviceAddr);
  aclrtFree(weightHHReverseDeviceAddr);
  aclrtFree(biasHIReverseDeviceAddr);
  aclrtFree(biasHHReverseDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(outHDeviceAddr);
  aclrtFree(outCDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
