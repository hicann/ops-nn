# aclnnLSTM

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  Ascend 950PR/Ascend 950DT  |     ×    |
|  <term>Atlas A3 training products/Atlas A3 inference products</term>  |     √    |
|  <term>Atlas A2 training products/Atlas A2 inference products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference products</term>   |     √    |
|  <term>Atlas training products</term>   |     √    |

## Function

- Description: Implements the long short-term memory (LSTM) network, which is a special recurrent neural network (RNN) model. Computes the LSTM network, receives the input sequence and initial state, and returns the output sequence and final state.
- Formula:
  
  $$
  \begin{aligned}
  (1)\qquad f_t &=\sigma(W_f[h_{t-1}, x_t] + b_f) \\
  (2)\qquad     i_t &=\sigma(W_i[h_{t-1}, x_t] + b_i) \\
  (3)\qquad     o_t &=\sigma(W_o[h_{t-1}, x_t] + b_o) \\
  (4)\qquad     \tilde{c}_t &=tanh(W_c[h_{t-1}, x_t] + b_c) \\
  (5)\qquad     c_t &=f_t ⊙ c_{t-1} + i_t ⊙ \tilde{c}_t \\
  (6)\qquad     c_{o}^{t} &=tanh(c_t) \\
  (7)\qquad     h_t &=o_t ⊙ c_{o}^{t} \\
  \end{aligned}
  $$

  - $x_t ∈ R^{d}$: input vector to the LSTM unit.
  - $f_t ∈ (0, 1)^{h}$: activation vector of the forget gate.
  - $i_t ∈ (0, 1)^{h}$: activation vector of the input/update gate.
  - $o_t ∈ (0, 1)^{h}$: activation vector of the output gate.
  - $h_i ∈ (-1, 1)^{h}$: hidden state vector, also known as output vector of the LSTM unit.
  - $\tilde{c}_t ∈ (-1, 1)^{h}$: cell input activation vector.
  - $c_t ∈ R^{h}$: cell state vector.
  - $W ∈ R^{h×d}, (U ∈ R^{h×h})∩(b ∈ R^{h})$: weight matrices and bias vector parameters which need to be learned during training.

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, `aclnnLSTMGetWorkspaceSize` is called to obtain the input parameters and compute the required workspace size based on the process. Then, `aclnnLSTM` is called to perform computation.

```Cpp
aclnnStatus aclnnLSTMGetWorkspaceSize(
    const aclTensor     *input,
    const aclTensorList *params,
    const aclTensorList *hx,
    const aclTensor     *batchSizes,
    bool                 hasBias,
    int64_t              numLayers,
    double               dropout,
    bool                 train,
    bool                 bidirectional,        
    bool                 batchFirst,
    aclTensor           *output,
    aclTensor           *hy,
    aclTensor           *cy,
    aclTensorList       *iOut,  
    aclTensorList       *jOut, 
    aclTensorList       *fOut,
    aclTensorList       *oOut,
    aclTensorList       *hOut,
    aclTensorList       *cOut,
    aclTensorList       *tanhCOut,
    uint64_t            *workspaceSize,
    aclOpExecutor       **executor);
```

```Cpp
aclnnStatus aclnnLSTM(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnLSTMGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1570px"><colgroup>
  <col style="width: 134px">
  <col style="width: 121px">
  <col style="width: 263px">
  <col style="width: 469px">
  <col style="width: 169px">
  <col style="width: 128px">
  <col style="width: 142px">
  <col style="width: 144px">
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
      <td>input</td>
      <td>Input</td>
      <td>Input vector of the LSTM unit.</td>
      <td>
      <ul>
          <li><strong>If the batchSizes pointer is null:</strong>
            <br>The shape format is determined by the batchFirst parameter.
            <ul>
              <li>batchFirst=False: (time_step, batch_size, input_size)</li>
              <li>batchFirst=True: (batch_size, time_step, input_size)</li>
            </ul>
            Note: batchFirst indicates whether the batch dimension is in the first dimension. time_step indicates the time dimension. batch_size indicates the number of samples processed at each time step. input_size indicates the number of input features.
          </li>
          <li><strong>If valid batchSizes are passed:</strong>
            <br>Shape format: (time_step * batch_size, input_size)
            <br>Note: The memory layout is the same as that of (time_step, batch_size, input_size).
          </li>
      </ul>
      </td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td> params</td>
      <td>Input</td>
      <td>Weight and bias tensor list in the LSTM operation.</td>
  <td>
  <ul>
    <p>The formula for calculating the list length is: <strong>2 * D * B * num_layers</strong></p>
    <ul>
    <li>num_layers: number of LSTM layers, corresponding to the numLayers parameter.</li>
    <li>D: If bidirection is set to True, D is 2. Otherwise, D is 1.</li>
    <li>B: If has_biases is set to True, B is 2. Otherwise, B is 1.</li>
    </ul>
    
    <p><strong>Special scenario (bidirection=True and has_biases=True):</strong></p>
    <p style="padding-left: 20px;">
      Parameter layout: [weight_ih_0, weight_hh_0, bias_ih_0, bias_hh_0, weight_ih_reverse_0, weight_hh_reverse_0, bias_ih_reverse_0, bias_hh_reverse_0]
    </p>
    
    <p><strong>Core parameters (using layer 0 as an example):</strong></p>
    <ul>
    <li>weight_ih_0: input weight parameter of layer 0, shape = (4 * hidden_size, cur_input_size)
      <br>Note: cur_input_size indicates the number of input features at each layer. (The value of the first layer is input_size, that of the subsequent layers is hidden_size, and that of the bidirectional layers is 2 x hidden_size.)
    </li>
    <li>weight_hh_0: hidden layer weight parameter of layer 0, shape=(4 * hidden_size, hidden_size)</li>
    <li>bias_ih_0: input weight bias of layer 0, shape=(4 * hidden_size)</li>
    <li>bias_hh_0: hidden layer weight bias of layer 0, shape=(4 * hidden_size)</li>
    </ul>
    </ul>
  </td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hx</td>
      <td>Optional input</td>
      <td>Initial hidden and cell state list in the LSTM operation.</td>
      <td>The list contains two elements. Each element supports three dimensions (D * num_layers, batch_size, hidden_size). If the input is empty, the initial hidden and cell states are 0.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>batchSizes</td>
      <td>Optional input</td>
      <td>Number of valid batches that are actually involved in computation at each time step. If nullptr is passed, the input is in fixed-length mode. Otherwise, the input is in variable-length mode.</td>
      <td>The shape is (time_step,). The elements must be sorted in descending order. The element value is a positive integer and cannot exceed the total number of batches. The value of the first element must be equal to the total number of batches.</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hasBias</td>
      <td>Input</td>
      <td>Whether biases are available.</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>numLayers</td>
      <td>Input</td>
      <td>Number of LSTM layers.</td>
      <td>/</td>
      <td>INT64</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>dropout</td>
      <td>Input</td>
      <td>Probability of random masking.</td>
      <td>This function is not supported.</td>
      <td>DOUBLE</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>train</td>
      <td>Input</td>
      <td>Indicates whether the model is in training mode.</td>
      <td>If train is set to True, the intermediate result is saved during forward LSTM computation for backpropagation. If train is set to False, the intermediate result is not saved during forward computation.</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>bidirectional</td>
      <td>Input</td>
      <td>Whether it is bidirectional.</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>batchFirst</td>
      <td>Input</td>
      <td>Indicates whether the input data format is B, T, H (B, T, H) on the first axis.</td>
      <td>/</td>
      <td>BOOL</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
       <tr>
      <td>output</td>
      <td>Output</td>
      <td>Indicates the output of each time step in the last layer of the LSTM operation.</td>
      <td><ul><li>If batchSizes is passed as a null pointer:<br>When batchFirst is set to False, the shape supports three dimensions (time_step, batch_size, D * hidden_size). Otherwise, the shape supports three dimensions (batch_size, time_step, D * hidden_size). </li><li>If valid batchSizes is passed:<br>The shape must be (time_step, batch_size, D * hidden_size).</li></ul></td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>Output</td>
      <td>Indicates the hidden layer (output of formula 7) at the last time step of each layer during the LSTM operation.</td>
      <td>The shape supports three dimensions (D * num_layers, batch_size, hidden_size) </td>.
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Output</td>
      <td>Indicates the cell state (output of formula 5) at the last time step of each layer during the LSTM operation.</td>
      <td>The shape supports three dimensions (D * num_layers, batch_size, hidden_size) </td>.
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy</td>
      <td>Output</td>
      <td>Indicates the hidden layer (output of formula (7)) at the last time step of each layer during the LSTM operation.</td>
      <td>The shape supports three dimensions: (D * num_layers, batch_size, hidden_size) </td>.
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cy</td>
      <td>Output</td>
      <td>Indicates the cell state (output of formula (5)) at the last time step of each layer during the LSTM operation.</td>
      <td>The shape supports three dimensions: (D * num_layers, batch_size, hidden_size) </td>.
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>iOut</td>
      <td>Output</td>
      <td>Indicates the activation value of the input gate (sigmoid output, output of formula (2)) at each layer during the LSTM operation.</td>
      <td>The length of the list is D x num_layers. Each shape in the list supports three dimensions: (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>jOut</td>
      <td>Output</td>
      <td> Candidate cell state (tanh output, output of formula 4) of each layer in the LSTM operation.</td>
      <td> The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>fOut</td>
      <td>Output</td>
      <td> Activation value of the forget gate (sigmoid output, output of formula 1) of each layer in the LSTM operation.</td>
      <td> The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>oOut</td>
      <td>Output</td>
      <td> Activation value of the output gate (sigmoid output, output of formula 3) of each layer in the LSTM operation.</td>
      <td> The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
      <tr>
      <td>hOut</td>
      <td>Output</td>
      <td>Indicates the hidden layer (output of formula 7) at each layer during the LSTM operation.</td>
      <td>The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>cOut</td>
      <td>Output</td>
      <td>Indicates the final cell state (output of formula 5) at each layer during the LSTM operation.</td>
      <td>The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
      <td>√</td>
    </tr>
      <tr>
      <td>tanhCOut</td>
      <td>Output</td>
      <td>Indicates the output of the final cell state at each layer after the tanh activation function is performed during the LSTM operation (output of formula 6).</td>
      <td>The list length is D x num_layers. Each shape in the list supports three dimensions (time_step, batch_size, hidden_size). When train is set to False, there is no output value.</td>
      <td>FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>/</td>
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
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Returns**

    `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

    The first-phase API implements input parameter validation. The following error codes may be returned.
    <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
    <col style="width: 319px">
    <col style="width: 108px">
    <col style="width: 621px">
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
        <td>The required input, output, or attribute is passed as a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>The input parameter type is aclTensor and its data type is not supported.</td>
      </tr>
    </tbody>
    </table>

## aclnnLSTM

- **Parameters**

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
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first segment of the API aclnnLSTMGetWorkspaceSize.</td>
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

- **Returns**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:
  - The aclnnLSTM is implemented in deterministic mode by default.
- All inputs and outputs support the FLOAT16 and FLOAT32 types, and their data types must be the same.

## Example

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file main.cpp
 */

#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_lstm.h"

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

void PrintOutResult(const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. (Boilerplate) Initialize the device and stream. For details, see the list of external AscendCL APIs.
    // Set the device ID in use.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. Construct the inputs and outputs based on the API definition.
    int time_step = 1;
    int batch_size = 1;
    int hidden_size = 1;
    int input_size = hidden_size;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    int64_t d_scale = bidirection == true ? 2 : 1;

    std::vector<int64_t> inputShape = {time_step, batch_size, input_size};
    std::vector<int64_t> outputShape = {time_step, batch_size, d_scale * hidden_size};
    std::vector<int64_t> hycyShape = {numLayers * d_scale, batch_size, hidden_size};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    std::vector<std::vector<int64_t>> outIListShape = {};
    std::vector<std::vector<int64_t>> outJListShape = {};
    std::vector<std::vector<int64_t>> outFListShape = {};
    std::vector<std::vector<int64_t>> outOListShape = {};
    std::vector<std::vector<int64_t>> outHListShape = {};
    std::vector<std::vector<int64_t>> outCListShape = {};
    std::vector<std::vector<int64_t>> outTanhCListShape = {};

    auto cur_input_size = input_size;
    for (int i = 0; i < numLayers; i++) {
        paramsListShape.push_back({hidden_size * 4, cur_input_size});
        paramsListShape.push_back({hidden_size * 4, hidden_size});

        outIListShape.push_back({time_step, batch_size, hidden_size});
        outJListShape.push_back({time_step, batch_size, hidden_size});
        outFListShape.push_back({time_step, batch_size, hidden_size});
        outOListShape.push_back({time_step, batch_size, hidden_size});
        if (isTraining == true) {
            outHListShape.push_back({time_step, batch_size, hidden_size});
            outCListShape.push_back({time_step, batch_size, hidden_size});
        } else {
            outHListShape.push_back({batch_size, hidden_size});
            outCListShape.push_back({batch_size, hidden_size});
        }
        outTanhCListShape.push_back({time_step, batch_size, hidden_size});
        cur_input_size = hidden_size;
    }

    void* inputDeviceAddr = nullptr;
    void* paramsListDeviceAddr[2 * numLayers];

    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* cyDeviceAddr = nullptr;
    void* outIListDeviceAddr[numLayers];
    void* outJListDeviceAddr[numLayers];
    void* outFListDeviceAddr[numLayers];
    void* outOListDeviceAddr[numLayers];
    void* outHListDeviceAddr[numLayers];
    void* outCListDeviceAddr[numLayers];
    void* outTanhCListDeviceAddr[numLayers];

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;

    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* cy = nullptr;
    aclTensorList* outIList = nullptr;
    aclTensorList* outJList = nullptr;
    aclTensorList* outFList = nullptr;
    aclTensorList* outOList = nullptr;
    aclTensorList* outHList = nullptr;
    aclTensorList* outCList = nullptr;
    aclTensorList* outTanhCList = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1);

    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr, aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hycyHostData(GetShapeSize(hycyShape), 1);
    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor<float>(hycyHostData, hycyShape, &cyDeviceAddr, aclDataType::ACL_FLOAT, &cy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outIListShape, outIListDeviceAddr, aclDataType::ACL_FLOAT, &outIList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outJListShape, outJListDeviceAddr, aclDataType::ACL_FLOAT, &outJList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outFListShape, outFListDeviceAddr, aclDataType::ACL_FLOAT, &outFList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outOListShape, outOListDeviceAddr, aclDataType::ACL_FLOAT, &outOList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outHListShape, outHListDeviceAddr, aclDataType::ACL_FLOAT, &outHList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(outCListShape, outCListDeviceAddr, aclDataType::ACL_FLOAT, &outCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(
        outTanhCListShape, outTanhCListDeviceAddr, aclDataType::ACL_FLOAT, &outTanhCList, 0.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // Call the first segment of the aclnnLSTM API.
    ret = aclnnLSTMGetWorkspaceSize(
        input, params, nullptr, nullptr, isbias, numLayers, 0.0, isTraining, bidirection, batchFirst, output, hy, cy,
        outIList, outJList, outFList, outOList, outHList, outCList, outTanhCList, &workspaceSize, &executor);

    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTMGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // Call the second segment of the aclnnLSTM API.
    ret = aclnnLSTM(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLSTM failed. ERROR: %d\n", ret); return ret);

    // 4. (Boilerplate) Wait until the task execution is complete.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.

    PrintOutResult(outputShape, &outputDeviceAddr);

    // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    aclDestroyTensor(cy);
    aclDestroyTensorList(outIList);
    aclDestroyTensorList(outJList);
    aclDestroyTensorList(outFList);
    aclDestroyTensorList(outOList);
    aclDestroyTensorList(outHList);
    aclDestroyTensorList(outCList);
    aclDestroyTensorList(outTanhCList);

    // 7. Release device resources.
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    aclrtFree(cyDeviceAddr);
    for (int i = 0; i < numLayers; i++) {
        aclrtFree(outIListDeviceAddr[i]);
        aclrtFree(outJListDeviceAddr[i]);
        aclrtFree(outFListDeviceAddr[i]);
        aclrtFree(outOListDeviceAddr[i]);
        aclrtFree(outHListDeviceAddr[i]);
        aclrtFree(outCListDeviceAddr[i]);
        aclrtFree(outTanhCListDeviceAddr[i]);
    }

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
