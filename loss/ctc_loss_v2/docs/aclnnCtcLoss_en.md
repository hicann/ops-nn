# aclnnCtcLoss

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/loss/ctc_loss_v2)

## Product Support

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √     |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference series products</term>                            |    √     |
| <term>Atlas training series products</term>                             |    √     |

## Function

- Description: Computes the connectionist temporal classification (CTC) loss.

- Formula:
  Let $y_{k}^{t}$ denote the probability that the true character is $k$ at time step $t$ (typically, $y_{k}^{t}$ is an element of the output matrix after the softmax operation). Let $L^{'T}$ denote the set of all sequences that can be formed from the character set $L^{'}$. Each sequence in $L^{'T}$ is called a path and denoted by $π$. The distribution of $π$ is given by Equation (1):

  $$
  p(π|x)=\prod_{t=1}^{T}y^{t}_{π_{t}} , \forall π \in L'^{T}. \tag{1}
  $$

  Define a many-to-one mapping B: $L^{'T} \to L^{\leq T}$. The conditional probability of $l \in L^{\leq T}$ is the sum of the probabilities of all paths corresponding to $l$ under mapping B, as shown in Equation (2):

  $$
  p(l|x)=\sum_{π \in B^{-1}(l)}p(π|x).\tag{2}
  $$

  The task of finding the label $l$ that maximizes $p(l|x)$ is referred to as decoding, as expressed in Equation (3):

  $$
  h(x)=^{arg \  max}_{l \in L^{ \leq T}} \ p(l|x).\tag{3}
  $$

  When **zeroInfinity** is **True**:
  
  $$
  h(x)=\begin{cases}0,&h(x) == Inf \text{ or } h(x) == -Inf \\h(x),&\text { else }\end{cases}
  $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnCtcLossGetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation flow. Then, **aclnnCtcLoss** is called to perform computation.

```Cpp
aclnnStatus aclnnCtcLossGetWorkspaceSize(
 const aclTensor*     logProbs,
 const aclTensor*     targets,
 const aclIntArray*   inputLengths,
 const aclIntArray*   targetLengths,
 int64_t blank, bool  zeroInfinity,
 aclTensor*           negLogLikelihoodOut,
 aclTensor*           logAlphaOut,
 uint64_t*            workspaceSize,
 aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnCtcLoss(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnCtcLossGetWorkspaceSize

- **Parameters**

    <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
    <col style="width: 177px">
    <col style="width: 120px">
    <col style="width: 273px">
    <col style="width: 292px">
    <col style="width: 152px">
    <col style="width: 110px">
    <col style="width: 151px">
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
        <td>logProbs</td>
        <td>Input</td>
        <td>Log probabilities of the output, corresponding to y in the formula.</td>
        <td>-</td>
        <td>FLOAT16, FLOAT, BFLOAT16</td>
        <td>ND</td>
        <td>(T, N, C) or (T, C)<br>T: input length, N: batch size, C: number of classes (must be greater than 0, including the blank label).</td>
        <td>√</td>
      </tr>
      <tr>
        <td>targets</td>
        <td>Input</td>
        <td>Target sequence labels, corresponding to π in the formula.</td>
        <td>If the shape is (N, S), S must be greater than or equal to the maximum value in targetLengths. If the shape is (SUM(targetLengths)), targets must be unpadded and concatenated in 1D. When logProbs is 2D, N equals 1.</td>
        <td>INT64, INT32, BOOL, FLOAT, FLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>inputLengths</td>
        <td>Input</td>
        <td>Actual lengths of the input sequences. T in the formula is an element in inputLengths.</td>
        <td>The array length is N. Each value in the array must be less than or equal to T. When logProbs is 2D, N equals 1.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>targetLengths</td>
        <td>Input</td>
        <td>Actual lengths of the target sequences. l in the formula is an element in targetLengths.</td>
        <td>The array length is N. When the shape of targets is (N, S), each value in the array must be less than or equal to S. When logProbs is 2D, N equals 1.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>blank</td>
        <td>Input</td>
        <td>Blank label index.</td>
        <td>The value must be in the range [0, C).</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>zeroInfinity</td>
        <td>Input</td>
        <td>Whether to zero out infinite losses and associated gradients, corresponding to zeroInfinity in the formula.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>negLogLikelihoodOut</td>
        <td>Output</td>
        <td>Output loss value, corresponding to h in the formula.</td>
        <td>The data type must be the same as that of logProbs. If logProbs is 3D, negLogLikelihoodOut is a tensor with shape (N). Otherwise, negLogLikelihoodOut is a 0D tensor.</td>
        <td>Same as logProbs.</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>logAlphaOut</td>
        <td>Output</td>
        <td>Log probability of possible input-to-target alignments, corresponding to p(l|x) in the formula.</td>
        <td>The data type must be the same as that of logProbs. When logProbs is 2D, N equals 1.</td>
        <td>Same as logProbs.</td>
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

  - For <term>Atlas inference series products</term> and <term>Atlas training series products</term>, the data type cannot be FLOAT16 or BFLOAT16.
  - **logAlphaOut**:
     - For <term>Atlas inference series products</term>, <term>Atlas training series products</term>, <term>Atlas A2 training series products/Atlas A2 inference series products</term>, and <term>Atlas A3 training series products/Atlas A3 inference series products</term>, the shape is ($N, T, (2*max(targetLengths)+8)/8*8$).
     
- **Returns**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown.

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>The passed logProbs, targets, inputLengths, targetLengths, negLogLikelihoodOut, or logAlphaOut is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>The data type of logProbs, targets, inputLengths, or targetLengths is not supported.</td>
      </tr>
      <tr>
      <td>The tensor shape of logProbs, targets, inputLengths, targetLengths, negLogLikelihoodOut, or logAlphaOut does not meet the corresponding requirements, or the length of the inputLengths or targetLengths array does not meet the requirements.</td>
      </tr>
      <tr>
      <td>blank is out of the valid value range.</td>
      </tr>
    </tbody>
    </table>

## aclnnCtcLoss

- **Parameters**
   <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>Size of the workspace to be allocated on the device, which is obtained by calling the first-phase API aclnnCtcLossGetWorkspaceSize.</td>
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

- **Value range constraints**
  - The values in `targets` must be in the range $[0, C – 1]$ and must not include the value corresponding to **blank**, where $C$ is the last dimension of `logProbs`, representing the number of classes.
  - The values in `input_lengths` must be in the range $[1, T]$, where $T$ is the zeroth dimension of `logProbs`, representing the input length.
  - The values in `target_lengths` must be greater than or equal to 1.
  - Each element in `target_lengths` must be less than or equal to the corresponding elements in `input_lengths`.

  If the first three constraints are violated, out-of-bounds behavior may occur on CPU/GPU, which may cause the computed results of **negLogLikelihoodOut** and **logAlphaOut** to differ from the CPU/GPU reference. If the fourth constraint is violated, the computed result of **logAlphaOut** for the corresponding batch may differ from the CPU/GPU reference.
  
- Deterministic computation:
  - **aclnnCtcLoss** defaults to a non-deterministic implementation. You can call **aclrtCtxSetSysParamOpt** to enable deterministic computation.

## Example

- For the <term>Atlas inference series products</term>, <term>Atlas training series products</term>, <term>Atlas A2 training series products/Atlas A2 inference series products</term>, and <term>Atlas A3 training series products/Atlas A3 inference series products</term>, the sample code is as follows, which is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_ctc_loss.h"

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
    // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
    // Set the device ID (deviceId) based on the actual device.
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // Customize error handling based on your requirements.
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. Construct inputs and outputs based on API definitions.
    std::vector<int64_t> logProbsShape = {12, 4, 5};
    std::vector<int64_t> targetsShape = {4, 7};
    std::vector<int64_t> negLoglikelihoodOutShape = {4};
    std::vector<int64_t> logAlphaOutShape = {4, 12, 16};
    void* logProbsDeviceAddr = nullptr;
    void* targetsDeviceAddr = nullptr;
    void* negLoglikelihoodOutDeviceAddr = nullptr;
    void* logAlphaOutDeviceAddr = nullptr;
    aclTensor* logProbs = nullptr;
    aclTensor* targets = nullptr;
    aclIntArray* inputLengths = nullptr;
    aclIntArray* targetLengths = nullptr;
    aclTensor* negLoglikelihoodOut = nullptr;
    aclTensor* logAlphaOut = nullptr;
    std::vector<float> logProbsHostData = {
      -1.0894, -2.7162, -0.9764, -1.9126, -2.6162,
      -2.0684, -2.4871, -2.0866, -1.7205, -0.7187,
      -2.4423, -1.2017, -1.4653, -1.1821, -2.5942,
      -2.4670, -2.7257, -1.4135, -2.1042, -0.7248,

      -3.7759, -1.3742, -1.2549, -1.5807, -1.4562,
      -1.3826, -1.8995, -1.8527, -0.9493, -2.8895,
      -1.6316, -2.6603, -2.5014, -0.6992, -1.8609,
      -1.9269, -2.2350, -0.8073, -1.8906, -1.8947,

      -0.3468, -2.5855, -2.0723, -2.7147, -3.6668,
      -0.9541, -1.7258, -2.0693, -1.6378, -2.1531,
      -3.5386, -3.4830, -0.2532, -2.0557, -3.3261,
      -1.1480, -1.8080, -0.8244, -3.2414, -3.1909,

      -0.8866, -0.7540, -4.4312, -3.4634, -2.6000,
      -1.2785, -1.8347, -3.3122, -0.7620, -2.8349,
      -1.4975, -1.3865, -0.9645, -3.8171, -2.0939,
      -2.3536, -2.0773, -1.4981, -0.8372, -2.0938,

      -1.2186, -0.8285, -2.9399, -2.1159, -2.3620,
      -2.3139, -0.6503, -2.7249, -1.2340, -3.7927,
      -0.7143, -2.5084, -3.2826, -2.6651, -1.1334,
      -1.6965, -1.9728, -2.3849, -1.6052, -0.9554,

      -1.6384, -1.2596, -2.1680, -1.8476, -1.3866,
      -3.0455, -0.5737, -2.5339, -2.1118, -1.6681,
      -2.4675, -2.8842, -0.4329, -3.6266, -1.6925,
      -3.1023, -2.7696, -1.2755, -0.6470, -2.4143,

      -2.0107, -2.0912, -1.3053, -0.8557, -3.0683,
      -1.2872, -3.6523, -1.6703, -2.7596, -0.8063,
      -2.4633, -1.2959, -1.6153, -2.3072, -1.0705,
      -3.0543, -0.6473, -1.1650, -2.9025, -2.7710,

      -3.5519, -2.0400, -1.8667, -1.4289, -0.8050,
      -1.4602, -0.7452, -1.5754, -3.1624, -3.1247,
      -1.4677, -1.2725, -2.9575, -1.8883, -1.2513,
      -1.2164, -1.5894, -2.2217, -2.3714, -1.2110,

      -2.0843, -0.6515, -1.4252, -2.9402, -2.7964,
      -1.5261, -2.5471, -1.7167, -1.9846, -0.9488,
      -1.4847, -1.7093, -1.4095, -1.7293, -1.7675,
      -0.9203, -4.2299, -1.8740, -1.4076, -1.6671,

      -1.9052, -0.8330, -2.1839, -2.2459, -1.6193,
      -2.9108, -1.2114, -1.4616, -1.7297, -1.4330,
      -2.2656, -0.7878, -1.8533, -1.8711, -2.0349,
      -2.2457, -2.1395, -1.4509, -0.7538, -2.6381,

      -0.8078, -2.1054, -2.6703, -1.1108, -3.3867,
      -1.7774, -1.8426, -1.9473, -1.3293, -1.3273,
      -1.3490, -1.9842, -2.5357, -2.2161, -0.8800,
      -1.5412, -1.8003, -2.7603, -0.8606, -2.0066,

      -1.8342, -2.2741, -1.8348, -1.5833, -0.9877,
      -3.5196, -2.3361, -0.9124, -0.9307, -2.5531,
      -1.4862, -1.2153, -1.4453, -3.4462, -1.5625,
      -2.6455, -1.4153, -1.3079, -1.1568, -2.2897};
    std::vector<int64_t> targetsHostData = {
      1, 2, 1, 1, 2, 4, 1,
      2, 2, 2, 2, 2, 2, 3,
      4, 2, 1, 4, 3, 1, 4,
      4, 1, 4, 2, 2, 2, 3};

    std::vector<float> negLoglikelihoodOutHostData = {0, 0, 0, 0};
    std::vector<float> logAlphaOutHostData = {
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,

      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    // Create a logProbs aclTensor.
    ret = CreateAclTensor(logProbsHostData, logProbsShape, &logProbsDeviceAddr, aclDataType::ACL_FLOAT, &logProbs);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a targets aclTensor.
    ret = CreateAclTensor(targetsHostData, targetsShape, &targetsDeviceAddr, aclDataType::ACL_INT64, &targets);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> inputLengthsSizeData = {10,10,10,10};
    inputLengths = aclCreateIntArray(inputLengthsSizeData.data(), 4);
    CHECK_RET(inputLengths != nullptr, return ACL_ERROR_BAD_ALLOC);
    std::vector<int64_t> targetLengthsSizeData = {2, 3, 1, 5};
    targetLengths = aclCreateIntArray(targetLengthsSizeData.data(), 4);
    CHECK_RET(targetLengths != nullptr, return ACL_ERROR_BAD_ALLOC);

    // Create a negLoglikelihoodOut aclTensor.
    ret = CreateAclTensor(negLoglikelihoodOutHostData, negLoglikelihoodOutShape, &negLoglikelihoodOutDeviceAddr, aclDataType::ACL_FLOAT, &negLoglikelihoodOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // Create a logAlphaOut aclTensor.
    ret = CreateAclTensor(logAlphaOutHostData, logAlphaOutShape, &logAlphaOutDeviceAddr, aclDataType::ACL_FLOAT, &logAlphaOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. Call the CANN operator library API. Modify the API as required.
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // Call the first-phase API of aclnnCtcLoss.
    ret = aclnnCtcLossGetWorkspaceSize(logProbs, targets, inputLengths, targetLengths, 0, false, negLoglikelihoodOut, logAlphaOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLossGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // Allocate device memory based on workspaceSize computed by the first-phase API.
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // Call the second-phase API of aclnnCtcLoss.
    ret = aclnnCtcLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCtcLoss failed. ERROR: %d\n", ret); return ret);
    // 4. (Boilerplate) Synchronize the stream and wait for task completion.
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. Obtain the output negLoglikelihoodOut and copy the result from the device to the host. Modify the code based on the API definition.
    auto size = GetShapeSize(negLoglikelihoodOutShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), negLoglikelihoodOutDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("negLoglikelihoodOut result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. Obtain the output logAlphaOut and copy the result from the device to the host. Modify the code based on the API definition.
    auto size1 = GetShapeSize(logAlphaOutShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), logAlphaOutDeviceAddr, size1 * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
      LOG_PRINT("logAlphaOut result[%ld] is: %f\n", i, resultData1[i]);
    }

    // 7. Destroy aclTensor and IntArray. Modify the code based on the API definition.
    aclDestroyTensor(logProbs);
    aclDestroyTensor(targets);
    aclDestroyIntArray(inputLengths);
    aclDestroyIntArray(targetLengths);
    aclDestroyTensor(negLoglikelihoodOut);
    aclDestroyTensor(logAlphaOut);

    // 8. Release device resources. Modify the code based on the API definition.
    aclrtFree(logProbsDeviceAddr);
    aclrtFree(targetsDeviceAddr);
    aclrtFree(negLoglikelihoodOutDeviceAddr);
    aclrtFree(logAlphaOutDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
