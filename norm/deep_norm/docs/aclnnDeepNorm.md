# aclnnDeepNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：对输入张量x的元素进行深度归一化，通过计算其均值和标准差，将每个元素标准化为具有零均值和单位方差的输出张量。
- 计算公式：
  
  $$
  DeepNorm(x_i^{\prime}) = ({x_i^{\prime} - \bar{x^{\prime}}})*{rstd} * gamma + beta,
  $$
  $$
  \text { where } rstd = \frac{1} {\sqrt{\frac{1}{n} \sum_{i=1}^n (x^{\prime}_i - \bar{x^{\prime}})^2 + epsilon} }, \quad \operatorname{x^{\prime}_i} = alpha * x_i   + gx_i
  $$
## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnDeepNormGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDeepNorm”接口执行计算。

```Cpp
aclnnStatus aclnnDeepNormGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gx,
  const aclTensor *beta,
  const aclTensor *gamma,
  double           alpha,
  double           epsilon,
  const aclTensor *meanOut,
  const aclTensor *rstdOut,
  const aclTensor *yOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```
```Cpp
aclnnStatus aclnnDeepNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnDeepNormGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1503px"><colgroup>
  <col style="width: 146px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 392px">
  <col style="width: 228px">
  <col style="width: 101px">
  <col style="width: 100px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示输入数据，通常为神经网络的中间层输出，对应公式中的`x`。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gx</td>
      <td>输入</td>
      <td>表示输入数据的梯度，用于反向传播，对应公式中的`gx`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape维度和输入`x`的维度相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>输入</td>
      <td>表示偏置参数，用于调整归一化后的输出，对应公式中的`beta`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape维度和输入`x`后几维的维度相同，后几维表示需要norm的维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示缩放参数，用于调整归一化后的输出，对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape维度和输入`x`后几维的维度相同，后几维表示需要norm的维度。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha</td>
      <td>输入</td>
      <td>表示权重参数，用于调整输入数据的权重，对应公式中的`alpha`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`epsilon`。</td>
      <td>-</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>meanOut</td>
      <td>输出</td>
      <td>表示计算输出的均值，用于归一化操作，对应公式中的`mean`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与输入`x`满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>（前几维的维度和输入`x`前几维的维度相同，前几维表示不需要norm的维度，其余维度大小为1）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut</td>
      <td>输出</td>
      <td>表示计算输出的标准差倒数，用于归一化操作，对应公式中`rstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与输入`x`满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>（前几维的维度和输入`x`前几维的维度相同，前几维表示不需要norm的维度，其余维度大小为1）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>表示归一化后的输出数据，对应公式中的`y`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`x`的数据类型保持一致。</li><li>shape维度和输入`x`的维度相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>            
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

-   **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入参数是必选输入，输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入和输出的shape不匹配或者不在支持的维度范围内。</td>
    </tr>
    <tr>
      <td>输入和输出的数据类型不满足参数说明中的约束。</td>
    </tr>
  </tbody></table>

## aclnnDeepNorm

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDeepNormGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>


- **返回值：**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- 功能维度：
  - 数据类型支持：
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x、gx、beta、gamma、yOut支持FLOAT32、FLOAT16、BFLOAT16。
    - rstdOut、meanOut支持：FLOAT32。
  - 数据格式支持：ND

- 未支持类型说明：

  DOUBLE：指令不支持DOUBLE。

- 边界值场景说明：
  - 当输入是inf时，输出为inf。
  - 当输入是NaN时，输出为NaN。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deep_norm.h"

#define CHECK_RET(cond, return_expr)\
do {                                \
  if (!(cond)) {                    \
    return_expr;                    \
  }                                 \
} while (0)

#define LOG_PRINT(message, ...)   \
    do {                          \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  float alpha = 0.3;
  float eps = 1e-6;
  std::vector<int64_t> xShape = {3, 1, 4};
  std::vector<int64_t> gxShape = {3, 1, 4};
  std::vector<int64_t> betaShape = {4};
  std::vector<int64_t> gammaShape = {4};
  std::vector<int64_t> outputMeanShape = {3, 1, 1};
  std::vector<int64_t> outputRstdShape = {3, 1, 1};
  std::vector<int64_t> outputYShape = {3, 1, 4};

  void *xDeviceAddr = nullptr;
  void *gxDeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *outputMeanDeviceAddr = nullptr;
  void *outputRstdDeviceAddr = nullptr;
  void *outputYDeviceAddr = nullptr;

  aclTensor *x = nullptr;
  aclTensor *gx = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *outputMean = nullptr;
  aclTensor *outputRstd = nullptr;
  aclTensor *outputY = nullptr;

  std::vector<float> xHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> gxHostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
  std::vector<float> betaHostData = {0, 1, 2, 3};
  std::vector<float> gammaHostData = {0, 1, 2, 3};
  std::vector<float> outputMeanHostData = {0, 1, 2};
  std::vector<float> outputRstdHostData = {0, 1, 2};
  std::vector<float> outputYHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

  // 创建self aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gxHostData, gxShape, &gxDeviceAddr, aclDataType::ACL_FLOAT, &gx);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  betaShape, & betaDeviceAddr, aclDataType::ACL_FLOAT, & beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(outputMeanHostData, outputMeanShape, &outputMeanDeviceAddr, aclDataType::ACL_FLOAT, &outputMean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outputRstdHostData, outputRstdShape, &outputRstdDeviceAddr, aclDataType::ACL_FLOAT, &outputRstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outputYHostData, outputYShape, &outputYDeviceAddr, aclDataType::ACL_FLOAT, &outputY);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // aclnnDeepNorm接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnDeepNorm第一段接口
  LOG_PRINT("\nUse aclnnDeepNorm Port.");
  ret = aclnnDeepNormGetWorkspaceSize(x, gx, beta, gamma, alpha, eps,
                                      outputMean, outputRstd, outputY,
                                      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNormGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnDeepNorm第二段接口
  ret = aclnnDeepNorm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeepNorm failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto outputMeanSize = GetShapeSize(outputMeanShape);
  std::vector<float> resultDataMean(outputMeanSize, 0);
  ret = aclrtMemcpy(resultDataMean.data(), resultDataMean.size() * sizeof(resultDataMean[0]), outputMeanDeviceAddr,
                    outputMeanSize * sizeof(resultDataMean[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("== pdx output");
  for (int64_t i = 0; i < outputMeanSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataMean[i]);
  }

  auto outputRstdSize = GetShapeSize(outputRstdShape);
  std::vector<float> resultDataRstd(outputRstdSize, 0);
  ret = aclrtMemcpy(resultDataRstd.data(), resultDataRstd.size() * sizeof(resultDataRstd[0]), outputRstdDeviceAddr,
                    outputRstdSize * sizeof(resultDataRstd[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("== pdx output");
  for (int64_t i = 0; i < outputRstdSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataRstd[i]);
  }

  auto outputYSize = GetShapeSize(outputYShape);
  std::vector<float> resultDataY(outputYSize, 0);
  ret = aclrtMemcpy(resultDataY.data(), resultDataY.size() * sizeof(resultDataY[0]), outputYDeviceAddr,
                    outputYSize * sizeof(resultDataY[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("== pdx output");
  for (int64_t i = 0; i < outputYSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataY[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(gx);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(outputMean);
  aclDestroyTensor(outputRstd);
  aclDestroyTensor(outputY);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(gxDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(outputMeanDeviceAddr);
  aclrtFree(outputRstdDeviceAddr);
  aclrtFree(outputYDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}

```
