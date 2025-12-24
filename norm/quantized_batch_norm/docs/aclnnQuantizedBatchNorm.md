# aclnnQuantizedBatchNorm

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：
  
  将输入Tensor做一个反量化的计算，再根据输入的weight、bias、epsilon做归一化，最后根据输出的outputScale以及outputZeroPoint做量化。
- 计算公式：
  
  1.反量化计算：
  
  $$
  x' = (x - inputZeroPoint) * inputScale
  $$
  
  2.归一化计算：
  
  $$
  y =\frac{x' - mean}{\sqrt{var + epsilon}} * weight + bias
  $$
  
  3.量化计算：
  
  $$
  output = round(\frac{y}{outputScale} + outputZeroPoint)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantizedBatchNormGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnQuantizedBatchNorm”接口执行计算。

```Cpp
aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(
  const aclTensor* input,
  const aclTensor* mean,
  const aclTensor* var,
  const aclScalar* inputScale,
  const aclScalar* inputZeroPoint,
  const aclScalar* outputScale,
  const aclScalar* outputZeroPoint,
  aclTensor*       weight,
  aclTensor*       bias,
  float            epsilon,
  aclTensor*       output,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```
```Cpp
aclnnStatus aclnnQuantizedBatchNorm(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantizedBatchNormGetWorkspaceSize

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
      <td>input</td>
      <td>输入</td>
      <td>表示模型输入的量化后的数据，对应公式中的`x`。</td>
      <td>不支持空Tensor。</td>
      <td>INT8、UINT8、INT32</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示模型输入数据的均值，对应公式中的`mean`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>var</td>
      <td>输入</td>
      <td>表示模型输入数据的方差，对应公式中的`var`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputScale</td>
      <td>输入</td>
      <td>表示模型输入数据的缩放系数，对应公式中的`inputScale`。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>inputZeroPoint</td>
      <td>输入</td>
      <td>表示模型输入数据的偏置，对应公式中的`inputZeroPoint`。</td>
      <td>传入值不能超过`input`对应数据类型的上下边界，例如INT8上下边界为[-128,127]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    </tr>
    <tr>
      <td>outputScale</td>
      <td>输入</td>
      <td>表示模型输出数据的缩放系数，对应公式中的`outputScale`。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputZeroPoint</td>
      <td>输入</td>
      <td>表示模型输出数据的偏置，对应公式中的`outputZeroPoint`。</td>
      <td>传入值不能超过`input`对应数据类型的上下边界，例如INT8上下边界为[-128,127]。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>可选参数，表示归一化权重，对应公式中的`weight`。</td>
      <td><ul><li>不支持空Tensor。</li><li>默认值为：1。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>可选参数，表示归一化偏置，对应公式中的`bias`。</td>
      <td><ul><li>不支持空Tensor。</li><li>默认值为：0。</li><li>shape长度与入参`input`中channel轴的长度相等。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>输入</td>
      <td>表示添加到方差中的值，以避免出现除以零的情况。对应公式中的`epsilon`。</td>
      <td>-</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>表示模型输出的量化后的数据，对应公式中的`output`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape、数据格式、数据类型与输入`input`保持一致。</li></ul></td>
      <td>INT8，UINT8，INT32</td>
      <td>NCHW</td>
      <td>4</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
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
      <td>传入的input或output是空指针。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>数据类型不在支持的范围之内，或者参数不满足上述约束。</td>
    </tr>
  </tbody></table>

## aclnnQuantizedBatchNorm

- **参数说明**：

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantizedBatchNormGetWorkspaceSize获取。</td>
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

- **返回值**：
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quantized_batch_norm.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorNCHW(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请Device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {1, 2, 1, 4};
    std::vector<int64_t> meanShape = {2};
    std::vector<int64_t> varShape = {2};
    std::vector<int64_t> weightShape = {2};
    std::vector<int64_t> biasShape = {2};
    std::vector<int64_t> outShape = {1, 2, 1, 4};
    void* selfDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* rMeanDeviceAddr = nullptr;
    void* rVarDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* varDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* var = nullptr;
    std::vector<int32_t> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> meanHostData = {0, 0};
    std::vector<float> varHostData = {1, 1};
    std::vector<int32_t> outHostData(8, 0);
    float inputScaleValue = 1.0f;
    int32_t inputZeroPointValue = 1;
    float outputScaleValue = 2.0f;
    int32_t outputZeroPointValue = 1;
    double eps = 1e-5;

    // 创建self aclTensor
    ret = CreateAclTensorNCHW(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_INT32, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建var aclTensor
    ret = CreateAclTensor(varHostData, varShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensorNCHW(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT32, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建inputScale aclScalar
    aclScalar* inputScale = aclCreateScalar(&inputScaleValue, aclDataType::ACL_FLOAT);
    // 创建inputZeroPoint aclScalar
    aclScalar* inputZeroPoint = aclCreateScalar(&inputZeroPointValue, aclDataType::ACL_INT32);
    // 创建outputScale aclScalar
    aclScalar* outputScale = aclCreateScalar(&outputScaleValue, aclDataType::ACL_FLOAT);
    // 创建outputZeroPoint aclScalar
    aclScalar* outputZeroPoint = aclCreateScalar(&outputZeroPointValue, aclDataType::ACL_INT32);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnQuantizedBatchNorm接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // 调用aclnnQuantizedBatchNorm第一段接口
    ret = aclnnQuantizedBatchNormGetWorkspaceSize(
        self, mean, var, inputScale, inputZeroPoint, outputScale, outputZeroPoint, weight, bias, eps, out,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizedBatchNormGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnQuantizedBatchNorm第二段接口
    ret = aclnnQuantizedBatchNorm(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantizedBatchNorm failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    printf("size is %ld", size);
    std::vector<int32_t> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(mean);
    aclDestroyTensor(var);
    aclDestroyTensor(out);
    aclDestroyScalar(inputScale);
    aclDestroyScalar(inputZeroPoint);
    aclDestroyScalar(outputScale);
    aclDestroyScalar(outputZeroPoint);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(varDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```