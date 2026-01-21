# aclnnForeachNonFiniteCheckAndUnscale

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>|√|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：遍历scaledGrads中的所有Tensor，检查是否存在Inf或NaN，如果存在则将foundInf设置为1.0，否则foundInf保持不变，并对scaledGrads中的所有Tensor进行反缩放。

- 计算公式：

  $$
  foundInf = \begin{cases}1.0, & 当 Inf \in  scaledGrads 或 NaN \in scaledGrads,\\
    foundInf, &其他.
  \end{cases}
  $$

  $$
   scaledGrads_i = {scaledGrads}_{i}*{invScale}.
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnForeachNonFiniteCheckAndUnscale”接口执行计算。

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(
  const aclTensorList *scaledGrads,
  const aclTensor     *foundInf,
  const aclTensor     *invScale,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```Cpp
aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
    <col style="width: 190px">
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
      <td>scaledGrads</td>
      <td>输入/输出</td>
      <td>表示进行反缩放计算的输入和输出张量列表，对应公式中的`scaledGrads`。</td>
      <td><ul><li>不支持空Tensor。</li><li>该参数中所有Tensor的数据类型保持一致。</li><li>支持的最大长度为256个。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>    
    <tr>
      <td>foundInf</td>
      <td>输入/输出</td>
      <td>表示用来标记输入scaledGrads中是否存在Inf或-Inf的张量，对应公式中的`foundInf`。</td>
      <td><ul><li>不支持空Tensor。</li><li>仅包含一个元素。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>invScale</td>
      <td>输入</td>
      <td>表示进行反缩放计算的张量，对应公式中的`invScale`。</td>
      <td><ul><li>不支持空Tensor。</li><li>仅包含一个元素。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
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
  
  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td>传入的scaledGrads、foundInf、invScale是空指针。</td>
    </tr>
    <tr>
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>scaledGrads、foundInf、invScale的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>scaledGrads长度超过限制。</td>
    </tr>
    <tr>
      <td>scaledGrads中的Tensor数据类型不相同。</td>
    </tr>
    <tr>
      <td>foundInf或invScale中的元素个数不为1。</td>
    </tr>
  </tbody></table>

## aclnnForeachNonFiniteCheckAndUnscale

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnForeachNonFiniteCheckAndUnscale默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_non_finite_check_and_unscale.h"

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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // 调用aclrtMemcpy将host侧数据复制到device侧内存上
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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape1 = {2, 3};
  std::vector<int64_t> selfShape2 = {1, 3};
  std::vector<int64_t> foundInfShape = {1};
  std::vector<int64_t> invScaleShape = {1};
  void* input1DeviceAddr = nullptr;
  void* input2DeviceAddr = nullptr;
  void* foundInfDeviceAddr = nullptr;
  void* invScaleDeviceAddr = nullptr;
  aclTensor* input1 = nullptr;
  aclTensor* input2 = nullptr;
  aclTensor* foundInf = nullptr;
  aclTensor* invScale = nullptr;
  std::vector<float> input1HostData = {1, 2, 3, 4, 5, 6};
  std::vector<float> input2HostData = {7, 8, 9};
  std::vector<float> foundInfHostData = {-1.0f};
  std::vector<float> invScaleHostData = {3.1f};
  // 创建input1 aclTensor
  ret = CreateAclTensor(input1HostData, selfShape1, &input1DeviceAddr, aclDataType::ACL_FLOAT, &input1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建input2 aclTensor
  ret = CreateAclTensor(input2HostData, selfShape2, &input2DeviceAddr, aclDataType::ACL_FLOAT, &input2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建foundInf aclTensor
  ret = CreateAclTensor(foundInfHostData, foundInfShape, &foundInfDeviceAddr, aclDataType::ACL_FLOAT, &foundInf);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建invScale aclTensor
  ret = CreateAclTensor(invScaleHostData, invScaleShape, &invScaleDeviceAddr, aclDataType::ACL_FLOAT, &invScale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  std::vector<aclTensor*> tempInput{input1, input2};
  aclTensorList* tensorListInput = aclCreateTensorList(tempInput.data(), tempInput.size());

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnForeachNonFiniteCheckAndUnscale第一段接口
  ret = aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(tensorListInput, foundInf, invScale, &workspaceSize,
                                                             &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnForeachNonFiniteCheckAndUnscale第二段接口
  ret = aclnnForeachNonFiniteCheckAndUnscale(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnForeachNonFiniteCheckAndUnscale failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfShape1);
  std::vector<float> self1Data(size, 0);
  ret = aclrtMemcpy(self1Data.data(), self1Data.size() * sizeof(self1Data[0]), input1DeviceAddr,
                    size * sizeof(self1Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self1 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self1 result[%ld] is: %f\n", i, self1Data[i]);
  }

  size = GetShapeSize(selfShape2);
  std::vector<float> self2Data(size, 0);
  ret = aclrtMemcpy(self2Data.data(), self2Data.size() * sizeof(self2Data[0]), input2DeviceAddr,
                    size * sizeof(self2Data[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy self2 result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("self2 result[%ld] is: %f\n", i, self2Data[i]);
  }

  size = GetShapeSize(foundInfShape);
  std::vector<float> foundInfData(size, 0);
  ret = aclrtMemcpy(foundInfData.data(), foundInfData.size() * sizeof(foundInfData[0]), foundInfDeviceAddr,
                    size * sizeof(foundInfData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("copy foundInf result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("foundInf result is: %f\n", foundInfData[0]);

  // 6. 释放aclTensorList和aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensorList(tensorListInput);
  aclDestroyTensor(foundInf);
  aclDestroyTensor(invScale);

  // 7.释放device资源，需要根据具体API的接口定义修改
  aclrtFree(input1DeviceAddr);
  aclrtFree(input2DeviceAddr);
  aclrtFree(foundInfDeviceAddr);
  aclrtFree(invScaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
