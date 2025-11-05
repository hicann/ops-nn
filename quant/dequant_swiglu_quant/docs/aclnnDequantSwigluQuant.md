# aclnnDequantSwigluQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明
- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。
- 计算公式：

  $$
  dequantOut = Dequant(x, weightScaleOptional, activationScaleOptional, biasOptional)
  $$

  $$
  swigluOut = Swiglu(dequantOut)=Swish(A)*B
  $$

  $$
  out = Quant(swigluOut, quantScaleOptional, quantOffsetOptional)
  $$

  其中，A表示dequantOut的前半部分，B表示dequantOut的后半部分。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnDequantSwigluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDequantSwigluQuant”接口执行计算。

```Cpp
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *weightScaleOptional,
    const aclTensor *activationScaleOptional,
    const aclTensor *biasOptional,
    const aclTensor *quantScaleOptional,
    const aclTensor *quantOffsetOptional,
    const aclTensor *groupIndexOptional,
    bool             activateLeft,
    char            *quantModeOptional,
    const aclTensor *yOut,
    const aclTensor *scaleOut,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuant(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```


## aclnnDequantSwigluQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>输入待处理的数据，公式中的x。</td>
      <td>shape为(N...,H)，最后一维需要是2的倍数，且x的维度必须大于1维。</td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weightScaleOptional</td>
      <td>输入</td>
      <td>weight的反量化scale，公式中的weightScaleOptional。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <td>activationScaleOptional</td>
      <td>输入</td>
      <td>激活函数的反量化scale，公式中的activationScaleOptional。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>Matmul的bias，公式中的biasOptional。</td>
      <td>shape支持1维，shape表示为[H]，且取值H和x最后一维保持一致。可选参数，支持传空指针。</td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantScaleOptional</td>
      <td>输入</td>
      <td>量化的scale，公式中的quantScaleOptional。</td>
      <td>-</td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantOffsetOptional</td>
      <td>输入</td>
      <td>量化的offset，公式中的quantOffsetOptional。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td>-</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>输入</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td>当值为false时，对输入的右半部分做激活。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>输入</td>
      <td>表示使用动态量化还是静态量化。</td>
      <td>支持“dynamic”和“static"。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>-</td>
      <td>-</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>-</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
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

    - weightScaleOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT，shape支持1维，shape表示为[H]，且取值H和x最后一维保持一致。可选参数，支持传空指针。
    - activationScaleOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT，shape为[N..., 1]，最后一维为1，其余和x保持一致。可选参数，支持传空指针。
    - quantScaleOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT、FLOAT16，当quantModeOptional为static时，shape为1维，值为1，shape表示为shape[1]；quantModeOptional为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。
    - quantOffsetOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT，当quantModeOptional为static时，shape为1维，值为1，shape表示为shape[1]：quantModeOptional为dynamic时，shape维数为1维，值为x的最后一维的二分之一，shape表示为shape[H/2]。可选参数，支持传空指针。
    - groupIndexOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持INT32、INT64，shape支持1维Tensor。可选参数，支持传空指针。
    - quantModeOptional参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：支持“dynamic”和“static"。
    - yOut参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持INT8。
    - scaleOut参数：
      - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：数据类型支持FLOAT。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  第一段接口会完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的x、yOut或scaleOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的参数维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不满足约束要求。</td>
    </tr>
    <tr>
      <td>输入的取值不满足要求。</td>
    </tr>
  </tbody></table>


## aclnnDequantSwigluQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDequantSwigluQuantGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。


## 约束说明

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>：
  - x的最后一维需要是2的倍数，且x的维数必须大于1维。
  - 当quantModeOptional为static时，quantScaleOptional和quantOffsetOptional为1维，值为1；quantModeOptional为dynamic时，quantScaleOptional和quantOffsetOptional的维数为1维，值为x的最后一维除以2。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> scaleShape = {16};
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> outShape = {2, 16};
  std::vector<int64_t> scaleOutShape = {2};
  void* xDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建scale aclTensor
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建offset aclTensor
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scaleOut aclTensor
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnDequantSwigluQuant第一段接口
  ret = aclnnDequantSwigluQuantGetWorkspaceSize(x, nullptr, nullptr, nullptr, scale, nullptr, nullptr, false, "dynamic", out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnDequantSwigluQuant第二段接口
  ret = aclnnDequantSwigluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```