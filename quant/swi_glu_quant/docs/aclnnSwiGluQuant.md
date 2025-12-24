# aclnnSwiGluQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 算子功能：在SwiGlu激活函数后添加quant操作，实现输入x的SwiGluQuant计算
- 算子支持范围：当前SwiGluQuant**仅支持MoE场景**，SwiGluQuant的输入x和group_index来自于GroupedMatMul算子和MoeInitRouting的输出，通过group_index入参实现MoE分组动态量化、静态per_tensor量化、静态per_channel量化功能。
- 动态量化计算公式：  
  
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
    scale=row\_max(abs(Y_{tmp}))/127
  $$

  $$
    Y = Cast(Mul(Y_{tmp}, Scale))
  $$
     其中，A表示输入x的前半部分，B表示输入x的后半部分，g表示group_index，G为group_index的分组数量。
     
- 静态量化计算公式：
  
  $$
    Act = SwiGLU(x) = Swish(A)*B \\
    Y_{tmp}[0\colon g[0],\colon] = Act[0\colon g[0],\colon] * smooth\_scales[0,\colon] + offsets[0,\colon], i=0 \\
    Y_{tmp}[g[i]\colon g[i+1], \colon] = Act[g[i]\colon g[i+1], \colon] *  smooth\_scales[i+1, \colon] + offsets[i+1, \colon], i \in (0, G) \cap \mathbb{Z}\\
  $$
  $$
    Y = Cast(Y_{tmp})
  $$
  其中，A表示输入x的前半部分，B表示输入x的后半部分，g表示group_index，G为group_index的分组数量。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwiGluQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwiGluQuant”接口执行计算。
```Cpp
aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *smoothScalesOptional,
  const aclTensor *offsetsOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwiGluQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```


## aclnnSwiGluQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1350px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 270px">
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
      <td><ul><li>不支持空Tensor。</li><li>x的最后一维需要为2的倍数，且x的维数必须大于1维，当前仅支持输入x的最后一维长度不超过8192。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>smoothScalesOptional</td>
      <td>输入</td>
      <td>量化的smooth_scales，公式中的smooth_scales。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape支持[G, N]，[G, ]，其中G代表groupIndex分组数量，N为计算输入x的最后一维大小的二分之一，当前版本下不支持用户传入空指针。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
     <tr>
      <td>offsetsOptional</td>
      <td>输入</td>
      <td>公式中的offsets。</td>
      <td><ul><li>支持空Tensor。</li><li>该参数在动态量化场景下不生效，用户传入空指针即可。</li><li>静态量化场景下：数据类型支持FLOAT。</li><li>per_channel模式下shape支持[G, N]。</li><li>per_tensor模式下shape支持[G, ]，且数据类型和shape需要与smoothScalesOptional保持一致。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>MoE分组需要的group_index，公式中的group_index。</td>
      <td><ul><li>支持空Tensor。</li><li>shape支持[G, ]，group_index内元素要求为非递减，且最大值不得超过输入x的除最后一维之外的所有维度大小之积；G的值不得超过输入x的除最后一维之外的所有维度大小之积。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>输入</td>
      <td>计算输入。</td>
      <td>true代表激活输入x的前半部分，false代表激活输入x后半部分。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>quantModeOptional</td>
      <td>输入</td>
      <td>计算输入。</td>
      <td>"static"表示静态量化、"dynamic"表示动态量化、"dynamic_msd"表示动态MSD量化。当前仅支持"dynamic"动态量化, "static"静态量化。静态量化仅支持per_tensor量化和per_channel量化，用户传入空指针时代表动态量化。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>计算输出yOut的shape最后一维大小为计算输入x最后一维的二分之一，其余维度与x保持一致。</td>
      <td>INT8</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>输出张量。</td>
      <td>计算输出scaleOut的shape与计算输入x相比，无最后一维，其余维度与计算输入x保持一致。</td>
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
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
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
      <td>传入的x或yOut是空指针。</td>
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
      <td>quantModeOptional不在指定的取值范围内。</td>
    </tr>
  </tbody></table>


## aclnnSwiGluQuant
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwiGluQuantGetWorkspaceSize获取。</td>
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

无。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swi_glu_quant.h"

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
  // 固定写法，AscendCL初始化
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
  // 1. （固定写法）device/stream初始化，参考AscendCL对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {3, 32};
  std::vector<int64_t> smoothScalesShape = {2, 16};
  std::vector<int64_t> groupIndexShape = {2};
  std::vector<int64_t> outShape = {3, 16};
  std::vector<int64_t> scaleShape = {3};
  void* xDeviceAddr = nullptr;
  void* smoothScalesDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* smoothScales = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scale = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                  23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                  43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 
                                  63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 
                                  83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95};
  std::vector<float> smoothScalesHostData = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                                      1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<float> groupIndexHostData = {1, 3};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleHostData = {0, 0, 0};
  
  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建scale aclTensor
  ret = CreateAclTensor(smoothScalesHostData, smoothScalesShape, &smoothScalesDeviceAddr, aclDataType::ACL_FLOAT, &smoothScales);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建groupIndex aclTensor
  ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT32, &groupIndex);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scale aclTensor
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSwiGluQuant第一段接口
  ret = aclnnSwiGluQuantGetWorkspaceSize(x, smoothScales, nullptr, groupIndex, false, "dynamic", out, scale, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwiGluQuant第二段接口
  ret = aclnnSwiGluQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwiGluQuant failed. ERROR: %d\n", ret); return ret);

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
  aclDestroyTensor(smoothScales);
  aclDestroyTensor(groupIndex);
  aclDestroyTensor(out);
  aclDestroyTensor(scale);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothScalesDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```