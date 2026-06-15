# aclnnSwigluGroupQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/swiglu_group_quant)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：在SwiGLU激活后执行分组低比特量化，支持FP8和FP4输出。
- 计算公式：
  令输入x的最后一维为D，左半部分为A，右半部分为B，计算：

  $$
  y_{tmp}=silu(A) \times B
  $$

  若传入weight，则量化前执行：

  $$
  y_{tmp}=y_{tmp} \times weight
  $$

  进行量化：

  $$
    scale=row\_max(abs(y_{tmp}))/dstTypeScale
  $$

  $$
    y = Cast(Mul(y_{tmp}, scale))
  $$

  quant_mode为0时输出FP8类型的yOut和FLOAT32类型的scaleOut；quant_mode为1时输出FP8/FP4类型的yOut和FLOAT8_E8M0类型的scaleOut。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwigluGroupQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroupQuant”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  int64_t          quantMode,
  int64_t          blockSize,
  bool             roundScale,
  double           clampLimit,
  bool             outputOrigin,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 260px">
  <col style="width: 110px">
  <col style="width: 220px">
  <col style="width: 420px">
  <col style="width: 180px">
  <col style="width: 90px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU输入。</td>
      <td><ul><li>shape为[...,D]。</li><li>D必须大于等于256，且能被256整除。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>量化前按token乘到SwiGLU输出上的权重。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li>不为空时，数据类型为FLOAT32，元素个数需等于x除最后一维外的元素个数之积。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>count模式的group token数。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li>不为空时，数据类型为INT64，shape为[groupNum]。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化类型。</td>
      <td><ul><li>支持取值35、36、40、41，分别表示FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。</li><li>dstType为40或41时，quantMode必须为1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>支持取值0和1。</li><li>0表示Block FP8模式。</li><li>1表示MX模式。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>量化块大小。</td>
      <td><ul><li>0表示使用当前量化模式的默认block大小。</li><li>quantMode为0时，支持0或128。</li><li>quantMode为1时，支持0或32。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>是否将scale取整为2的幂。</td>
      <td>quantMode为1时，roundScale必须为true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>SwiGLU计算前的clamp阈值。</td>
      <td><ul><li>-1.0表示不启用clamp。</li><li>启用clamp时，clampLimit必须大于0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否输出量化前的SwiGLU结果。</td>
      <td>MX FP4模式下，yOriginOut仅作占位。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化输出。</td>
      <td><ul><li>dstType为35或36时，shape为[...,D/2]。</li><li>dstType为40或41时，shape为[...,D/4]。</li><li>数据类型需与dstType一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化scale输出。</td>
      <td><ul><li>quantMode为0时，shape为[...,ceil((D/2)/128)]，数据类型为FLOAT32。</li><li>quantMode为1时，shape为[...,ceil(ceil((D/2)/32)/2),2]，数据类型为FLOAT8_E8M0。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1-8</td>
      <td>×</td>
    </tr>
    <tr>
      <td>yOriginOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化前的SwiGLU结果。</td>
      <td><ul><li>shape为[...,D/2]。</li><li>数据类型需与x一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
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
      <td>x、yOut、scaleOut、yOriginOut、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不满足约束。</td>
    </tr>
    <tr>
      <td>dstType、quantMode、blockSize、roundScale或clampLimit不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>dstType、quantMode、blockSize、roundScale和scaleOut数据类型组合不匹配。</td>
    </tr>
    <tr>
      <td>weightOptional或groupIndexOptional不满足可选输入约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- quantMode为0时，仅支持FP8输出，blockSize支持0或128。
- quantMode为1时，支持FP8/FP4输出，blockSize支持0或32，roundScale必须为true。
- dstType为FLOAT4_E2M1或FLOAT4_E1M2时，必须使用quantMode=1。
- scaleOut的数据类型必须与quantMode匹配：Block FP8为FLOAT32，MX为FLOAT8_E8M0。
- 确定性计算：
  - aclnnSwigluGroupQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant.h"

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
  for (auto dim : shape) {
    shapeSize *= dim;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return ACL_SUCCESS;
}

bool CheckHardwareSupport() {
  const char* socName = aclrtGetSocName();
  if (socName == nullptr) {
    LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
    return true;
  }

  LOG_PRINT("Current SOC: %s\n", socName);
  if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
    return true;
  }

  LOG_PRINT("Warning: SwigluGroupQuant only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
            socName);
  return false;
}

void Finalize(int32_t deviceId, aclrtStream stream) {
  (void)aclrtDestroyStream(stream);
  (void)aclrtResetDevice(deviceId);
  (void)aclFinalize();
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return ACL_SUCCESS;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  if (!CheckHardwareSupport()) {
    LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
    Finalize(deviceId, stream);
    return ACL_SUCCESS;
  }

  std::vector<int64_t> xShape = {2, 256};
  std::vector<int64_t> yShape = {2, 128};
  std::vector<int64_t> scaleOutShape = {2, 1};
  std::vector<int64_t> yOriginShape = {2, 128};

  std::vector<uint16_t> xHostData(GetShapeSize(xShape), 0);
  for (size_t i = 0; i < xHostData.size(); ++i) {
    xHostData[i] = static_cast<uint16_t>(i % 23);
  }
  std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
  std::vector<float> scaleOutHostData(GetShapeSize(scaleOutShape), 0.0f);
  std::vector<uint16_t> yOriginHostData(GetShapeSize(yOriginShape), 0);

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  void* yOriginDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* y = nullptr;
  aclTensor* scaleOut = nullptr;
  aclTensor* yOrigin = nullptr;

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, ACL_FLOAT8_E4M3FN, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginDeviceAddr, ACL_FLOAT16, &yOrigin);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  int64_t dstType = 36;
  int64_t quantMode = 0;
  int64_t blockSize = 0;
  bool roundScale = false;
  double clampLimit = -1.0;
  bool outputOrigin = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  ret = aclnnSwigluGroupQuantGetWorkspaceSize(x, nullptr, nullptr, dstType, quantMode, blockSize, roundScale,
                                              clampLimit, outputOrigin, y, scaleOut, yOrigin, &workspaceSize,
                                              &executor);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::vector<uint8_t> resultData(GetShapeSize(yShape), 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("result[0] is: %d\n", resultData[0]);

  aclDestroyTensor(x);
  aclDestroyTensor(y);
  aclDestroyTensor(scaleOut);
  aclDestroyTensor(yOrigin);
  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  aclrtFree(yOriginDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  Finalize(deviceId, stream);
  return ACL_SUCCESS;
}
```
