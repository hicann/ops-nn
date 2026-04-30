# aclnnSigmoidCrossEntropyWithLogitsV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss/sigmoid_cross_entropy_with_logits_v2)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：计算 SigmoidCrossEntropyWithLogitsV2 损失。

- 计算公式：

  设输入为 self（logits）与 target，定义：

  $$
  max\_val = \max(-self, 0)
  $$

  $$
  log\_term = max\_val + \log\left(\exp(-max\_val) + \exp(-self - max\_val)\right)
  $$

  不带 posWeight 时：

  $$
  loss = (1 - target) * self + log\_term
  $$

  带 posWeight 时：

  $$
  log\_weight = (posWeight - 1) * target + 1
  $$

  $$
  loss = (1 - target) * self + log\_weight * log\_term
  $$

  若 weight 非空：

  $$
  loss = loss * weight
  $$

  reduction 语义：0 表示 none，1 表示 mean，2 表示 sum。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnBinaryCrossEntropyWithLogits”接口执行计算。

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(
    const aclTensor* self,
    const aclTensor* target,
    const aclTensor* weightOptional,
    const aclTensor* posWeightOptional,
    int64_t reduction,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)

aclnnStatus aclnnBinaryCrossEntropyWithLogits(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize

- **参数说明：**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1475px"><colgroup>
  <col style="width: 205px">
  <col style="width: 120px">
  <col style="width: 320px">
  <col style="width: 320px">
  <col style="width: 130px">
  <col style="width: 115px">
  <col style="width: 120px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">self(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入 logits，公式中的 self。</td>
      <td class="tg-0pky">shape 需与 target 一致。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">target(aclTensor*)</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">标签输入，公式中的 target。</td>
      <td class="tg-0pky">shape 需与 self 一致。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional(aclTensor*)</td>
      <td class="tg-0pky">可选输入</td>
      <td class="tg-0pky">样本权重。</td>
      <td class="tg-0pky">若存在需可广播到 target 的 shape。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">posWeightOptional(aclTensor*)</td>
      <td class="tg-0pky">可选输入</td>
      <td class="tg-0pky">正样本权重。</td>
      <td class="tg-0pky">若存在需可广播到 target 的 shape。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction（int64_t）</td>
      <td class="tg-0pky">可选属性</td>
      <td class="tg-0pky">指定输出缩减方式。</td>
      <td class="tg-0pky">支持 0/1/2：0-none，1-mean，2-sum。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">out(aclTensor*)</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">输出损失值。</td>
      <td class="tg-0pky">当 reduction=0，out shape 与 self 一致；当 reduction=1/2，out 需为标量 shape。</td>
      <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">0 维或与输入同形</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在 Device 侧申请的 workspace 大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回 op 执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 120px">
  <col style="width: 761px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回值</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">传入的 self、target 或 out 是空指针时。</td>
    </tr>
    <tr>
      <td class="tg-0pky" rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky" rowspan="7">161002</td>
      <td class="tg-0pky">self、target 或 out 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">out 的数据类型与 target 不一致。</td>
    </tr>
    <tr>
      <td class="tg-0pky">self 与 target 的 shape 不一致。</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightOptional 或 posWeightOptional 无法广播到 target 的 shape。</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction 不在 0/1/2 范围内。</td>
    </tr>
    <tr>
      <td class="tg-0pky">输入维度超过 8 维。</td>
    </tr>
    <tr>
      <td class="tg-0pky">reduction 为 mean/sum 时，out 不是标量 shape。</td>
    </tr>
  </tbody>
  </table>

## aclnnBinaryCrossEntropyWithLogits

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
  <col style="width: 184px">
  <col style="width: 134px">
  <col style="width: 833px">
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
          <td>在 Device 侧申请的 workspace 内存地址。</td>
        </tr>
        <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize 获取。</td>
        </tr>
        <tr>
          <td>executor</td>
          <td>输入</td>
          <td>op 执行器，包含了算子计算流程。</td>
        </tr>
        <tr>
          <td>stream</td>
          <td>输入</td>
          <td>指定执行任务的 Stream。</td>
        </tr>
      </tbody>
    </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
    - aclnnBinaryCrossEntropyWithLogits 默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnn_binary_cross_entropy_with_logits.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> commonShape = {32, 32};
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* posWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;

  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* posWeight = nullptr;
  aclTensor* out = nullptr;

  int64_t elementCount = GetShapeSize(commonShape);
  std::vector<float> selfHostData(elementCount, 1.0f);
  std::vector<float> targetHostData(elementCount, 0.5f);
  std::vector<float> weightHostData(elementCount, 1.0f);
  std::vector<float> posWeightHostData(elementCount, 2.0f);
  std::vector<float> outHostData(elementCount, 0.0f);

  ret = CreateAclTensor(selfHostData, commonShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(targetHostData, commonShape, &targetDeviceAddr, aclDataType::ACL_FLOAT, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(weightHostData, commonShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(posWeightHostData, commonShape, &posWeightDeviceAddr, aclDataType::ACL_FLOAT, &posWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outHostData, commonShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int64_t reduction = 0; // 0:none, 1:mean, 2:sum

  // 调用aclnnBinaryCrossEntropyWithLogitsGet第一段接口
  ret = aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(
      self, target, weight, posWeight, reduction, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize failed. ERROR: %d\\n", ret); return ret);

  // 根据第一段接口返回的 workspaceSize 申请 device 内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\\n", ret); return ret);
  }

  // 调用aclnnBinaryCrossEntropyWithLogitsGet第二段接口
  ret = aclnnBinaryCrossEntropyWithLogits(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBinaryCrossEntropyWithLogits failed. ERROR: %d\\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\\n", ret); return ret);

  // 5. 获取输出并回拷到 host
  auto size = GetShapeSize(commonShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\\n", i, resultData[i]);
  }

  // 6. 释放 aclTensor
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(posWeight);
  aclDestroyTensor(out);

  // 7. 释放 device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(posWeightDeviceAddr);
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