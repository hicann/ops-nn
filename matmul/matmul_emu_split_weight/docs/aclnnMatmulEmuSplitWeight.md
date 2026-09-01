# aclnnMatmulEmuSplitWeight

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：使用双路BF16 GEMM融合模拟FP32精度矩阵乘法。离线将FP32权重拆分为高位BF16与低位残差BF16，推理阶段执行两次BF16 GEMM并做线性组合，激活值全程保持BF16，两路矩阵乘均运行在Cube上。

- 计算公式：

  $$
  Y = X \times W_{high} + scale \times (X \times W_{low})
  $$

  其中 $X$ 为BF16激活矩阵，$W_{high}$ 为FP32权重截断BF16得到的高位权重，$W_{low}$ 为FP32权重残差除以scale后截断BF16得到的低位残差权重，$scale$ 为缩放因子（典型值1/256=0.00390625）。

- 权重离线拆分公式：

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnMatmulEmuSplitWeightGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnMatmulEmuSplitWeight"接口执行计算。

```cpp
aclnnStatus aclnnMatmulEmuSplitWeightGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *wHigh,
  const aclTensor *wLow,
  const aclTensor *y,
  float            wLowScale,
  int8_t           yDtype,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMatmulEmuSplitWeight(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnMatmulEmuSplitWeightGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
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
        <th>维度</th>
        <th>非连续</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>表示激活矩阵，对应公式中的$X$。</td>
        <td><li>数据类型必须为BF16。</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wHigh</td>
        <td>输入</td>
        <td>表示高位权重，对应公式中的$W_{high}$。</td>
        <td><li>数据类型必须为BF16，与x一致。</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wLow</td>
        <td>输入</td>
        <td>表示低位残差权重，对应公式中的$W_{low}$。</td>
        <td><li>数据类型必须为BF16，与x一致。</li><li>shape必须与wHigh完全一致。</li></td>
        <td>BFLOAT16</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>wLowScale</td>
        <td>输入</td>
        <td>缩放因子，对应公式中的scale。</td>
        <td><li>仅支持1/256（0.00390625）。</li></td>
        <td>float</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yDtype</td>
        <td>输入</td>
        <td>输出数据类型标识。</td>
        <td><li>仅支持0（FP32）。</li></td>
        <td>int8_t</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>表示计算的输出矩阵，对应公式中的$Y$。</td>
        <td><li>仅支持FP32输出（yDtype=0）。</li><li>shape为[M, N]。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景报错：
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thead>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>传入的x、wHigh、wLow或y是空指针。</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>x、wHigh、wLow的数据类型不为BFLOAT16。</td>
      </tr>
      <tr>
        <td>y的数据类型不为FLOAT32。</td>
      </tr>
      <tr>
        <td>yDtype不为0。</td>
      </tr>
      <tr>
        <td>wLowScale为NaN或Inf，或wLowScale不等于1/256（0.00390625）。</td>
      </tr>
      <tr>
        <td>输入的数据格式不为ND。</td>
      </tr>
      <tr>
        <td>输入的维度不为2维，或K维度不匹配（x的K与wHigh/wLow的K不一致），或wLow的shape与wHigh不一致，或y shape与[M, N]不匹配，或M、K、N不在(0, INT32_MAX]范围内。</td>
      </tr>
  </tbody></table>

## aclnnMatmulEmuSplitWeight

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnMatmulEmuSplitWeightGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：aclnnMatmulEmuSplitWeight默认确定性实现。

  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：aclnnMatmulEmuSplitWeight默认确定性实现。
  <!-- end id8 -->

- **Shape约束**：
  - M、K、N取值范围为(0, INT32_MAX]。
- **转置**：aclnn接口本身不接收转置属性。如需转置，调用方应在传入前对tensor做permute处理，算子支持非连续Tensor，无需额外做contiguous。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_matmul_emu_split_weight.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. device/stream初始化
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出
  // x: BF16 [128, 256], wHigh/wLow: BF16 [256, 128], y: FP32 [128, 128]
  std::vector<int64_t> xShape = {128, 256};
  std::vector<int64_t> wShape = {256, 128};
  std::vector<int64_t> yShape = {128, 128};
  void* xDeviceAddr = nullptr;
  void* wHighDeviceAddr = nullptr;
  void* wLowDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* wHigh = nullptr;
  aclTensor* wLow = nullptr;
  aclTensor* y = nullptr;

  // 构造BF16数据（这里用float占位，实际使用时需传入BF16数据）
  std::vector<float> xHostData(128 * 256, 1.0f);
  std::vector<float> wHighHostData(256 * 128, 1.0f);
  std::vector<float> wLowHostData(256 * 128, 0.0f);
  std::vector<float> yHostData(128 * 128, 0.0f);

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wHighHostData, wShape, &wHighDeviceAddr, aclDataType::ACL_BF16, &wHigh);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(wLowHostData, wShape, &wLowDeviceAddr, aclDataType::ACL_BF16, &wLow);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用aclnnMatmulEmuSplitWeight
  // wLowScale: 缩放因子，仅支持1/256 (0.00390625)
  // yDtype: 输出数据类型，仅支持0 (FP32)
  float wLowScale = 0.00390625f;
  int8_t yDtype = 0;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  ret = aclnnMatmulEmuSplitWeightGetWorkspaceSize(x, wHigh, wLow, y, wLowScale, yDtype, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulEmuSplitWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnMatmulEmuSplitWeight(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulEmuSplitWeight failed. ERROR: %d\n", ret); return ret);

  // 4. 同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值
  auto size = GetShapeSize(yShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < std::min(size, 16L); i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放资源
  aclDestroyTensor(x);
  aclDestroyTensor(wHigh);
  aclDestroyTensor(wLow);
  aclDestroyTensor(y);
  aclrtFree(xDeviceAddr);
  aclrtFree(wHighDeviceAddr);
  aclrtFree(wLowDeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
