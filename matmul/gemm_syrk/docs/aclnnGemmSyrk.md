# aclnnGemmSyrk

## 产品支持情况

<!-- npu="950" id10 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id10 -->
<!-- npu="A3" id1 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="310b" id3 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id3 -->
<!-- npu="310p" id4 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id4 -->
<!-- npu="910" id5 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id5 -->

## 功能说明

- 接口功能：完成对称秩k更新（syrk，参考 cublas `?syrk`）计算，计算 $\alpha$ 乘以 A 与其转置的乘积，再与
  $\beta$ 和 C 的乘积求和，结果原地写回 C（输出完整对称矩阵，上三角和下三角均写出，而非只写单个三角）。
- 计算公式（transposeX = false，a 为 (…, m, k)，2-6 维，前面为 batch 轴）：

  $$
  C = \alpha \times (A @ A^T) + \beta \times C
  $$

  transposeX = true 时，a 以转置的 (…, k, m) 布局存储，计算
  $C = \alpha \times (A^T @ A) + \beta \times C$（对应 cublas syrk 的 OP_T 语义）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGemmSyrkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGemmSyrk”接口执行计算。

```cpp
aclnnStatus aclnnGemmSyrkGetWorkspaceSize(
  const aclTensor *a,
  aclTensor       *cRef,
  const aclScalar *alphaOptional,
  const aclScalar *betaOptional,
  bool            transposeX,
  const char      *fillMode,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGemmSyrk(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnGemmSyrkGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 111px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>a</td>
      <td>输入</td>
      <td>表示矩阵乘的输入矩阵，公式中的A。</td>
      <td><ul><li>数据类型需要与cRef一致（FLOAT16或BFLOAT16），参见<a href="#约束说明">约束说明</a>。</li>
      <li>transposeX为true时，a为转置的(…, k, m)存储（2-6维），m为a的最后一维。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2~6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>cRef</td>
      <td>输入&输出</td>
      <td>表示对称矩阵C，公式中的C。原地更新：结果直接写回该tensor的内存，输入输出为同一地址。</td>
      <td><ul><li>shape为(…, m, m)（2-6维，与a的batch轴一致），最后两维相等且等于a的m轴（transposeX为true时m为a的最后一维）。</li>
      <li>batch轴需要与a一致（原地更新不支持广播）。</li>
      <li>建议输入C为对称矩阵：$\beta \neq 0$时输出矩阵的对称性依赖输入C对称。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2~6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alphaOptional(α)</td>
      <td>输入</td>
      <td>表示公式中的α。</td>
      <td>为nullptr时默认取1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>betaOptional(β)</td>
      <td>输入</td>
      <td>表示公式中的β。</td>
      <td>为nullptr时默认取1.0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX</td>
      <td>输入</td>
      <td>是否按转置布局解读a。</td>
      <td>为true时a以(k, m)存储，计算$C = \alpha \times (A^T @ A) + \beta \times C$。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>fillMode</td>
      <td>输入</td>
      <td>输出区域模式。</td>
      <td>可选值为"full"（完整对称矩阵）/"up"（上三角）/"low"（下三角）；当前仅支持"full"，传"up"/"low"返回参数错误。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：
    - 仅支持FLOAT16、BFLOAT16数据类型；
    - fillMode当前仅支持"full"，"up"/"low"为原型预留值，尚未实现；
    - k轴为0时，接口自动路由为逐元素计算$C = \beta \times C$，不进入matmul计算路径。
  <!-- end id9 -->

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 257px">
  <col style="width: 121px">
  <col style="width: 431px">
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
      <td>传入的a或cRef是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>a和cRef的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>cRef不是方阵，或m轴与a不匹配，或batch轴与a不一致。</td>
    </tr>
    <tr>
      <td>fillMode不为"full"。</td>
    </tr>
    <tr>
      <td>当前设备不是Ascend 950PR/Ascend 950DT。</td>
    </tr>
  </tbody>
  </table>

## aclnnGemmSyrk

- **参数说明：**

  <div style="overflow-x: auto;">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGemmSyrkGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的stream。</td>
    </tr>
  </tbody>
  </table>
  </div>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：

  <!-- npu="950" id11 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：aclnnGemmSyrk默认确定性实现（每个输出tile由单条Mmad链按固定顺序累加，无原子操作与切K归约）。
  <!-- end id11 -->

- a与cRef的数据类型必须一致（FLOAT16或BFLOAT16），格式仅支持ND，维度为2~3维。
- cRef必须为方阵且m轴与a一致（transposeX为true时m为a的最后一维），batch轴与a一致（原地更新不支持广播）。
- k轴必须大于等于1；k轴为0时接口自动路由为逐元素计算$C = \beta \times C$。
- fillMode当前仅支持"full"；"up"/"low"为原型预留值，尚未实现。
- 建议输入C为对称矩阵：$\beta \neq 0$时输出矩阵的对称性依赖输入C对称。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gemm_syrk.h"

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

// 将FP16的uint16_t表示转换为float表示
float Fp16ToFloat(uint16_t h) {
  int s = (h >> 15) & 0x1;  // sign
  int e = (h >> 10) & 0x1F; // exponent
  int f = h & 0x3FF;        // fraction
  if (e == 0) {
    if (f == 0) {
      return s ? -0.0f : 0.0f;
    }
    float sig = f / 1024.0f;
    float result = sig * powf(2.0f, -24.0f);
    return s ? -result : result;
  } else if (e == 31) {
    return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
  }
  float result = (1.0f + f / 1024.0f) * powf(2.0f, static_cast<float>(e - 15));
  return s ? -result : result;
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
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // GemmSyrk: C = alpha * (A @ A^T) + beta * C，C输入输出同地址原地更新。
  // A为(128, 64)全1.0，C为(128, 128)全2.0（对称），alpha=beta=1.0时结果为 64 * 1.0 + 2.0 = 66.0。
  std::vector<int64_t> aShape = {128, 64};
  std::vector<int64_t> cShape = {128, 128};
  void* aDeviceAddr = nullptr;
  void* cDeviceAddr = nullptr;
  aclTensor* a = nullptr;
  aclTensor* cRef = nullptr;
  std::vector<uint16_t> aHostData(GetShapeSize(aShape), 0b0011110000000000); // fp16的1.0
  std::vector<uint16_t> cHostData(GetShapeSize(cShape), 0b0100000000000000); // fp16的2.0
  // 创建a aclTensor
  ret = CreateAclTensor(aHostData, aShape, &aDeviceAddr, aclDataType::ACL_FLOAT16, &a);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建cRef aclTensor（cRef既是输入也是输出，原地更新）
  ret = CreateAclTensor(cHostData, cShape, &cDeviceAddr, aclDataType::ACL_FLOAT16, &cRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float alphaValue = 1.0f;
  float betaValue = 1.0f;
  // 创建alpha/beta aclScalar，nullptr时默认取1.0
  aclScalar* alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  aclScalar* beta = aclCreateScalar(&betaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr && beta != nullptr, LOG_PRINT("aclCreateScalar failed\n"); return ACL_ERROR_INTERNAL_ERROR);
  bool transposeX = false;
  const char* fillMode = "full";

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnGemmSyrk第一段接口
  ret = aclnnGemmSyrkGetWorkspaceSize(a, cRef, alpha, beta, transposeX, fillMode, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrkGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnGemmSyrk第二段接口
  ret = aclnnGemmSyrk(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGemmSyrk failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(cShape);
  std::vector<uint16_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), cDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  // 期望结果全部为66.0（64个1.0的乘积和 + 2.0）
  for (int64_t i = 0; i < size; i++) {
    float diff = std::fabs(Fp16ToFloat(resultData[i]) - 66.0f);
    CHECK_RET(diff < 1e-3f, LOG_PRINT("result[%ld] is: %f, expected 66.0\n", i, Fp16ToFloat(resultData[i]));
              return ACL_ERROR_FAILURE);
  }
  LOG_PRINT("aclnnGemmSyrk in-place result check: PASS\n");

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(a);
  aclDestroyTensor(cRef);
  aclDestroyScalar(alpha);
  aclDestroyScalar(beta);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(aDeviceAddr);
  aclrtFree(cDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
