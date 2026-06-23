# aclnnSwigluGroupQuant

[📄 查看源码](https://gitcode.com/shilulu/ops-nn/tree/9.1.0/activation/swiglu_group_quant)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

**支持说明**：
- 本算子支持 quant_mode=0/1/2/3/4 五种量化模式，本文档聚焦 **quant_mode=3（HiFp8 Dynamic Quant，HiF8 动态量化）**。
- quant_mode=3 仅在 **Ascend 950PR/Ascend 950DT** 芯片支持。

## 功能说明

- 接口功能：SwigluGroupQuant 算子实现 SwiGLU 激活函数与分组动态量化（quant_mode=3，HiF8 Dynamic Quant）融合计算。将输入 x 沿最后一维切分为两半 x0、x1，对 x0 施加 SiLU 激活后与 x1 相乘，再根据可选的 weight 加权，最终对加权结果动态计算缩放因子并量化为 hifloat8，输出量化结果 y、缩放因子 y_scale 以及可选的原始激活值 y_origin。
- 算子支持范围：支持 MoE 场景（传入 group_index）和非 MoE 场景（group_index 传空），支持可选的 Clamp 处理、Weight 加权、output_origin 输出。
- 计算流程：
  - 步骤〇：GroupIndex 处理（可选）→ 计算 real_bs
  - 步骤一：输入切分（将 x 切分为 x0 和 x1）
  - 步骤二：Clamp 处理（可选，仅处理前 real_bs 行）
  - 步骤三：SwiGLU 激活（仅处理前 real_bs 行）
  - 步骤四：Weight 加权（可选，仅处理前 real_bs 行）
  - 步骤五：HiF8 动态量化（仅处理前 real_bs 行）

### 计算公式

- **GroupIndex 处理（可选）**：当提供 `group_index` 时，仅处理前 real_bs 行，$\text{real\_bs} = \min(\sum_{g=0}^{G-1} \text{group\_index}[g], N)$，其中 $G$ 为 MoE 专家分组数，$N$ 为输入第一维。

- **输入切分**：$x_0 = x[:, :H], \quad x_1 = x[:, H:]$

- **Clamp 处理（可选，clamp_limit > 0 时）**：$x_0 = \min(x_0, c), \quad x_1 = \text{clip}(x_1, -c, c)$，其中 $c$ 为 `clamp_limit`。

- **SwiGLU 激活**：$y = \text{silu}(x_0) \times x_1 = \frac{x_0}{1 + e^{-x_0}} \times x_1$

- **Weight 加权（可选）**：$y = y \times w$，其中 $w$ 为 weight；未提供时跳过。

- **HiF8 动态量化（quant_mode=3）**：
  - 非分组（无 group_index）：$\text{amax} = \max(\text{absmax}(y), \epsilon), \quad s = \frac{\text{amax}}{M_{\text{hif8}}}, \quad y_{\text{quant}} = \text{hif8\_cast}(y / s)$，y_scale shape=[1]
  - 分组（有 group_index）：对每个 group $g$ 独立计算 $\text{amax}_g = \max(\text{absmax}(y_g), \epsilon), \quad s_g = \frac{\text{amax}_g}{M_{\text{hif8}}}, \quad y_{\text{quant},g} = \text{hif8\_cast}(y_g / s_g)$，y_scale shape=[G]

  其中 $M_{\text{hif8}}$ 为 `dst_type_max_finite`（默认 448.0），$\epsilon$ 为数值稳定性常数，`hif8_cast` 为 HiFloat8 类型转换函数。

- **output_origin（可选）**：当 `output_origin=true` 时，额外输出 SwiGLU 加权后的原始激活值 y_origin（未量化），dtype 与 x 一致，shape 为 [N, H]。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSwigluGroupQuantGetWorkspaceSize"接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用"aclnnSwigluGroupQuant"接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantGetWorkspaceSize(
    const aclTensor   *x,
    const aclTensor   *weightOptional,
    const aclTensor   *groupIndexOptional,
    const aclTensor   *scaleOptional,
    int64_t            dstType,
    int64_t            quantMode,
    int64_t            blockSize,
    bool               roundScale,
    float              clampLimit,
    float              dstTypeMaxFinite,
    bool               outputOrigin,
    const aclTensor   *y,
    const aclTensor   *yScale,
    const aclTensor   *yOriginOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSwigluGroupQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
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
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入张量，SwiGLU 输入数据。</td>
      <td><ul><li>shape=[N, 2H] 或 [B, S, 2H]，最后一维必须为偶数。</li><li>N 为 token 数，2H 为输入最后一维。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>MoE 权重张量，用于加权计算。</td>
      <td><ul><li>shape=[N, 1] 或 [B, S, 1]，需与 x 的第一维一致。</li><li>未提供时跳过加权。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>GroupIndex 张量，动态核分配。</td>
      <td><ul><li>shape=[G]，dtype=INT64。</li><li>G 为 MoE 专家分组数。</li><li>未提供时使用全局量化，y_scale shape=[1]。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>静态量化输入的 scale 张量（仅 quant_mode=2 使用）。</td>
      <td><ul><li>quant_mode=3 时该参数不使用，可传空。</li><li>quant_mode=2 时，group_index 存在则 shape=[G]，否则 shape=[1]。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>输出数据类型。</td>
      <td><ul><li>quant_mode=3 时固定为 HiFloat8（27），该参数不生效。</li><li>quant_mode=0/1 时指定 FP8 类型。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>本文档场景取值 3（HiFp8 Dynamic Quant）。</li><li>完整取值范围 {0, 1, 2, 3, 4}。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>分组大小（可选，默认 0 自动选择）。</td>
      <td><ul><li>quant_mode=3 时不使用，默认 0。</li><li>quant_mode=0 只支持 0 或 128。</li><li>quant_mode=1 只支持 0 或 32。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundScale（bool）</td>
      <td>输入</td>
      <td>Scale 取整优化（仅 quant_mode=0 有效）。</td>
      <td><ul><li>quant_mode=3 时忽略。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（float）</td>
      <td>输入</td>
      <td>Clamp 阈值（可选）。</td>
      <td><ul><li>取值范围 ≥ 0.0。</li><li>clampLimit=0 表示不启用 Clamp。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMaxFinite（float）</td>
      <td>输入</td>
      <td>HiFloat8 类型的最大有限值，输出类型放缩系数。</td>
      <td><ul><li>quant_mode=3 时用于计算 scale = amax / dstTypeMaxFinite。</li><li>默认 448.0。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否输出 yOrigin。</td>
      <td><ul><li>true 时输出 SwiGLU 原始激活值 y_origin。</li><li>false 时 yOriginOptional 可传空。</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>量化输出张量。</td>
      <td><ul><li>shape=[N, H] 或 [B, S, H]，H = 2H/2。</li></ul></td>
      <td>HIFLOAT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yScale（aclTensor*）</td>
      <td>输出</td>
      <td>缩放因子张量。</td>
      <td><ul><li>无 group_index 时 shape=[1]。</li><li>有 group_index 时 shape=[G]。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOptional（aclTensor*）</td>
      <td>输出（可选）</td>
      <td>SwiGLU 原始激活值。</td>
      <td><ul><li>outputOrigin=true 时输出，shape=[N, H]，dtype 与 x 一致。</li><li>outputOrigin=false 时可传空。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
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
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>必选参数 x/y/yScale 为 nullptr。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>161002</td>
      <td>x、weight 等输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>多个输入 tensor 之间的 shape 信息不匹配、输入属性不在取值范围（详见参数说明）。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 173px">
  <col style="width: 124px">
  <col style="width: 852px">
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
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnSwigluGroupQuantGetWorkspaceSize 获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - 当提供 `group_index` 参数时：
    - **前 real_bs 行**：保证计算结果确定性（相同输入 → 相同输出）
    - **后 N-real_bs 行**：不保证确定性，输出值可能因硬件、运行次数而异
    - **应用场景**：MoE 中只有前 real_bs 个 token 属于当前专家，后续行数据无实际意义
  - 当未提供 `group_index` 参数时（real_bs=N）：所有行数据保证计算结果确定性

- 输入 shape 约束：
  - x 最后一维必须为偶数（$2H$）
  - y 最后一维为 $H$，与 x 最后一维的一半对应
  - x 最后一维上限 ≤ 8192

- 量化轴约束：
  - quant_mode=3 只支持 -1 轴量化（最后一维），无 axis 属性参数

- 可选参数约束：
  - weight 的 shape 需与 x 的第一维一致
  - group_index 为 1 维 INT64 张量，shape=[G]
  - scale 仅 quant_mode=2 使用，quant_mode=3 时传空

- 数据类型约束：
  - x 支持 FLOAT、FLOAT16、BFLOAT16
  - weight 必须为 FLOAT 类型
  - group_index 必须为 INT64 类型
  - y 为 HIFLOAT8 类型
  - yScale 为 FLOAT 类型
  - yOrigin 与 x 数据类型一致

- Clamp 约束：
  - clamp_limit 必须 ≥ 0.0
  - clamp_limit=0 表示不启用 Clamp

- 属性有效性约束：
  - quant_mode=3 时，dst_type 固定 HiFloat8，block_size、round_scale 不生效
  - output_origin 所有模式均支持

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
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

template <typename T>
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor, T value) {
  int64_t shapeSize = GetShapeSize(shape);
  std::vector<T> hostData(shapeSize, value);
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 非分组模式 (quant_mode=3, HiF8 Dynamic Quant), float32 输入
  std::vector<int64_t> xShape = {128, 2048};
  std::vector<int64_t> yShape = {128, 1024};
  std::vector<int64_t> yScaleShape = {1};

  void* xDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* yScaleDeviceAddr = nullptr;

  aclTensor* xTensor = nullptr;
  aclTensor* yTensor = nullptr;
  aclTensor* yScaleTensor = nullptr;

  int64_t xSize = GetShapeSize(xShape);
  std::vector<float> xHostData(xSize, 1.0f);
  for (int64_t i = 0; i < xSize; i++) {
    xHostData[i] = static_cast<float>((i % 20) - 10) * 0.5f;
  }

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // y 为 HIFLOAT8，按 uint8 存储
  ret = CreateAclTensorWithValue<uint8_t>(yShape, &yDeviceAddr, aclDataType::ACL_HIFLOAT8, &yTensor, 0);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensorWithValue<float>(yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScaleTensor, 0.0f);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 属性参数
  int64_t dstType = 27;            // HiFloat8
  int64_t quantMode = 3;           // HiFp8 Dynamic Quant
  int64_t blockSize = 0;
  bool roundScale = false;
  float clampLimit = 0.0f;
  float dstTypeMaxFinite = 448.0f;
  bool outputOrigin = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // 可选输入：weight / groupIndex / scale 均传空（非分组动态量化）
  ret = aclnnSwigluGroupQuantGetWorkspaceSize(
      xTensor, nullptr, nullptr, nullptr,
      dstType, quantMode, blockSize, roundScale, clampLimit, dstTypeMaxFinite, outputOrigin,
      yTensor, yScaleTensor, nullptr,
      &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSwigluGroupQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuant failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 拷贝 y_scale 结果到 Host
  std::vector<float> yScaleResultData(GetShapeSize(yScaleShape), 0);
  ret = aclrtMemcpy(yScaleResultData.data(), yScaleResultData.size() * sizeof(float),
                    yScaleDeviceAddr, GetShapeSize(yScaleShape) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yScale result failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("yScale output:\n");
  for (int64_t i = 0; i < GetShapeSize(yScaleShape); i++) {
    LOG_PRINT("  yScale[%ld] = %f\n", i, yScaleResultData[i]);
  }

  aclDestroyTensor(xTensor);
  aclDestroyTensor(yTensor);
  aclDestroyTensor(yScaleTensor);

  aclrtFree(xDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(yScaleDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
