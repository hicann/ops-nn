# aclnnSwigluMxQuantWithDualAxis

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/swiglu_mx_quant_with_dual_axis)

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | × |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | × |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：融合算子，实现 SwiGLU 激活函数与双轴动态块量化的组合计算。先对输入计算 SwiGLU 激活函数，然后对结果同时在 -1 轴和 -2 轴进行基于块的动态量化，输出低精度的 FP4/FP8 张量和对应的缩放因子。

- 计算公式：

  **阶段 1：SwiGLU 激活函数**

  - 当 `activateLeft=true` 时：
    
    $$
    A_i = x_i \text{的前半部分}, B_i = x_i \text{的后半部分}
    $$

  - 当 `activateLeft=false` 时：
    
    $$
    A_i = x_i \text{的后半部分}, B_i = x_i \text{的前半部分}
    $$

  - SwiGLU 计算：
    
    $$
    swigluOut_i = Swish(A_i) * B_i = \frac{A_i}{1 + e^{-A_i}} * B_i
    $$

  **阶段 2：双轴动态块量化**

  - **-1 轴量化（列方向）**：将 SwiGLU 结果在 -1 轴上按照 32 个数进行分组，一组 32 个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale1 = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
    $$

  - **-2 轴量化（行方向）**：将 SwiGLU 结果在 -2 轴上按照 32 个数进行分组，一组 32 个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

    $$
    shared\_exp = floor(log_2(max_j(|V_j|))) - emax
    $$

    $$
    mxscale2 = 2^{shared\_exp}
    $$

    $$
    P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
    $$

  - -1 轴量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出 y1Out，mxscale1 按对应的 -1 轴维度上的分组组成输出 mxscale1Out。-2 轴量化后的 $P_{j}$ 按对应的 $V_{j}$ 的位置组成输出 y2Out，mxscale2 按对应的 -2 轴维度上的分组组成输出 mxscale2Out。

  - emax: 对应数据类型的最大正则数的指数位。

    |   DataType    | emax |
    | :-----------: | :--: |
    |  FLOAT4_E2M1  |  2   |
    |  FLOAT4_E1M2  |  0   |
    | FLOAT8_E4M3FN |  8   |
    |  FLOAT8_E5M2  |  15  |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSwigluMxQuantWithDualAxis"接口执行计算。

```cpp
aclnnStatus aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  int64_t          dstType,
  double           maxDtypeValue,
  const aclTensor *y1Out,
  const aclTensor *mxscale1Out,
  const aclTensor *y2Out,
  const aclTensor *mxscale2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnSwigluMxQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize

- **参数说明：**

  <table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
  <col style="width: 250px">
  <col style="width: 120px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>x (aclTensor*)</td>
      <td>输入</td>
      <td>输入张量，公式中的 x。</td>
      <td><ul><li>shape 为 [M, 2N]，最后一维必须为偶数。</li><li>不支持空 Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional (aclTensor*)</td>
      <td>输入</td>
      <td>分组索引，用于控制分组量化边界。</td>
      <td><ul><li>shape 为 [G]，采用 cumsum 模式，表示每个 group 的行数累积值。</li><li>当前不支持传入空Tensor</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>activateLeft (bool)</td>
      <td>输入</td>
      <td>SwiGLU 激活侧选择。</td>
      <td><ul><li>True 表示左半部分为 hidden，右半部分为 gate。</li><li>False 表示右半部分为 hidden，左半部分为 gate。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>输入</td>
      <td>表示数据转换的模式，对应公式中的 round_mode。</td>
      <td><ul><li>仅支持dstType为36，对应输出 y1Out 和 y2Out 数据类型为 FLOAT8_E4M3FN 时，仅支持 {"rint"}。</li><li>传入空指针时，采用 "rint" 模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>输入</td>
      <td>表示 mxscale1Out 和 mxscale2Out 的计算方法。</td>
      <td><ul><li>取值范围：{1}，代表 cuBLAS 实现。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>表示指定数据转换后 y1Out 和 y2Out 的类型。</td>
      <td><ul><li>输入范围为 {36}，分别对应输出 y1Out 和 y2Out 的数据类型为 {36: FLOAT8_E4M3FN}</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDtypeValue (double)</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td><ul><li>maxDtypeValue取值仅支持0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示 SwiGLU 结果量化 -1 轴后的对应结果，对应公式中的 <i>P<sub>i</sub></i>。</td>
      <td><ul><li>shape 为 [M, N]，即 SwiGLU 输出的一半。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示 -1 轴每个分组对应的量化尺度，对应公式中的 mxscale1。</td>
      <td><ul><li>shape 为 [M, (ceil(N/32)+2-1)/2, 2]，需进行偶数 pad，pad 填充值为 0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示 SwiGLU 结果量化 -2 轴后的对应结果，对应公式中的 <i>P<sub>j</sub></i>。</td>
      <td><ul><li>shape 为 [M, N]。</li></ul></td>
      <td>FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示 -2 轴每个分组对应的量化尺度，对应公式中的 mxscale2。</td>
      <td><ul><li>当 groupIndexOptional 存在时，shape 为 [floor(M/64)+G, N, 2]。</li><li>需进行偶数 pad，pad 填充值为 0。</li><li>mxscale2Out 输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
      <td>输出</td>
      <td>返回 op 执行器，包含了算子计算流程。</td>
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

  <table><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td>x、y1Out、mxscale1Out、y2Out 或 mxscale2Out 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x、roundModeOptional、dstType、scaleAlg、y1Out、mxscale1Out、y2Out、mxscale2Out 的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、y1Out、y2Out、mxscale1Out 或 mxscale2Out 的 shape 不满足校验条件。</td>
    </tr>
    <tr>
      <td>roundModeOptional、dstType、scaleAlg 不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>x 的最后一维不能被2整除。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnSwigluMxQuantWithDualAxis

- **参数说明：**

  <table><colgroup>
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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize 获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 输入 x 必须为 2 维张量，最后一维必须能被 2 整除（shape 为 [M, 2N]）。
- FP8 输出类型（FLOAT8_E4M3FN）仅支持 "rint" 舍入模式。
- groupIndexOptional采用 cumsum 模式，每个值表示对应 group 的行数累积值，groupIndexOptional的每个元素值需要大于0且最后一个元素值要等于M。
- 关于 mxscale1Out、mxscale2Out 的 shape 约束说明：
  - mxscale1Out.shape[-2] = (ceil(N/32) + 2 - 1) / 2。
  - mxscale1Out.shape[-1] = 2。
  - mxscale2Out.shape[-1] = 2。
  - 当 groupIndexOptional 存在时，mxscale2Out.shape[0] = floor(M/64) + G；当 groupIndexOptional 不存在时，mxscale2Out.shape[0] = ceil(M/64)。
  - 其他维度与 SwiGLU 输出一致。
- 默认支持饱和模式。
- 确定性说明：aclnnSwigluMxQuantWithDualAxis 默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnSwigluMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // 输入 x shape: [M, 2N] = [256, 1024]
    std::vector<int64_t> xShape = {256, 1024};
    // group_index shape: [G] = [1]，采用 cumsum 模式，值为 256 表示一个 group 包含所有 256 行
    std::vector<int64_t> groupIndexShape = {1};
    // SwiGLU 输出 shape: [M, N] = [256, 512]
    std::vector<int64_t> y1OutShape = {256, 512};
    std::vector<int64_t> y2OutShape = {256, 512};
    // mxscale1 shape: [M, ceil(N/32)/2 pad偶数, 2] = [256, 8, 2]
    std::vector<int64_t> mxscale1OutShape = {256, 8, 2};
    // mxscale2 shape: [floor(M/64) + G, N, 2] = [4 + 1, 512, 2] = [5, 512, 2]
    std::vector<int64_t> mxscale2OutShape = {5, 512, 2};

    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* mxscale1OutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* mxscale2OutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* mxscale1Out = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* mxscale2Out = nullptr;

    // 输入数据初始化（BF16）
    std::vector<uint16_t> xHostData(256 * 1024, 0);
    for (int64_t i = 0; i < 256 * 1024; i++) {
        xHostData[i] = static_cast<uint16_t>(i % 100);
    }
    // group_index 数据：cumsum 模式，值为 256 表示只有一个 group，包含所有行
    std::vector<int64_t> groupIndexHostData = {256};
    std::vector<uint8_t> y1OutHostData(256 * 512, 0);
    std::vector<uint8_t> y2OutHostData(256 * 512, 0);
    std::vector<uint8_t> mxscale1OutHostData(256 * 8 * 2, 0);
    std::vector<uint8_t> mxscale2OutHostData(5 * 512 * 2, 0);

    // 参数设置
    bool activateLeft = true;
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t dstType = 36;  // FLOAT8_E4M3FN
    int64_t scaleAlg = 1;  // cuBLAS
    double maxDtypeValue = 0.0;

    // 创建 x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 groupIndex aclTensor
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64, &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y1Out aclTensor
    ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 mxscale1Out aclTensor
    ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 y2Out aclTensor
    ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建 mxscale2Out aclTensor
    ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnSwigluMxQuantWithDualAxis第一段接口（带 groupIndex 输入）
    ret = aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, activateLeft, roundModeOptional,
        scaleAlg, dstType, maxDtypeValue, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnSwigluMxQuantWithDualAxis第二段接口
    ret = aclnnSwigluMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size1 = GetShapeSize(y1OutShape);
    auto size2 = GetShapeSize(y2OutShape);
    std::vector<uint8_t> y1OutData(size1, 0);
    std::vector<uint8_t> y2OutData(size2, 0);

    ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(uint8_t), y1OutDeviceAddr,
                      size1 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out from device to host failed. ERROR: %d\n", ret);
              return ret);
    ret = aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(uint8_t), y2OutDeviceAddr,
                      size2 * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2Out from device to host failed. ERROR: %d\n", ret);
              return ret);

    // 打印部分输出结果
    LOG_PRINT("y1Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size1; i++) {
        LOG_PRINT("y1Out[%ld] = %d\n", i, y1OutData[i]);
    }
    LOG_PRINT("y2Out first 10 elements:\n");
    for (int64_t i = 0; i < 10 && i < size2; i++) {
        LOG_PRINT("y2Out[%ld] = %d\n", i, y2OutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnSwigluMxQuantWithDualAxisTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantWithDualAxisTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
