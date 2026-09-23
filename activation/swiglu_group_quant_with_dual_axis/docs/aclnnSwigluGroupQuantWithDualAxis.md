# aclnnSwigluGroupQuantWithDualAxis

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/9.2.0/activation/swiglu_group_quant_with_dual_axis)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

### 接口功能

融合Clipped-SwiGLU、可选逐行weight乘法以及**两个方向**的MX FP8量化，一次调用输出五路结果：

- `y1Out`、`y1ScaleOut`：沿最后一维（-1轴）量化，每个token行内每32个元素共享一个尺度。
- `y2Out`、`y2ScaleOut`：沿行方向（-2轴）量化；提供`groupIndexOptional`时按分组边界、
  组内每32行且每列共享一个尺度；不提供时按全体行成块。
- `yOriginOut`：可选的乘weight前激活结果。

`groupIndexOptional`使用**cumsum（累积组端索引）**语义；`weightOptional`仅允许在group场景使用。
第一路的量化数值与单轴算子`aclnnSwigluGroupQuant`（`quantMode=5`）逐字节一致。

### 计算公式

Clipped-SwiGLU激活可统一写为：

$$
A' =
\begin{cases}
\min(A, L), & L > 0 \\
A, & L = -1
\end{cases}, \qquad
B' =
\begin{cases}
\min(\max(B, -L), L), & L > 0 \\
B, & L = -1
\end{cases}
$$

$$
F = A' \cdot \sigma(\alpha \cdot A') \cdot (B' + \beta)
$$

其中 $A = x[:, :H]$、$B = x[:, H:]$、$H = D/2$，$L$ 为`clampLimit`，$\alpha$ 为`alpha`，$\beta$ 为`bias`；$L=-1$ 表示关闭Clamp。

<details>
<summary><strong>阶段一：Clipped-SwiGLU激活</strong></summary>

1. **切分**（仅沿末维 -1轴前后切分）：$A = x[:,\ :H]$，$B = x[:,\ H:]$
2. **Clamp**（在SiLU之前，`clampLimit > 0`时生效）：
   $A' = \min(A, L)$，即仅对门控分支做上界截断；$B' = \min(\max(B, -L), L)$，即对线性分支做对称截断。
   `clampLimit = -1.0`时 $A'=A$、$B'=B$。
3. **变体SwiGLU**：$F = A' \cdot \sigma(\alpha \cdot A') \cdot (B' + \beta)$。
   当`alpha=1.0`、`bias=0.0`且`clampLimit=-1.0`时退化为标准SwiGLU。
4. **逐token加权**（`weightOptional`非空时）：$u = R_T(R_T(F) \cdot w[t])$；
   `weight`缺省时 $u = R_T(F)$。$R_T(\cdot)$ 表示舍入到输入dtype，$u$ 即量化输入。

</details>

<details>
<summary><strong>阶段二：两路MX FP8量化（CuBALS，块大小32）</strong></summary>

两路均采用CuBALS（`scaleAlg=1`）尺度算法，块大小固定为32。块内绝对值最大值
$amax$ 映射到目标类型可表示范围 $s^{raw} = amax / Amax(DType)$（E4M3FN为448，E5M2为57344）。

对于有限非零的 $s^{raw}$，将其按FP32位模式解释。记 $E_b$ 为FP32的**8 bit偏置指数域**，
$m \in [0,1)$ 为尾数域对应的小数部分，则CuBALS的指数取整规则为：

$$
E_b^{*} =
\begin{cases}
E_b + 1, & 0 < E_b < 254 \text{ 且 } m > 0 \\
E_b + 1, & E_b = 0 \text{ 且 } m > 0.5 \\
E_b, & \text{否则}
\end{cases}
$$

E8M0直接保存该偏置指数编码：

$$
mxScale = E_b^{*}, \qquad s = 2^{E_b^{*}-127}
$$

块内数据 $d = Q8(u / s)$，FP32→FP8采用**RINT（就近舍入）**；全零块`mxScale = 0`。
未使用的scale padding/预留位置不属于有效结果，调用方不应依赖其值。

- **第一路（-1轴）**：`[T, H]`内每行独立、行内每32元素成块。
- **第二路（-2轴）**：group场景按`groupIndexOptional`累积组端索引分组，组内每32行成块、
  每列独立尺度、块不跨组；组末不足32行时仅在尺度计算的临时块内按0补齐。non-group场景按全体行成块。

</details>

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnSwigluGroupQuantWithDualAxisGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluGroupQuantWithDualAxis”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightOptional,
  const aclTensor *groupIndexOptional,
  int64_t          dstType,
  int64_t          quantMode,
  double           clampLimit,
  bool             outputOrigin,
  double           alpha,
  double           bias,
  const aclTensor *y1Out,
  const aclTensor *y1ScaleOut,
  const aclTensor *y2Out,
  const aclTensor *y2ScaleOut,
  const aclTensor *yOriginOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluGroupQuantWithDualAxisGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 240px">
  <col style="width: 110px">
  <col style="width: 180px">
  <col style="width: 460px">
  <col style="width: 170px">
  <col style="width: 90px">
  <col style="width: 130px">
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
      <td><ul><li>shape为[T, D]（D = 2H）。</li><li>D须为偶数、<b>64元素对齐</b>且H ≥ 32（即D ≥ 64）。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>逐token权重，用于Clipped-SwiGLU输出的加权计算。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li><b>仅当提供groupIndexOptional时允许</b>，non-group场景传非空将被拒绝。</li><li>元素个数需等于合轴行数T。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td><b>累积组端索引（cumsum语义，元素值本身即组端行索引）</b>，仅用于第二路分组量化。</td>
      <td><ul><li>可选参数，不支持空Tensor。</li><li>NULL缺省 = 退化为non-group，第二路按全体行成块。</li><li>不为空时须INT64、1维 [G]，元素 ≥ 0且<b>非递减</b>、末元素等于T。</li><li>相邻元素相等表示空组，不产块。</li><li><b>不承担输出截断</b>：激活与第一路始终处理全部T行。</li><li>元素值在设备侧校验，详见“约束说明”。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>目标量化类型。</td>
      <td><ul><li>支持 <b>36</b>（FLOAT8_E4M3FN，默认）与 <b>35</b>（FLOAT8_E5M2）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li><b>仅支持1</b>（双轴MX FP8）。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>Clipped-SwiGLU的clamp门限（公式中的L）。</td>
      <td><ul><li>-1.0表示不启用clamp。</li><li>启用时必须为有限正数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputOrigin（bool）</td>
      <td>输入</td>
      <td>是否额外输出乘weight前的激活y_origin。</td>
      <td><ul><li>true时yOriginOut为 [T, H] 的量化前激活（供训练反向缓存中间结果）。</li><li>false时yOriginOut为 [0] 占位，内容无效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>变体SwiGLU的silu缩放系数。</td>
      <td><ul><li>必须为有限正数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（double）</td>
      <td>输入</td>
      <td>变体SwiGLU的线性分支偏置。</td>
      <td><ul><li>必须为有限数。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>第一路（-1轴）量化输出。</td>
      <td><ul><li>shape为[T, H]。</li><li>数据类型需与dstType一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y1ScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>第一路量化尺度。</td>
      <td><ul><li>shape为[T, ceil(ceil(H/32)/2), 2]。</li><li>最后一维最多存放相邻的2个32-element block scale；未使用的padding位置无有效值保证，调用方不应依赖其内容。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>第二路（-2轴）量化输出。</td>
      <td><ul><li>shape为[T, H]。</li><li>数据类型需与dstType一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2ScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>第二路量化尺度。</td>
      <td><ul><li>shape为 [floor(T/64) + G, H, 2]（group）或 [ceil(T/64), H, 2]（non-group），G为groupIndexOptional的元素个数。</li><li>最后一维slot0/slot1最多对应两个相邻的32-row block scale；group场景可能包含为分组边界保留的物理位置，未使用的预留槽位无有效值保证，调用方不应依赖其内容。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOut（aclTensor*）</td>
      <td>输出</td>
      <td>乘weight前的激活结果（含clamp/alpha/bias效果）。</td>
      <td><ul><li>outputOrigin为true时shape为 [T, H]，数据类型与x一致；false时为 [0] 占位。</li><li>不支持空指针。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2（outputOrigin=true）/ 1（outputOrigin=false）</td>
      <td>√</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>x、y1Out、y1ScaleOut、y2Out、y2ScaleOut、yOriginOut、workspaceSize或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>输入或输出的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的shape不满足约束（例如x非二维、D未按64对齐）。</td>
    </tr>
    <tr>
      <td>dstType或quantMode不符合支持的值。</td>
    </tr>
    <tr>
      <td>clampLimit、alpha或bias取值非法（非有限、alpha非正等）。</td>
    </tr>
    <tr>
      <td>weightOptional与groupIndexOptional的组合不合法（non-group传weight）。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>tiling阶段失败，例如weight元素数与T不匹配、groupIndexOptional维数或元素个数非法。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuantWithDualAxis

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantWithDualAxisGetWorkspaceSize获取。</td>
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

- `x`仅支持二维`[T, 2H]`、FLOAT16或BFLOAT16，非空，`2H >= 64`且能被64整除。
- `quantMode`仅支持`1`；`dstType`仅支持FLOAT8_E4M3FN（默认）与FLOAT8_E5M2。
- `weightOptional`仅在提供`groupIndexOptional`时允许，元素数必须等于`T`；non-group场景传`weight`将被拒绝。
- `groupIndexOptional`的元素值在**设备执行时**校验：非法值（负值、递减、越界、末元素不等于`T`）
  触发设备执行异常；异步调用可能在后续流同步时才报告失败，失败后所有输出均不可使用。
  `GetWorkspaceSize`阶段只校验元数据（dtype、维数、元素个数非零），不保证元素值合法。
- `outputOrigin`为true/false均支持：true时`yOriginOut`为`[T, H]`，false时为`[0]`占位。
- 第一路的量化数值与单轴算子`aclnnSwigluGroupQuant`（`quantMode=5`）逐字节一致。
- 默认支持确定性计算；仅支持前向。
- `x`、`weightOptional`、`groupIndexOptional`均支持非连续Tensor。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_swiglu_group_quant_with_dual_axis.h"

namespace SwigluExample {
inline int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (int64_t dim : shape) {
        if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) {
            throw std::invalid_argument("tensor shape must be positive and fit int64");
        }
        size *= dim;
    }
    return size;
}

// The MX V2 path and the dual-axis operator only ship an Ascend950 kernel, so
// running their aclnn calls on another SOC fails inside tiling. Report a skip
// instead of a false failure. An unavailable SOC name keeps the legacy
// behaviour of running the example.
inline bool CheckHardwareSupport(const char* operatorName)
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        std::cout << "Warning: cannot get SOC name, skip hardware check" << std::endl;
        return true;
    }
    std::cout << "Current SOC: " << socName << std::endl;
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }
    std::cout << "Warning: " << operatorName << " only supports Ascend950, current SOC '" << socName
              << "' is not supported. Skip test." << std::endl;
    return false;
}

// Owns every acquired resource until stream work has finished. API void pointers
// stay inside this wrapper; callers receive only borrowed tensor descriptors.
class Session {
public:
    explicit Session(int& status) : status_(status) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    ~Session()
    {
        if (pending_) {
            Record(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        }
        if (executor_ != nullptr) {
            Record(aclDestroyAclOpExecutor(executor_), "aclDestroyAclOpExecutor");
        }
        if (workspace_ != nullptr) {
            Record(aclrtFree(workspace_), "aclrtFree(workspace)");
        }
        for (auto& tensor : tensors_) {
            if (tensor->descriptor != nullptr) {
                Record(aclDestroyTensor(tensor->descriptor), "aclDestroyTensor");
            }
            if (tensor->device != nullptr) {
                Record(aclrtFree(tensor->device), "aclrtFree(tensor)");
            }
        }
        if (stream_ != nullptr) {
            Record(aclrtDestroyStream(stream_), "aclrtDestroyStream");
        }
        if (deviceSet_) {
            Record(aclrtResetDevice(deviceId_), "aclrtResetDevice");
        }
        if (initialized_) {
            Record(aclFinalize(), "aclFinalize");
        }
    }

    void Check(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            Record(result, operation);
            throw std::runtime_error(operation);
        }
    }

    void Init(int32_t deviceId)
    {
        deviceId_ = deviceId;
        Check(aclInit(nullptr), "aclInit");
        initialized_ = true;
        Check(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_), "aclrtCreateStream");
    }

    template <typename T>
    aclTensor* CreateTensor(const std::vector<T>& host, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        const int64_t elements = Numel(shape);
        if (static_cast<uint64_t>(elements) != host.size() ||
            host.size() > std::numeric_limits<size_t>::max() / sizeof(T) || aclDataTypeSize(dtype) != sizeof(T)) {
            throw std::invalid_argument("tensor shape, storage or dtype size mismatch");
        }
        const size_t bytes = host.size() * sizeof(T);
        std::vector<int64_t> strides(shape.size(), 1);
        // Numel validated every positive dimension and the total product above.
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        tensors_.push_back(std::make_unique<TensorResource>());
        auto& tensor = *tensors_.back();
        Check(aclrtMalloc(&tensor.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(tensor)");
        Check(aclrtMemcpy(tensor.device, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy");
        tensor.descriptor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                            shape.data(), shape.size(), tensor.device);
        Check(tensor.descriptor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS, "aclCreateTensor");
        return tensor.descriptor;
    }

    aclOpExecutor** ExecutorAddress() { return &executor_; }

    aclOpExecutor* PrepareExecutor()
    {
        // Keep ownership explicit; a repeatable executor is destroyed by Session.
        Check(aclSetAclOpExecutorRepeatable(executor_), "aclSetAclOpExecutorRepeatable");
        return executor_;
    }

    void* AllocateWorkspace(uint64_t bytes)
    {
        if (bytes > 0) {
            Check(aclrtMalloc(&workspace_, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc(workspace)");
        }
        return workspace_;
    }

    aclrtStream StreamForLaunch()
    {
        // Even a failed launch may have submitted work; synchronize before freeing.
        pending_ = true;
        return stream_;
    }

    void Synchronize()
    {
        Check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");
        pending_ = false;
    }

private:
    struct TensorResource {
        void* device = nullptr;
        aclTensor* descriptor = nullptr;
    };
    void Record(int result, const char* operation)
    {
        if (result != ACL_SUCCESS) {
            std::cerr << operation << " failed: " << result << std::endl;
            if (status_ == ACL_SUCCESS) {
                status_ = result;
            }
        }
    }
    int& status_;
    int32_t deviceId_ = 0;
    bool initialized_ = false;
    bool deviceSet_ = false;
    bool pending_ = false;
    aclrtStream stream_ = nullptr;
    aclOpExecutor* executor_ = nullptr;
    void* workspace_ = nullptr;
    std::vector<std::unique_ptr<TensorResource>> tensors_;
};
} // namespace SwigluExample

using SwigluExample::CheckHardwareSupport;
using SwigluExample::Numel;

int main()
{
    int status = ACL_SUCCESS;
    {
        SwigluExample::Session session(status);
        try {
            session.Init(0);
            if (!CheckHardwareSupport("SwigluGroupQuantWithDualAxis")) {
                std::cout << "\n=== Test SKIPPED (hardware not supported) ===" << std::endl;
                return ACL_SUCCESS;
            }
            const std::vector<int64_t> xShape{64, 128};
            const std::vector<int64_t> yShape{64, 64};
            const std::vector<int64_t> scale1Shape{64, 1, 2};
            const std::vector<int64_t> scale2Shape{1, 64, 2};
            std::vector<uint16_t> xHost(Numel(xShape), 0x3C00);
            std::vector<uint8_t> yHost(Numel(yShape), 0);
            std::vector<uint8_t> scale1Host(Numel(scale1Shape), 0);
            std::vector<uint8_t> scale2Host(Numel(scale2Shape), 0);
            std::vector<uint16_t> originHost(Numel(yShape), 0);

            aclTensor* x = session.CreateTensor(xHost, xShape, ACL_FLOAT16);
            aclTensor* y1 = session.CreateTensor(yHost, yShape, ACL_FLOAT8_E4M3FN);
            aclTensor* s1 = session.CreateTensor(scale1Host, scale1Shape, ACL_FLOAT8_E8M0);
            aclTensor* y2 = session.CreateTensor(yHost, yShape, ACL_FLOAT8_E4M3FN);
            aclTensor* s2 = session.CreateTensor(scale2Host, scale2Shape, ACL_FLOAT8_E8M0);
            aclTensor* origin = session.CreateTensor(originHost, yShape, ACL_FLOAT16);

            uint64_t workspaceSize = 0;
            auto ret = aclnnSwigluGroupQuantWithDualAxisGetWorkspaceSize(x, nullptr, nullptr, ACL_FLOAT8_E4M3FN, 1, 7.0,
                                                                         true, 1.702, 1.0, y1, s1, y2, s2, origin,
                                                                         &workspaceSize, session.ExecutorAddress());
            session.Check(ret, "aclnnSwigluGroupQuantWithDualAxis");
            aclOpExecutor* executor = session.PrepareExecutor();
            void* workspace = session.AllocateWorkspace(workspaceSize);
            ret = aclnnSwigluGroupQuantWithDualAxis(workspace, workspaceSize, executor, session.StreamForLaunch());
            session.Check(ret, "aclnnSwigluGroupQuantWithDualAxis");
            session.Synchronize();

        } catch (const std::exception& error) {
            std::cerr << error.what() << std::endl;
            if (status == ACL_SUCCESS) {
                status = ACL_ERROR_INVALID_PARAM;
            }
        }
    }
    if (status != ACL_SUCCESS) {
        return 1;
    }
    std::cout << "aclnnSwigluGroupQuantWithDualAxis example succeeded" << std::endl;
    return 0;
}
```
