# aclnnQuantMaxV2

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

- 接口功能：

输入 `x` 支持按 `groupListOptional` 分组（不传 `groupListOptional` 时整个 tensor 视为单组，即 Per-Tensor），对每组求 `|x|` 的最大值 `amax`，并输出量化结果 `y`。 `amax` 为必选输出。传入 `xScaleOptional`时为静态量化；不传时为动态量化。静态量化不输出 `yScaleOutOptional`（传 `nullptr`）；动态量化时输出 `yScaleOutOptional`（shape 为 `[numGroups]`，Per-Tensor 为 `[1]`）。

相较于 `aclnnGroupedQuantMax`，V2接口支持"scale系数可选、grouplist分组可选、动态量化输出 `yScale`"三项能力，`amax` 升级为必选输出，并新增属性 `dstTypeMax`。

本算子支持静态量化与动态量化两种模式、Per-Group 与 Per-Tensor（不传 groupListOptional 时 Per-Group 退化为单组）两类实现，其组合关系如下：

<table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 180px">
  <col style="width: 180px">
  <col style="width: 240px">
  <col style="width: 500px">
  </colgroup>
  <thead>
    <tr>
      <th>xScaleOptional</th>
      <th>groupListOptional</th>
      <th>模式</th>
      <th>输出</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>不传（<code>nullptr</code>）</td>
      <td>不传（<code>nullptr</code>）</td>
      <td>Dynamic Per-Tensor</td>
      <td><code>y</code> + <code>amax[1]</code>（必选）+ <code>yScaleOutOptional[1]</code>（必选）</td>
    </tr>
    <tr>
      <td>传</td>
      <td>不传（<code>nullptr</code>）</td>
      <td>Static Per-Tensor</td>
      <td><code>y</code> + <code>amax[1]</code>（必选）+ <code>yScaleOutOptional</code>（不输出，传 <code>nullptr</code>）</td>
    </tr>
    <tr>
      <td>不传（<code>nullptr</code>）</td>
      <td>传</td>
      <td>Dynamic Per-Group</td>
      <td><code>y</code> + <code>amax[G]</code>（必选）+ <code>yScaleOutOptional[G]</code>（必选）</td>
    </tr>
    <tr>
      <td>传</td>
      <td>传</td>
      <td>Static Per-Group</td>
      <td><code>y</code> + <code>amax[G]</code>（必选）+ <code>yScaleOutOptional</code>（不输出，传 <code>nullptr</code>）</td>
    </tr>
  </tbody></table>

- 计算公式：

  `groupListOptional` 沿 dim-0 将 `x` 切分为 `numGroups`（记 `G`）组，`groupList[g]` 为前 `g+1` 组在 dim-0 上的累计结束位置（cumsum 语义）。对每个 group（`g = 0, 1, …, G-1`）：

  $$
  x_g = x\left[\,groupList[g-1] : groupList[g],\ \ldots\,\right] \qquad(\ g = 0\ \text{时下界为}\ 0\ )
  $$

  $$
  a_g = \max\left(|x_g|\right)
  $$

  **静态量化**（传入 `xScaleOptional`，`xScale[g] = 1/s` 为常规系数 `s` 的倒数）：
  $$
  q_g = xScale[g], \qquad y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **动态量化**（不传 `xScaleOptional`，`dstTypeMax` 为目标类型最大有限值）：

  $$
  q_g = \begin{cases}
  \dfrac{dstTypeMax}{a_g}, & a_g > 0 \\[4pt]
  0, & a_g = 0
  \end{cases}, \qquad
  y_g = \mathrm{Cast}\left(x_g \times q_g,\ dstType,\ roundMode\right)
  $$

  **输出**：

  $$
  amax[g] = a_g \quad(\text{必选输出}), \qquad
  y = \mathrm{Concat}\left(y_0,\ y_1,\ \ldots,\ y_{G-1}\right), \qquad
  yScale[g] = \dfrac{1}{q_g} \quad(\text{仅动态量化输出})
  $$

  其中，`yScaleOutOptional` 为动态量化下 `y` 的量化系数（`yScale[g] = 1/q_g`，即 `amax[g] / dstTypeMax`；`a_g = 0` 时 `q_g = 0`，对应取 `yScale[g] = 0`），满足 `x ≈ y × yScale`；静态量化不输出 `yScaleOutOptional`（传 `nullptr`）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnQuantMaxV2GetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnQuantMaxV2"接口执行计算。

```cpp
aclnnStatus aclnnQuantMaxV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *xScaleOptional,
  const aclTensor *groupListOptional,
  const char      *roundMode,
  int64_t          dstType,
  float            dstTypeMax,
  const aclTensor *yOut,
  const aclTensor *amaxOut,
  const aclTensor *yScaleOutOptional,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantMaxV2(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnQuantMaxV2GetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 350px">
    <col style="width: 250px">
    <col style="width: 100px">
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
        <td>表示待量化张量，对应公式中的 <code>x</code>。</td>
        <td><ul><li>支持空Tensor。</li></ul></td>
        <td>FLOAT32、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>xScaleOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>表示量化缩放因子 <code>s</code> 的倒数。</td>
        <td><ul><li><code>nullptr</code>：表示动态量化，量化系数按本组 <code>amax</code> 现算。</li></ul><ul><li>非<code>nullptr</code>：表示静态量化。</li></ul><ul><li>不支持空Tensor。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>groupListOptional（aclTensor*）</td>
        <td>可选输入</td>
        <td>表示输入x在0轴上每个group的偏移（cumsum模式），对应公式中group_list。</td>
        <td><ul><li>表示量化分组的累计结束位置，要求每个值在 <code>[0, x.shape[0]]</code> 范围内、非递减（允许相等，即空 group），且最后一个数必须与x的0轴大小相等；与V1一致，该数值约束由调用方保证，aclnn C接口与PyTorch接口（<code>cann_ops_nn.quant_max</code>）均不做校验（避免同步D2H开销），违反约束的结果未定义。</li></ul><ul><li><code>nullptr</code>：表示Per-Tensor（整个tensor视为单组）。</li></ul><ul><li>非<code>nullptr</code>：表示Per-Group。</li></ul><ul><li>shape为[num_groups]，num_groups取值范围为1-384。</li></ul><ul><li>不支持空Tensor。</li></ul></td>
        <td>INT64</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>roundMode（const char*）</td>
        <td>输入</td>
        <td>表示量化结果转换到目标类型时使用的舍入模式（必选属性）。</td>
        <td><ul><li><code>dstType=34</code> 支持 "round"、"hybrid"。</li></ul><ul><li><code>dstType=35/36</code> 仅支持 "rint"。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstType（int64_t）</td>
        <td>输入</td>
        <td>表示 <code>yOut</code> 的目标数据类型（必选属性）。</td>
        <td><ul><li>支持取值 <code>34</code> = HIFLOAT8、<code>35</code> = FLOAT8_E5M2、<code>36</code> = FLOAT8_E4M3FN。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>dstTypeMax（float）</td>
        <td>输入</td>
        <td>表示目标类型的最大有限值。</td>
        <td><ul><li>默认值 <code>15.0</code>，取值范围 <code>0 ~ 32768</code>。</li></ul><ul><li>仅 Dynamic 模式生效（Static 模式下不参与系数计算）。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOut（aclTensor*）</td>
        <td>输出</td>
        <td>表示量化结果，对应公式中的 <code>y</code>。</td>
        <td><ul><li>shape 与 <code>x</code> 完全一致。</li></ul><ul><li>dtype 由 <code>dstType</code> 决定。</li></ul></td>
        <td>HIFLOAT8、FLOAT8_E5M2、FLOAT8_E4M3FN</td>
        <td>ND（与 <code>x</code> 相同）</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>amaxOut（aclTensor*）</td>
        <td>输出</td>
        <td>表示量化前分组的 <code>x</code> 绝对值最大值。</td>
        <td><ul><li>必选输出。</li></ul><ul><li>shape 必须为 <code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）。</li></ul><ul><li>数据类型必须与 <code>x</code> 一致。</li></ul></td>
        <td>FLOAT32、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yScaleOutOptional（aclTensor*）</td>
        <td>可选输出</td>
        <td>表示 <code>y</code> 的量化系数（<code>yScale = 1/q</code>，动态量化下为 <code>amax/dstTypeMax</code>）。</td>
        <td><ul><li>仅动态量化（不传 <code>xScaleOptional</code>）时输出，shape 必须为 <code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）。</li></ul><ul><li>静态量化传 <code>nullptr</code>。</li></ul><ul><li>支持非连续视图（内部 Contiguous 计算 + ViewCopy 按原 strides 回写）。</li></ul></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
        <td><ul><li>-</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td><ul><li>-</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 300px">
    <col style="width: 150px">
    <col style="width: 550px">
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
        <td><code>x</code>、<code>yOut</code>、<code>amaxOut</code>、<code>roundMode</code> 存在空指针。</td>
      </tr>
      <tr>
        <td rowspan="12">ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><code>x</code> 或 <code>amaxOut</code> 的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td><code>xScaleOptional</code> 的数据类型非 FLOAT32、<code>groupListOptional</code> 的数据类型非 INT64。</td>
      </tr>
      <tr>
        <td><code>amaxOut</code> 为 <code>nullptr</code>，或 <code>amaxOut</code> 的 shape 不为 <code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）。</td>
      </tr>
      <tr>
        <td><code>amaxOut</code> 的数据类型与 <code>x</code> 不一致。</td>
      </tr>
      <tr>
        <td><code>yScaleOutOptional</code> 的数据类型非 FLOAT32。</td>
      </tr>
      <tr>
        <td>不传 <code>xScaleOptional</code>（Dynamic）但 <code>yScaleOutOptional</code> 为 <code>nullptr</code>，或 <code>yScaleOutOptional</code> 的 shape 不为 <code>[numGroups]</code>（Per-Tensor 为 <code>[1]</code>）。</td>
      </tr>
      <tr>
        <td>传入 <code>xScaleOptional</code>（Static）但 <code>yScaleOutOptional</code> 非 <code>nullptr</code>（静态量化不输出 <code>yScale</code>）。</td>
      </tr>
      <tr>
        <td><code>dstType</code> 不在 <code>{34, 35, 36}</code>，或 <code>roundMode</code> 与 <code>dstType</code> 不匹配。</td>
      </tr>
      <tr>
        <td><code>dstTypeMax</code> 超出 <code>0 ~ 32768</code>。</td>
      </tr>
      <tr>
        <td><code>x</code> 的维度数不在 2~8，或 <code>yOut</code> 的 shape 与 <code>x</code> 不一致。</td>
      </tr>
      <tr>
        <td><code>groupListOptional</code> 的维度数不为 1，或 <code>numGroups</code> 不在 <code>1 ~ 384</code>。</td>
      </tr>
      <tr>
        <td>同时传入 <code>xScaleOptional</code> 与 <code>groupListOptional</code> 时两者 <code>numGroups</code> 不一致。</td>
      </tr>
    </tbody></table>

## aclnnQuantMaxV2

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 700px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnQuantMaxV2GetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的AscendCL Stream流。</td>
      </tr>
    </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnQuantMaxV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_quant_max_v2.h"

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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnQuantMaxV2Test(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("Init acl success.\n");

    // 动态量化 + 分组场景：x shape [T, H] = [8, 4]，group_list = {4, 8} 表示两组各 4 行
    std::vector<int64_t> xShape = {8, 4};
    std::vector<int64_t> groupListShape = {2};
    std::vector<int64_t> amaxShape = {2};
    std::vector<int64_t> yScaleShape = {2};

    // x 的值选择可以覆盖多种情况
    std::vector<float> xHostData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f,
                                    0.5f, -0.5f, 0.0f, 448.0f, -448.0f, 0.125f, -0.125f, 2.5f};
    std::vector<int64_t> groupListHostData = {4, 8};

    // 输出 y 初始化为 0（dstType = 36 对应 FLOAT8_E4M3FN）
    std::vector<uint8_t> yHostData(GetShapeSize(xShape), 0);
    // amax 与 yScale 初始化为 0
    std::vector<float> amaxHostData(GetShapeSize(amaxShape), 0.0f);
    std::vector<float> yScaleHostData(GetShapeSize(yScaleShape), 0.0f);

    void* xDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* amaxDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* groupListTensor = nullptr;
    aclTensor* yTensor = nullptr;
    aclTensor* amaxTensor = nullptr;
    aclTensor* yScaleTensor = nullptr;

    // 创建输入 x tensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输入 group_list tensor（int64, shape [2]）
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT64,
                          &groupListTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupListTensorPtr(groupListTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupListDeviceAddrPtr(groupListDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 y tensor（FLOAT8_E4M3FN, shape 与 x 相同）
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 amax tensor（float, shape [2]）
    ret = CreateAclTensor(amaxHostData, amaxShape, &amaxDeviceAddr, aclDataType::ACL_FLOAT, &amaxTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> amaxTensorPtr(amaxTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> amaxDeviceAddrPtr(amaxDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 yScale tensor（float, shape [2]）
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yScaleTensorPtr(yScaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yScaleDeviceAddrPtr(yScaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用 aclnnQuantMaxV2 接口
    // 不传 xScale（nullptr）表示动态量化；静态量化时传入 xScale 并将 yScale 置为 nullptr
    const char* roundMode = "rint";
    int64_t dstType = 36; // FLOAT8_E4M3FN
    float dstTypeMax = 15.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 第一段接口：获取 workspace 大小
    ret = aclnnQuantMaxV2GetWorkspaceSize(xTensor, nullptr, groupListTensor, roundMode, dstType, dstTypeMax, yTensor,
                                          amaxTensor, yScaleTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantMaxV2GetWorkspaceSize success, workspaceSize: %lu\n", workspaceSize);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 第二段接口：执行计算
    ret = aclnnQuantMaxV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxV2 failed. ERROR: %d\n", ret); return ret);

    // 4. 同步等待
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("aclnnQuantMaxV2 execution success.\n");

    // 5. 取回结果
    auto ySize = GetShapeSize(xShape);
    ret = aclrtMemcpy(yHostData.data(), ySize * sizeof(uint8_t), yDeviceAddr, ySize * sizeof(uint8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(amaxHostData.data(), GetShapeSize(amaxShape) * sizeof(float), amaxDeviceAddr,
                      GetShapeSize(amaxShape) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy amax from device to host failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(yScaleHostData.data(), GetShapeSize(yScaleShape) * sizeof(float), yScaleDeviceAddr,
                      GetShapeSize(yScaleShape) * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yScale from device to host failed. ERROR: %d\n", ret); return ret);

    for (int64_t i = 0; i < ySize; i++) {
        LOG_PRINT("yOut[%ld] is: %d\n", i, yHostData[i]);
    }
    for (int64_t i = 0; i < GetShapeSize(amaxShape); i++) {
        LOG_PRINT("amaxOut[%ld] is: %f\n", i, amaxHostData[i]);
    }
    for (int64_t i = 0; i < GetShapeSize(yScaleShape); i++) {
        LOG_PRINT("yScaleOut[%ld] is: %f\n", i, yScaleHostData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnQuantMaxV2Test(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMaxV2Test failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    LOG_PRINT("All test cases passed!\n");
    return 0;
}
```
