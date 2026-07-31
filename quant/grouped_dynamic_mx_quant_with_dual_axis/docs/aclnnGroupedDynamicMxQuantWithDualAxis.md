# aclnnGroupedDynamicMxQuantWithDualAxis

[查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/grouped_dynamic_mx_quant_with_dual_axis)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：不支持
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

- 接口功能：对二维输入x进行分组动态MX双轴量化。算子在单次调用中基于groupIndex完成-1轴和-2轴量化，输出y1Out、y1ScaleOut、y2Out和y2ScaleOut。两个量化轴均依据groupIndex确定分组范围：在各分组内，-1轴按列方向进行MX量化，-2轴按行方向进行MX量化。
- 计算说明：量化以32个元素为一个MX block。scaleAlg = 1时，每个block生成FLOAT8_E8M0类型scale，并使用该scale将block内元素转换为dstDtype指定的数据类型。

- 计算公式：

  对一个MX block内的输入数据：

  $$
  D^{b}=[d^{i}]_{i=1}^{k},\quad k=32
  $$

  计算block内元素的最大绝对值：

  $$
  Amax(D^{b})=\max_i(|d^{i}|)
  $$

  当scaleAlg=1时，根据目标FP8类型最大可表示值计算FP32 scale：

  $$
  S_{fp32}^{b}=\frac{Amax(D^{b})}{Amax(dstDtype)}
  $$

  从块缩放因子$S_{fp32}^{b}$中提取无偏指数$E_{int}^{b}$和尾数$M_{fixp}^{b}$。

  为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

  $$
  E_{int}^{b} = \begin{cases}
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为正规数，且} E_{int}^{b} < 254 \text{且} M_{fixp}^{b} > 0 \\
    E_{int}^{b} + 1, & \text{如果} S_{fp32}^{b} \text{为非正规数，且} M_{fixp}^{b} > 0.5 \\
    E_{int}^{b}, & \text{否则}
  \end{cases}
  $$

  计算块缩放因子：

  $$
  S_{ue8m0}^{b}=2^{E_{int}^{b}}
  $$

  计算块转换因子：

  $$
  R_{fp32}^{b}=\frac{1}{fp32(S_{ue8m0}^{b})}
  $$

  对block内每个元素执行量化：

  $$
  p_i=cast\_to\_dst\_type(d^{i}\times R_{fp32}^{b},roundMode)
  $$

  输出该block对应的scale和量化结果：

  $$
  (S_{ue8m0}^{b},[p_i]_{i=1}^{k})
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize”接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用“aclnnGroupedDynamicMxQuantWithDualAxis”接口执行计算。

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndex,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  int64_t          dstDtype,
  double           maxDtypeValue,
  const aclTensor *y1Out,
  const aclTensor *y1ScaleOut,
  const aclTensor *y2Out,
  const aclTensor *y2ScaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicMxQuantWithDualAxis(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>待量化的输入Tensor，对应公式中的输入数据D。</td>
      <td><ul><li>shape为[M,N]。</li><li>N需要大于0且能被64整除。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndex（const aclTensor*）</td>
      <td>输入</td>
      <td>量化分组索引，采用cumsum形式描述各group边界。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[G]。</li><li>每个元素表示一个group的结束行索引，值需非负且非递减；最后一个元素需等于M。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional（char*）</td>
      <td>可选输入</td>
      <td>量化结果转换到目标输出y类型时使用的舍入模式。</td>
      <td><ul><li>可选输入，传入空指针时表示使用"rint"。</li><li>非空时仅支持取值"rint"。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg（int64_t）</td>
      <td>输入</td>
      <td>y1ScaleOut和y2ScaleOut的scale计算算法。</td>
      <td><ul><li>仅支持取值为1。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstDtype（int64_t）</td>
      <td>输入</td>
      <td>y1Out和y2Out的目标输出数据类型。</td>
      <td><ul><li>仅支持取值35和36，其中35表示FLOAT8_E5M2，36表示FLOAT8_E4M3FN。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxDtypeValue（double）</td>
      <td>输入</td>
      <td>预留参数。</td>
      <td><ul><li>仅支持取值0.0。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（const aclTensor*）</td>
      <td>输出</td>
      <td>表示x量化-1轴后的对应结果。</td>
      <td><ul><li>shape需要与x一致，为[M,N]。</li><li>数据类型由调用者预分配Tensor时指定，且需要与dstDtype一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y1ScaleOut（const aclTensor*）</td>
      <td>输出</td>
      <td>表示-1轴每个分组对应的量化尺度。</td>
      <td><ul><li>shape为[M,ceil(N/64),2]。</li><li>最后一维每2个scale成对存放；N方向按32个元素生成一个scale，并进行偶数pad，pad填充值为0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2Out（const aclTensor*）</td>
      <td>输出</td>
      <td>表示x量化-2轴后的对应结果。</td>
      <td><ul><li>shape需要与x一致，为[M,N]。</li><li>数据类型由调用者预分配Tensor时指定，且需要与dstDtype一致。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y2ScaleOut（const aclTensor*）</td>
      <td>输出</td>
      <td>表示-2轴每个分组对应的量化尺度。</td>
      <td><ul><li>shape为[floor(M/64)+G,N,2]，其中G为groupIndex元素个数。</li><li>倒数第二维方向按32个元素生成一个scale，并进行偶数pad。</li><li>y2ScaleOut输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
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
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1056px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>x、groupIndex、y1Out、y1ScaleOut、y2Out、y2ScaleOut、workspaceSize或executor为空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x、groupIndex、y1Out、y1ScaleOut、y2Out或y2ScaleOut的数据类型、数据格式或shape不在支持范围内。</td>
    </tr>
    <tr>
      <td>x不是2维ND Tensor，或x.shape[1]不大于0或不能被64整除。</td>
    </tr>
    <tr>
      <td>groupIndex为空Tensor、不是1维ND Tensor、元素值为负、元素值不满足非递减要求，或最后一个元素不等于x.shape[0]。</td>
    </tr>
    <tr>
      <td>roundModeOptional、scaleAlg、dstDtype或maxDtypeValue不在支持取值范围内。</td>
    </tr>
    <tr>
      <td>输出Tensor的shape与接口要求不一致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnGroupedDynamicMxQuantWithDualAxis

- **参数说明**

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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>Device侧workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>Device侧workspace大小，由第一段接口aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnGroupedDynamicMxQuantWithDualAxis默认确定性实现。
- N需要64对齐。
- y1Out和y2Out的shape需要与x一致，均为[M,N]；y1Out和y2Out的数据类型需要与dstDtype指定的目标类型一致。
- y1ScaleOut的shape由x决定，需要为[M,ceil(N/64),2]；y2ScaleOut的shape由x和groupIndex共同决定，需要为[floor(M/64)+G,N,2]，其中G为groupIndex元素个数。
- groupIndex采用cumsum模式，每个元素表示对应group的结束行索引；每个元素值需要非负且非递减，最后一个元素需要等于M（即x.shape[0]），M=0时最后一个元素为0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_grouped_dynamic_mx_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        if (i != 0 && shapeSize > INT64_MAX / i) {
            return -1;
        }
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    *stream = nullptr;
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        aclFinalize();
        return ret;
    }
    ret = aclrtCreateStream(stream);
    if (ret != ACL_SUCCESS) {
        aclrtResetDevice(deviceId);
        aclFinalize();
        return ret;
    }
    return ACL_SUCCESS;
}

class AclEnvironmentGuard {
public:
    AclEnvironmentGuard(int32_t deviceId, aclrtStream stream) : deviceId_(deviceId), stream_(stream) {}

    ~AclEnvironmentGuard()
    {
        aclrtDestroyStream(stream_);
        aclrtResetDevice(deviceId_);
        aclFinalize();
    }

private:
    int32_t deviceId_;
    aclrtStream stream_;
};

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    const int64_t shapeSize = GetShapeSize(shape);
    if (shapeSize <= 0 || hostData.size() != static_cast<size_t>(shapeSize) ||
        static_cast<uint64_t>(shapeSize) > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    const size_t size = static_cast<size_t>(shapeSize) * sizeof(T);
    *deviceAddr = nullptr;
    *tensor = nullptr;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ret;
    }

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    if (*tensor == nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

int CopyAndPrintOutput(const char* name, std::vector<uint8_t>& hostData, void* deviceAddr)
{
    auto ret = aclrtMemcpy(hostData.data(), hostData.size() * sizeof(uint8_t), deviceAddr,
                           hostData.size() * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    constexpr size_t PRINT_COUNT = 10;
    std::printf("%s first 10 elements:\n", name);
    for (size_t i = 0; i < PRINT_COUNT && i < hostData.size(); i++) {
        std::printf("%s[%zu] = %u\n", name, i, static_cast<unsigned int>(hostData[i]));
    }
    return ACL_SUCCESS;
}

int aclnnGroupedDynamicMxQuantWithDualAxisTest(int32_t deviceId)
{
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    AclEnvironmentGuard environmentGuard(deviceId, stream);

    // x shape: [M, N] = [256, 1024], N must be 64-aligned
    constexpr int64_t M = 256;
    constexpr int64_t N = 1024;
    constexpr int64_t BLOCK_SIZE = 32;
    constexpr int64_t SCALE_PAIR = 2;
    constexpr int64_t SCALE_AXIS_UNIT = BLOCK_SIZE * SCALE_PAIR;
    constexpr int64_t G = 4; // 4 groups, each has 64 rows

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> groupIndexShape = {G};
    std::vector<int64_t> y1ScaleOutShape = {M, N / SCALE_AXIS_UNIT, SCALE_PAIR};
    std::vector<int64_t> y2ScaleOutShape = {M / SCALE_AXIS_UNIT + G, N, SCALE_PAIR};

    // groupIndex: cumsum, last element must equal M
    std::vector<int64_t> groupIndexHostData = {64, 128, 192, 256};

    // x data: BF16 (uint16_t storage), fill with deterministic pattern
    std::vector<uint16_t> xHostData(M * N, 0);
    for (int64_t i = 0; i < M * N; i++) {
        xHostData[i] = static_cast<uint16_t>((i % 200) - 100);
    }

    // output buffers
    std::vector<uint8_t> y1OutHostData(M * N, 0);
    std::vector<uint8_t> y2OutHostData(M * N, 0);
    std::vector<uint8_t> y1ScaleOutHostData(GetShapeSize(y1ScaleOutShape), 0);
    std::vector<uint8_t> y2ScaleOutHostData(GetShapeSize(y2ScaleOutShape), 0);

    // attributes
    char* roundModeOptional = const_cast<char*>("rint");
    int64_t scaleAlg = 1;  // cuBLAS
    int64_t dstDtype = 36; // FLOAT8_E4M3FN
    double maxDtypeValue = 0.0;

    void* xDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1OutDeviceAddr = nullptr;
    void* y1ScaleOutDeviceAddr = nullptr;
    void* y2OutDeviceAddr = nullptr;
    void* y2ScaleOutDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupIndex = nullptr;
    aclTensor* y1Out = nullptr;
    aclTensor* y1ScaleOut = nullptr;
    aclTensor* y2Out = nullptr;
    aclTensor* y2ScaleOut = nullptr;

    // create x (BF16)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create groupIndex (INT64)
    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndex);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> groupIndexTensorPtr(groupIndex, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> groupIndexDeviceAddrPtr(groupIndexDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y1OutHostData, xShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y1ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y1ScaleOutHostData, y1ScaleOutShape, &y1ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y1ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1ScaleOutTensorPtr(y1ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y1ScaleOutDeviceAddrPtr(y1ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2Out (FLOAT8_E4M3FN)
    ret = CreateAclTensor(y2OutHostData, xShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // create y2ScaleOut (FLOAT8_E8M0)
    ret = CreateAclTensor(y2ScaleOutHostData, y2ScaleOutShape, &y2ScaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                          &y2ScaleOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2ScaleOutTensorPtr(y2ScaleOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> y2ScaleOutDeviceAddrPtr(y2ScaleOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // first-stage API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGroupedDynamicMxQuantWithDualAxisGetWorkspaceSize(x, groupIndex, roundModeOptional, scaleAlg, dstDtype,
                                                                 maxDtypeValue, y1Out, y1ScaleOut, y2Out, y2ScaleOut,
                                                                 &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // allocate workspace
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // second-stage API
    ret = aclnnGroupedDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // sync
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CopyAndPrintOutput("y1Out", y1OutHostData, y1OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y1ScaleOut", y1ScaleOutHostData, y1ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2Out", y2OutHostData, y2OutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CopyAndPrintOutput("y2ScaleOut", y2ScaleOutHostData, y2ScaleOutDeviceAddr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    return ACL_SUCCESS;
}

int main() { return aclnnGroupedDynamicMxQuantWithDualAxisTest(0); }

```
