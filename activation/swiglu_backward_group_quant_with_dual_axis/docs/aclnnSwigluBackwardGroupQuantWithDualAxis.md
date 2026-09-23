# aclnnSwigluBackwardGroupQuantWithDualAxis

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/9.2.0/activation/swiglu_backward_group_quant_with_dual_axis)

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

- 接口功能：融合算子，完成带可选Clamp和weight的SwiGLU反向计算，并对grad_x分别沿-1轴和-2轴进行CuBLAS算法的动态MX量化，输出FP8结果及E8M0缩放因子。

- 反向计算公式：

  将x沿最后一维切分为a和b：

  $$
  a = \mathbf{x}[:, 0:H], \qquad b = \mathbf{x}[:, H:2H]
  $$

  当clampLimit > 0时：

  $$
  a_c = \min(a, \text{clampLimit})
  $$

  $$
  b_c = \min(\max(b, -\text{clampLimit}), \text{clampLimit})
  $$

  当clampLimit=-1时，$a_c=a$、$b_c=b$。令：

  $$
  s = \sigma(\alpha a_c) = \frac{1}{1+\exp(-\alpha a_c)}
  $$

  如果不传入weight，$g=\mathbf{gradY}$；如果传入weight，$g=\mathbf{gradY}\times\mathbf{weight}[:,None]$。反向结果为：

  $$
  \mathbf{grad}_{a} =
  g \times (b_c+\text{bias}) \times s
  \times (1+\alpha a_c(1-s)) \times \text{mask}_{a}
  $$

  $$
  \mathbf{grad}_{b} =
  g \times a_c \times s \times \text{mask}_{b}
  $$

  其中，$\text{mask}_{a}=\mathbb{I}(a\leq\text{clampLimit})$，
  $\text{mask}_{b}=\mathbb{I}(-\text{clampLimit}\leq b\leq\text{clampLimit})$。
  未启用Clamp时两个mask均为1。最终：

  $$
  \mathbf{gradX} = \operatorname{concat}(\mathbf{grad}_{a}, \mathbf{grad}_{b})
  $$

- gradWeight计算公式（仅当同时传入weight和yOrigin时计算）：

  $$
  \mathbf{gradWeight}[t] =
  \sum_{h=0}^{H-1}\mathbf{gradY}[t,h]\times\mathbf{yOrigin}[t,h]
  $$

- 双轴动态MX量化：

  以32个元素为一个MX量化块，计算块内绝对值最大值对应的二次幂缩放因子：

  $$
  \text{scale} =
  \operatorname{ceil\_power\_of\_two}
  \left(\frac{\operatorname{amax}}{\text{fp8\_max}}\right)
  $$

  $$
  \text{output} =
  \operatorname{cast}_{\text{dstType}}
  \left(\frac{\mathbf{gradX}}{\text{scale}}\right)
  $$

  dstType=36时使用FLOAT8_E4M3FN的fp8_max，dstType=35时使用FLOAT8_E5M2的fp8_max；两种类型的量化流程一致，仅目标类型的最大值不同。

  - -1轴量化结果为y1Out和scale1Out。
  - -2轴量化结果为y2Out和scale2Out。
  - scale1Out和scale2Out的数据类型均为FLOAT8_E8M0。

- group场景的groupIndexOptional采用cumsum模式。设其元素为
  $\{t_0,t_1,\ldots,t_{G-1}\}$，必须满足：

  $$
  0 < t_0 < t_1 < \cdots < t_{G-1}=T
  $$

  第g组覆盖行区间$[t_{g-1},t_g)$，其中$t_{-1}=0$。groupIndexOptional仅用于-2轴量化分组边界，不改变SwiGLU反向计算的全部T行。

  group场景的scale2Out shape为[floor(T/64)+G, 2H, 2]。其中额外的分组行用于实现group边界偏移排布，不是普通的无效padding；读取scale2Out时必须遵循相同的group布局。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize接口获取计算所需workspace大小以及包含算子计算流程的执行器，再调用aclnnSwigluBackwardGroupQuantWithDualAxis接口执行计算。

```Cpp
aclnnStatus aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize(
    const aclTensor *gradY,
    const aclTensor *x,
    const aclTensor *weightOptional,
    const aclTensor *yOriginOptional,
    const aclTensor *groupIndexOptional,
    double           clampLimit,
    double           alpha,
    double           bias,
    int64_t          quantMode,
    int64_t          dstType,
    const aclTensor *y1Out,
    const aclTensor *scale1Out,
    const aclTensor *y2Out,
    const aclTensor *scale2Out,
    const aclTensor *gradWeightOutOptional,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```Cpp
aclnnStatus aclnnSwigluBackwardGroupQuantWithDualAxis(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize

- **参数说明：**

  <table><colgroup>
  <col style="width: 190px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 360px">
  <col style="width: 220px">
  <col style="width: 100px">
  <col style="width: 120px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU输出的反向梯度。</td>
      <td><ul><li>shape为[T, H]。</li><li>数据类型必须与x一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>SwiGLU前向输入。</td>
      <td><ul><li>shape为[T, 2H]。</li><li>最后一维必须为64的整数倍。</li><li>数据类型必须为FLOAT16或BFLOAT16。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>每个token对应的weight。</td>
      <td><ul><li>shape为[T]。</li><li>仅group场景支持传入。</li><li>传入weight时必须同时传入yOrigin。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>SwiGLU前向输出的原始值，用于计算gradWeight。</td>
      <td><ul><li>shape为[T, H]，必须与gradY一致。</li><li>数据类型必须与x一致。</li><li>必须与weight成对传入。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>分组量化边界。</td>
      <td><ul><li>shape为[G]，dtype为INT64。</li><li>采用cumsum模式，必须满足严格递增且最后一个值为T。</li><li>传空指针时表示非group场景。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>Clamp反向传播阈值。</td>
      <td><ul><li>取值为-1.0或大于0。</li><li>-1.0表示不启用Clamp。</li></ul></td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>SwiGLU变体的alpha系数。</td>
      <td>alpha系数，取值必须大于0。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bias（double）</td>
      <td>输入</td>
      <td>SwiGLU变体的bias系数。</td>
      <td>bias系数。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantMode（int64_t）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>当前仅支持1，表示双轴MX量化。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>FP8输出类型。</td>
      <td>35表示FLOAT8_E5M2，36表示FLOAT8_E4M3FN。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out（aclTensor*）</td>
      <td>输出</td>
      <td>-1轴量化结果。</td>
      <td>shape与x一致，数据类型由dstType决定。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale1Out（aclTensor*）</td>
      <td>输出</td>
      <td>-1轴MX缩放因子。</td>
      <td>shape为[T, ceil(2H/64), 2]，dtype为FLOAT8_E8M0。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out（aclTensor*）</td>
      <td>输出</td>
      <td>-2轴量化结果。</td>
      <td>shape与x一致，数据类型由dstType决定。</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale2Out（aclTensor*）</td>
      <td>输出</td>
      <td>-2轴MX缩放因子。</td>
      <td><ul><li>非group：shape为[ceil(T/64), 2H, 2]。</li><li>group：shape为[floor(T/64)+G, 2H, 2]，额外行用于group偏移布局。</li><li>dtype为FLOAT8_E8M0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional（aclTensor*）</td>
      <td>输出（可选）</td>
      <td>weight的梯度。</td>
      <td><ul><li>仅在weight和yOrigin同时传入时输出。</li><li>shape为[T]，dtype与weight一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回Device侧所需workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回包含算子执行流程的op执行器。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，错误包括：

  <table><colgroup>
  <col style="width: 260px">
  <col style="width: 130px">
  <col style="width: 760px">
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
      <td>必选输入、输出或workspace/executor指针为空。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入输出数据类型、数据格式、shape或属性值不满足约束。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在算子支持范围内。</td>
    </tr>
  </tbody>
  </table>

## aclnnSwigluBackwardGroupQuantWithDualAxis

- **参数说明：**

  <table><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 850px">
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
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>包含算子执行流程的op执行器。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- aclnnSwigluBackwardGroupQuantWithDualAxis默认确定性实现。
- aclnnSwigluBackwardGroupQuantWithDualAxis默认非Batch一致性实现，不支持通过aclrtSetSysParamOpt开启Batch一致性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_backward_group_quant_with_dual_axis.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
                             aclTensor** tensor, T value)
{
    int64_t shapeSize = GetShapeSize(shape);
    std::vector<T> hostData(shapeSize, value);
    return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> gradYShape = {128, 64};
    std::vector<int64_t> xShape = {128, 128};
    std::vector<int64_t> weightShape = {128};
    std::vector<int64_t> groupIndexShape = {2};
    std::vector<int64_t> y1Shape = {128, 128};
    std::vector<int64_t> scale1Shape = {128, 2, 2};
    std::vector<int64_t> scale2Shape = {4, 128, 2};

    void* gradYDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* yOriginDeviceAddr = nullptr;
    void* groupIndexDeviceAddr = nullptr;
    void* y1DeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* y2DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;

    aclTensor* gradYTensor = nullptr;
    aclTensor* xTensor = nullptr;
    aclTensor* weightTensor = nullptr;
    aclTensor* yOriginTensor = nullptr;
    aclTensor* groupIndexTensor = nullptr;
    aclTensor* y1Tensor = nullptr;
    aclTensor* scale1Tensor = nullptr;
    aclTensor* y2Tensor = nullptr;
    aclTensor* scale2Tensor = nullptr;
    aclTensor* gradWeightTensor = nullptr;

    int64_t gradYSize = GetShapeSize(gradYShape);
    std::vector<aclFloat16> gradYHostData(gradYSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < gradYSize; i++) {
        gradYHostData[i] = aclFloatToFloat16(static_cast<float>(i % 10) * 0.1f);
    }

    int64_t xSize = GetShapeSize(xShape);
    std::vector<aclFloat16> xHostData(xSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = aclFloatToFloat16(static_cast<float>((i % 20) - 10) * 0.5f);
    }

    int64_t weightSize = GetShapeSize(weightShape);
    std::vector<float> weightHostData(weightSize, 0.0f);
    for (int64_t i = 0; i < weightSize; i++) {
        weightHostData[i] = static_cast<float>((i % 5) + 1) * 0.2f;
    }

    int64_t yOriginSize = GetShapeSize(gradYShape);
    std::vector<aclFloat16> yOriginHostData(yOriginSize, aclFloatToFloat16(0.0f));
    for (int64_t i = 0; i < yOriginSize; i++) {
        yOriginHostData[i] = aclFloatToFloat16(static_cast<float>((i % 8) + 1) * 0.3f);
    }

    int64_t groupIndexSize = GetShapeSize(groupIndexShape);
    std::vector<int64_t> groupIndexHostData(groupIndexSize, 0);
    int64_t groupStride = gradYShape[0] / groupIndexSize;
    for (int64_t i = 0; i < groupIndexSize; i++) {
        groupIndexHostData[i] = (i + 1) * groupStride;
    }

    ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(yOriginHostData, gradYShape, &yOriginDeviceAddr, aclDataType::ACL_FLOAT16, &yOriginTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(groupIndexHostData, groupIndexShape, &groupIndexDeviceAddr, aclDataType::ACL_INT64,
                          &groupIndexTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(y1Shape, &y1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                            &scale1Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(y1Shape, &y2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<uint8_t>(scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0,
                                            &scale2Tensor, 0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorWithValue<float>(weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT,
                                          &gradWeightTensor, 0.0f);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double clampLimit = -1.0f;
    double alpha = 1.702f;
    double bias = 0.0f;
    int64_t quantMode = 1;
    int64_t dstType = 36;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize(
        gradYTensor, xTensor, weightTensor, yOriginTensor, groupIndexTensor, clampLimit, alpha, bias, quantMode,
        dstType, y1Tensor, scale1Tensor, y2Tensor, scale2Tensor, gradWeightTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnSwigluBackwardGroupQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnSwigluBackwardGroupQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluBackwardGroupQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    auto y1ResultSize = GetShapeSize(y1Shape);
    std::vector<uint8_t> y1ResultData(y1ResultSize, 0);
    ret = aclrtMemcpy(y1ResultData.data(), y1ResultData.size() * sizeof(uint8_t), y1DeviceAddr,
                      y1ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("y1 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < y1ResultSize; i++) {
        LOG_PRINT("y1[%ld] = %d\n", i, static_cast<int>(y1ResultData[i]));
    }

    auto scale1ResultSize = GetShapeSize(scale1Shape);
    std::vector<uint8_t> scale1ResultData(scale1ResultSize, 0);
    ret = aclrtMemcpy(scale1ResultData.data(), scale1ResultData.size() * sizeof(uint8_t), scale1DeviceAddr,
                      scale1ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale1 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("scale1 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < scale1ResultSize; i++) {
        LOG_PRINT("scale1[%ld] = %d\n", i, static_cast<int>(scale1ResultData[i]));
    }

    auto y2ResultSize = GetShapeSize(y1Shape);
    std::vector<uint8_t> y2ResultData(y2ResultSize, 0);
    ret = aclrtMemcpy(y2ResultData.data(), y2ResultData.size() * sizeof(uint8_t), y2DeviceAddr,
                      y2ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y2 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("y2 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < y2ResultSize; i++) {
        LOG_PRINT("y2[%ld] = %d\n", i, static_cast<int>(y2ResultData[i]));
    }

    auto scale2ResultSize = GetShapeSize(scale2Shape);
    std::vector<uint8_t> scale2ResultData(scale2ResultSize, 0);
    ret = aclrtMemcpy(scale2ResultData.data(), scale2ResultData.size() * sizeof(uint8_t), scale2DeviceAddr,
                      scale2ResultSize * sizeof(uint8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale2 result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("scale2 output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < scale2ResultSize; i++) {
        LOG_PRINT("scale2[%ld] = %d\n", i, static_cast<int>(scale2ResultData[i]));
    }

    auto gradWeightResultSize = GetShapeSize(weightShape);
    std::vector<float> gradWeightResultData(gradWeightResultSize, 0);
    ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(float), gradWeightDeviceAddr,
                      gradWeightResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradWeight result from device to host failed. ERROR: %d\n", ret);
              return ret);

    LOG_PRINT("gradWeight output (first 10 elements):\n");
    for (int64_t i = 0; i < 10 && i < gradWeightResultSize; i++) {
        LOG_PRINT("gradWeight[%ld] = %f\n", i, gradWeightResultData[i]);
    }

    aclDestroyTensor(gradYTensor);
    aclDestroyTensor(xTensor);
    aclDestroyTensor(weightTensor);
    aclDestroyTensor(yOriginTensor);
    aclDestroyTensor(groupIndexTensor);
    aclDestroyTensor(y1Tensor);
    aclDestroyTensor(scale1Tensor);
    aclDestroyTensor(y2Tensor);
    aclDestroyTensor(scale2Tensor);
    aclDestroyTensor(gradWeightTensor);

    aclrtFree(gradYDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(yOriginDeviceAddr);
    aclrtFree(groupIndexDeviceAddr);
    aclrtFree(y1DeviceAddr);
    aclrtFree(scale1DeviceAddr);
    aclrtFree(y2DeviceAddr);
    aclrtFree(scale2DeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
