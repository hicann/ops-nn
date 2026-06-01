# aclnnRmsNormQuantV3

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/rms_norm_quant_v3)

## 产品支持情况

| 产品 | 是否支持 |
| :---------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                                |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>                        |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                                         |    ×    |
| <term>Atlas 推理系列产品</term>                                                |    ×    |
| <term>Atlas 训练系列产品</term>                                                 |    ×    |

## 功能说明

- 接口功能：RmsNorm算子是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。RmsNormQuantV3算子将RmsNorm算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。同时在RmsNormQuantV2算子的基础上新增了Rstd的输出。
- 计算公式：

  $$
  quant\_in_i=\frac{x_i}{\operatorname{Rms}(\mathbf{x})} gamma_i + beta_i, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  - divMode为True时：
  
    $$
    y=round((quant\_in/scale)+offset)
    $$
  
  - divMode为False时：
  
    $$
    y=round((quant\_in*scale)+offset)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRmsNormQuantV3GetWorkspaceSize”接口获取入参并根据计算流程所需workspace大小，再调用“aclnnRmsNormQuantV3”接口执行计算。

```Cpp
aclnnStatus aclnnRmsNormQuantV3GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *gamma,
  const aclTensor *scale,
  const aclTensor *offsetOptional,
  const aclTensor *betaOptional,
  double           epsilon,
  bool             divMode,
  bool             outputRstd,
  aclTensor       *y,
  aclTensor       *rstd,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormQuantV3(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormQuantV3GetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 271px">
    <col style="width: 330px">
    <col style="width: 223px">
    <col style="width: 101px">
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
      <td>表示标准化过程中的源数据张量。对应公式中的`x`。</td>
      <td><ul><li>不支持空Tensor。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的`gamma`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型需要与`x`保持一致。</li><li>如果shape为1维，shape需要与`x`最后一维的维度保持一致。</li><li>如果shape为2维，则第一维必须为1，第二维需要与`x`最后一维的维度保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化过程中得到y进行的scale张量，对应公式中的`scale`。</td>
      <td><ul><li>不支持空Tensor。</li><li>维度为1。shape大小为1或者为x的最后一维的维度。</li><li>数据类型约束详见约束说明。</li><li>该参数的值不能为0。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示量化过程中得到y进行的offset张量，对应公式中的`offset`。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>shape需要与`scale`保持一致。</li><li>数据类型约束详见约束说明。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>betaOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示标准化过程中的偏移张量。对应公式中的`beta`。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>数据类型需要与`x`保持一致。</li><li>如果shape为1维，shape需要与`x`最后一维的维度保持一致。</li><li>如果shape为2维，则第一维必须为1，第二维需要与`x`最后一维的维度保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>公式中的输入`epsilon`，用于防止除0错误，数据类型为DOUBLE。</td>
      <td>建议传较小的正数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode（bool）</td>
      <td>输入</td>
      <td>公式中决定量化公式是否使用除法的参数，数据类型为bool。</td>
      <td>支持True和False。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputRstd（bool）</td>
      <td>输入</td>
      <td>表示指定是否输出有效的rstdOut。</td>
      <td><ul><li>支持True和False。</li><li>当outputRstd为False时，rstdOut为无效输出。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y（aclTensor*）</td>
      <td>输出</td>
      <td>表示最终量化输出Tensor，对应公式中的`y`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入`x`一致。</li></ul></td>
      <td>INT8、INT32、INT4、FLOAT8、HIF8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化后的标准差的倒数。对应公式中Rms(x)的倒数。</td>
      <td><ul><li>不支持空Tensor。</li><li>当outputRstd为True时，shape与入参x的shape前几维保持一致，前几维指x的维度减去gamma的维度，表示不需要norm的维度，rstdOut的-1轴是1。</li><li>当outputRstd为False时，该参数的最终输出无效。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
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
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入的x、gamma、scale、和y是空指针。</td>
    </tr>
    <tr>
      <td>当outputRstd为True时，传入rstd是空指针。</td>
    </tr>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>当输出yDtype为int32时，y的尾轴大小不等于x的尾轴大小的1/8，并且x的尾轴不能被8整除。</td>
    </tr>  
        <tr>
      <td>gamma不满足维度数为1-2维，scale不满足维度数为1维。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="7">561002</td>
      <td>输入或输出参数的维度数和数据类型不在范围之内。</td>
    </tr>
    <tr>
      <td>输入或输出参数的数据类型组合不在约束说明之内。</td>
    </tr>
    <tr>
      <td>输入x和输出y的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>当输出yDtype为int4时，x的尾轴大小不是偶数。</td>
    </tr>
    <tr>
      <td>gamma、beta(若存在)的shape不是完全相同的shape，或轴长不等于x的尾轴大小，或者类型不相同。</td>
    </tr>
    <tr>
      <td>gamma的维度和x的需要做norm的维度不相同，或rstdOut的维度和x的不需要norm的维度不相同，或rstdOut对应x归一化维度的轴长不为1。</td>
    </tr>
    <tr>
      <td>scale、offsetOptional(若存在)的shape不是完全相同的shape，或轴长不等于1或x的尾轴大小，或者类型不相同。</td>
    </tr>

  </tbody></table>

## aclnnRmsNormQuantV3

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRmsNormQuantV3GetWorkspaceSize获取。</td>
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
  </tbody>
  </table>

- **返回值**
  
  aclnnStatus：返回状态码。（具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)）

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：当`y`的数据类型为INT4时，`x`、`gamma`以及`beta`的最后一维必须为偶数。
- <term>Ascend 950PR/Ascend 950DT</term>：当`y`的数据类型为INT32时，`y`的最后一维必须是`x`最后一维的1/8。
- 各产品型号支持数据类型说明：

  - <term>Ascend 950PR/Ascend 950DT</term>：

    | x数据类型 | gamma数据类型 | scale数据类型 | offsetOptional数据类型 | betaOptional数据类型 |epsilon数据类型 | y数据类型 | rstd数据类型 |
    | --------- | ------------- |  ------------- | -------------- |------------- | --------- |--------- |--------- |
    | FLOAT16   | FLOAT16       |  FLOAT16       | INT8           |FLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | INT8           | BFLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT16       | FLOAT16           | FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  BFLOAT16       | BFLOAT16           |BFLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT32   | FLOAT32       |  FLOAT32       | FLOAT32           |FLOAT32       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | INT32           |FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32      | INT32           |BFLOAT16       | DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | FLOAT16   | FLOAT16       |  FLOAT32       | FLOAT32           | FLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8      | FLOAT32       |
    | BFLOAT16   | BFLOAT16       |  FLOAT32       | FLOAT32           | BFLOAT16       |DOUBLE      |INT8、INT4、FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8     | FLOAT32       |

- 确定性计算：
  - aclnnRmsNormQuantV3默认确定性实现。
  
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_quant_v3.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

// 检查当前硬件是否支持
bool CheckHardwareSupport()
{
    const char* socName = aclrtGetSocName();
    if (socName == nullptr) {
        LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
        return true;
    }

    LOG_PRINT("Current SOC: %s\n", socName);

    // 该算子仅支持Ascend950
    if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
        return true;
    }

    LOG_PRINT("Warning: This operator only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
              socName);
    return false;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
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

int main()
{
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 检查硬件支持
    if (!CheckHardwareSupport()) {
        LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
        Finalize(deviceId, stream);
        return 0;
    }

    // 2. 构造输入与输出
    // x: [2, 64] FP16, gamma: [64] FP16, scale: [1] FP16, offset: [1] INT8, beta: [64] FP16
    // y: [2, 64] INT8, rstd: [2, 1] FLOAT32
    std::vector<int64_t> x_shape = {2, 64};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scale_shape = {1};
    std::vector<int64_t> offset_shape = {1};
    std::vector<int64_t> beta_shape = {64};
    std::vector<int64_t> y_shape = {2, 64};
    std::vector<int64_t> rstd_shape = {2, 1};

    void* x_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scale_device_addr = nullptr;
    void* offset_device_addr = nullptr;
    void* beta_device_addr = nullptr;
    void* y_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* rstd = nullptr;

    // 初始化输入数据
    // x: FP16 0.5 = 0x3800
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3800);
    // gamma: FP16 1.5 = 0x3e00
    std::vector<uint16_t> gamma_host_data(GetShapeSize(gamma_shape), 0x3e00);
    // scale: FP16 1.0 = 0x3C00
    std::vector<uint16_t> scale_host_data(GetShapeSize(scale_shape), 0x3C00);
    // offset: INT8 0
    std::vector<int8_t> offset_host_data(GetShapeSize(offset_shape), 0);
    // beta: FP16 0.0 = 0x0000
    std::vector<uint16_t> beta_host_data(GetShapeSize(beta_shape), 0);
    // 输出初始化
    std::vector<int8_t> y_host_data(GetShapeSize(y_shape), 0);
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 0.0f);

    double epsilon = 1e-6;
    bool divMode = true;
    bool outputRstd = true;

    LOG_PRINT("Input x shape: [2, 64], dtype: FP16\n");
    LOG_PRINT("Gamma shape: [64], dtype: FP16\n");
    LOG_PRINT("Scale shape: [1], dtype: FP16\n");
    LOG_PRINT("Offset shape: [1], dtype: INT8\n");
    LOG_PRINT("Beta shape: [64], dtype: FP16\n");
    LOG_PRINT("Output y shape: [2, 64], dtype: INT8\n");
    LOG_PRINT("Output rstd shape: [2, 1], dtype: FLOAT32\n");
    LOG_PRINT("epsilon: %e, divMode: %s, outputRstd: %s\n",
              epsilon, divMode ? "true" : "false", outputRstd ? "true" : "false");

    // 创建x aclTensor (FP16)
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建gamma aclTensor (FP16)
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT16, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaTensorPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDeviceAddrPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建scale aclTensor (FP16)
    ret = CreateAclTensor(scale_host_data, scale_shape, &scale_device_addr, aclDataType::ACL_FLOAT16, &scale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scale_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scale failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建offset aclTensor (INT8)
    ret = CreateAclTensor(offset_host_data, offset_shape, &offset_device_addr, aclDataType::ACL_INT8, &offset);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetTensorPtr(offset, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetDeviceAddrPtr(offset_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offset failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建beta aclTensor (FP16)
    ret = CreateAclTensor(beta_host_data, beta_shape, &beta_device_addr, aclDataType::ACL_FLOAT16, &beta);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> betaTensorPtr(beta, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> betaDeviceAddrPtr(beta_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor beta failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建y aclTensor (INT8输出)
    ret = CreateAclTensor(y_host_data, y_shape, &y_device_addr, aclDataType::ACL_INT8, &y);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(y_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor y failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建rstd aclTensor (FLOAT32输出)
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdTensorPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDeviceAddrPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. 调用CANN算子库API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnRmsNormQuantV3第一段接口
    LOG_PRINT("Calling aclnnRmsNormQuantV3GetWorkspaceSize...\n");
    ret = aclnnRmsNormQuantV3GetWorkspaceSize(
        x, gamma, scale, offset, beta, epsilon, divMode, outputRstd,
        y, rstd, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // 根据第一段接口计算出的workspaceSize申请device内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // 调用aclnnRmsNormQuantV3第二段接口
    LOG_PRINT("Calling aclnnRmsNormQuantV3...\n");
    ret = aclnnRmsNormQuantV3(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormQuantV3 failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    {
        // 拷贝y结果 (INT8)
        auto size = GetShapeSize(y_shape);
        std::vector<int8_t> y_result(size, 0);
        ret = aclrtMemcpy(y_result.data(), y_result.size() * sizeof(y_result[0]),
                          yDeviceAddrPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy y from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output y (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  y[%ld] = %d\n", i, y_result[i]);
        }

        // 拷贝rstd结果 (FLOAT32)
        size = GetShapeSize(rstd_shape);
        std::vector<float> rstd_result(size, 0.0f);
        ret = aclrtMemcpy(rstd_result.data(), rstd_result.size() * sizeof(rstd_result[0]),
                          rstdDeviceAddrPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy rstd from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output rstd values:\n");
        for (int64_t i = 0; i < size; i++) {
            LOG_PRINT("  rstd[%ld] = %f\n", i, rstd_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormQuantV3 Test PASSED ===\n");

    // 6. 资源自动释放
    Finalize(deviceId, stream);

    return 0;
}
```
