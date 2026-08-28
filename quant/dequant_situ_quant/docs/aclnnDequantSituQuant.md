# aclnnDequantSituQuant

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：不支持
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

- 接口功能：在Situ激活函数前后添加dequant和quant操作，实现x的DequantSituQuant计算。

- 计算公式：

1. 根据输入数据类型x的不同，反量化路径不同：

   - INT8路径

     $$
     dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i + bias_i
     $$

   - INT32路径

     $$
     dequantOut_i = cast\_to\_float(x_i) \times weight\_scale_i \times activation\_scale_i + bias_i
     $$

   - BF16/FLOAT16路径（预反量化）

    $$
    dequantOut_i = cast\_to\_float(x_i)
    $$

2. Situ激活

   $$
   situ_a = \beta \times \tanh(gate / \beta) \times sigmoid(gate)
   $$

   当linear_beta > 0时：

   $$
   up = linear\_beta \times \tanh(up / linear\_beta)
   $$

   $$
   situOut = situ_a \times up
   $$

    其中，当activate_left为true时，gate取dequantOut的前半部分，up取后半部分；当activate_left为false时，gate取dequantOut的后半部分，up取前半部分。

3. 量化

   - quant_type = static

     $$
     y_i = trunc(situOut_i / quant\_scale_i + quant\_offset_i)
     $$

   - quant_type = dynamic

    $$
    scale_i = absmax(situOut_i) / 127
    $$

    $$
    y_i = trunc(situOut_i / scale_i)
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnDequantSituQuantGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnDequantSituQuant"接口执行计算。

```Cpp
aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor* x,
    const aclTensor* weightScaleOptional,
    const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional,
    const aclTensor* quantScaleOptional,
    const aclTensor* quantOffsetOptional,
    const aclTensor* groupIndexOptional,
    float             beta,
    float             linearBeta,
    bool              activateLeft,
    char*             quantTypeOptional,
    const aclTensor*  yOut,
    const aclTensor*  yScaleOut,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnDequantSituQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnDequantSituQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 320px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>公式中的输入x。</td>
      <td><ul><li>不支持空Tensor。</li><li>当x的数据类型为INT8时，x维度≥2维；当x的数据类型为INT32/BF16/FLOAT16时，x维度为2维。</li><li>最后一维需要是2的倍数。</li></ul></td>
      <td>INT8、INT32、BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>weightScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>反量化的weight scale。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(2H,)或(1,)。</li><li>INT8和INT32必选，BF16/FLOAT16不使用。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>activationScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>反量化的activation scale。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(1,)。</li><li>仅INT32必选，其他类型不使用。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>反量化的bias。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(2H,)或(1,)。</li><li>INT8和INT32可选，BF16/FLOAT16不使用。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantScaleOptional（aclTensor*）</td>
      <td>输入</td>
      <td>量化的scale。</td>
      <td><ul><li>不支持空Tensor。</li><li>`quantTypeOptional`为static时必选，shape为(H,)或(1,)。</li><li>`quantTypeOptional`为dynamic时可选，作为smoothScale使用。</li><li>仅INT8使用。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantOffsetOptional（aclTensor*）</td>
      <td>输入</td>
      <td>量化的offset。</td>
      <td><ul><li>不支持空Tensor。</li><li>当`quantTypeOptional`为static时必选，shape同`quantScaleOptional`。</li><li>当`quantTypeOptional`为dynamic时可选。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>MoE分组的group_index。</td>
      <td><ul><li>不支持空Tensor。</li><li>INT32/BF16/FLOAT16可选，INT8不使用。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>beta（float）</td>
      <td>输入</td>
      <td>Situ激活的beta参数。</td>
      <td>不能为0。默认4.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta（float）</td>
      <td>输入</td>
      <td>Situ激活的linear_beta参数。</td>
      <td>当值≤0时不启用。默认25.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>是否对输入的左半部分做Situ激活。</td>
      <td>当值为false时，对输入的右半部分做激活。默认true。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantTypeOptional（char*）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td>仅支持{"static", "dynamic"}。默认"dynamic"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化后的输出。</td>
      <td><ul><li>不支持空Tensor。</li><li>INT8: shape为x.shape[:-1]+[H]；其他: shape为[M, H]。</li></ul></td>
      <td>INT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>x</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>动态量化的scale。</td>
      <td><ul><li>不支持空Tensor。</li><li>INT8: shape为x.shape[:-1]；其他: shape为[M]。</li><li>`quantTypeOptional`为static时输出无意义值。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>x</td>
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

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed;width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
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
    <td rowspan="4">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="4">161001</td>
    <td>传入的x、yOut、yScaleOut是空指针。</td>
  </tr>
  <tr>
    <td>INT8时weightScaleOptional为空指针。</td>
  </tr>
  <tr>
    <td>INT32时weightScaleOptional或activationScaleOptional为空指针。</td>
  </tr>
  <tr>
    <td>`quantTypeOptional`为static时quantScaleOptional为空指针。</td>
  </tr>
  <tr>
    <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="6">161002</td>
    <td>x、yOut、yScaleOut为空tensor。</td>
  </tr>
  <tr>
    <td>x、weightScaleOptional等的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>quantTypeOptional不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x的最后一维不是2的倍数。</td>
  </tr>
  <tr>
    <td>beta为0。</td>
  </tr>
  <tr>
    <td>各输入约束不满足。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
</tbody></table>

## aclnnDequantSituQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDequantSituQuantGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnDequantSituQuant默认确定性实现。

- x的最后一维需要是2的倍数。
- 当x的数据类型为INT8时，x维度≥2维；当x的数据类型为INT32/BF16/FLOAT16时，x维度为2维。
- beta参数不能为0。
- INT8：必须提供`weightScaleOptional`，禁止`activationScaleOptional`和`groupIndexOptional`。
- INT32：必须提供`weightScaleOptional`和`activationScaleOptional`，禁止`quantScaleOptional`和`quantOffsetOptional`。
- BF16/FLOAT16：所有可选输入均不使用（预反量化模式）。
- 当`quantTypeOptional`为static时，`quantScaleOptional`必须提供。
- 当`quantTypeOptional`为dynamic时，`quantScaleOptional`可选（作为smoothScale使用）。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_situ_quant.h"

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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {16, 64};
    std::vector<int64_t> wsShape = {64};
    std::vector<int64_t> yShape = {16, 32};
    std::vector<int64_t> yScaleShape = {16};

    auto xSize = GetShapeSize(xShape);
    std::vector<int8_t> xHostData(xSize);
    for (int64_t i = 0; i < xSize; i++) {
        xHostData[i] = static_cast<int8_t>((i * 7 + 3) % 100);
    }
    std::vector<float> wsHostData(64, 0.1f);
    std::vector<int8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<float> yScaleHostData(GetShapeSize(yScaleShape), 0.0f);

    void* xDeviceAddr = nullptr;
    void* wsDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* ws = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_INT8, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(wsHostData, wsShape, &wsDeviceAddr, aclDataType::ACL_FLOAT, &ws);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT, &yScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    float beta = 4.0;
    float linearBeta = 25.0;
    bool activateLeft = true;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnDequantSituQuant第一段接口
    ret = aclnnDequantSituQuantGetWorkspaceSize(x, ws, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 beta, linearBeta, activateLeft,
                                                 const_cast<char*>("dynamic"),
                                                 y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnDequantSituQuant第二段接口
    ret = aclnnDequantSituQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSituQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto yResultSize = GetShapeSize(yShape);
    std::vector<int8_t> yResult(yResultSize, 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size() * sizeof(int8_t), yDeviceAddr,
                      yResultSize * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yResult from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < yResultSize; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yResult[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(ws);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(wsDeviceAddr);
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
