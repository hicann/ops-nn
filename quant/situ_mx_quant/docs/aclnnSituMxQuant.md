# aclnnSituMxQuant

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

- 接口功能：将Situ激活函数与动态MX（Microscaling）量化融合为一个算子，对输入的数据x进行Situ激活后，对激活的结果进行MX量化，输出量化后的结果和scale。

- 计算公式：

1. Situ激活

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

    其中，当activate_left为true时，gate取x的前半部分，up取后半部分；当activate_left为false时，gate取x的后半部分，up取前半部分。

2. MX量化（OCP算法）

$$
shared\_exp = floor(log2(max(|situOut_i|))) - emax
$$

$$
y\_scale = 2^{shared\_exp}  (E8M0)
$$

$$
y = cast\_to\_fp8(situOut / y\_scale)
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用"aclnnSituMxQuantGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSituMxQuant"接口执行计算。

```Cpp
aclnnStatus aclnnSituMxQuantGetWorkspaceSize(
    const aclTensor*  x,
    double            beta,
    double            linearBeta,
    bool              activateLeft,
    int64_t           axis,
    int64_t           dstType,
    char*             roundModeOptional,
    const aclTensor*  yOut,
    const aclTensor*  yScaleOut,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnSituMxQuant(
    void*            workspace,
    uint64_t         workspaceSize,
    aclOpExecutor*   executor,
    aclrtStream      stream)
```

## aclnnSituMxQuantGetWorkspaceSize

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
      <td><ul><li>不支持空Tensor。</li><li>shape支持1-8维。</li><li>最后一维需要是2的倍数。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（double）</td>
      <td>输入</td>
      <td>Situ激活的beta参数。</td>
      <td>必须大于0。默认1.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>linearBeta（double）</td>
      <td>输入</td>
      <td>Situ激活的linear_beta参数。</td>
      <td>当值≤0时不启用。默认0.0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>是否对输入的左半部分做Situ激活。</td>
      <td>当值为false时，对输入的右半部分做激活。默认false。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>量化轴。</td>
      <td>当前仅支持-1。默认-1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>输出y的数据类型。</td>
      <td>输入范围为{35, 36}，分别对应{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。默认36。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional（char*）</td>
      <td>输入</td>
      <td>量化舍入模式。</td>
      <td>支持{"rint", "round", "floor"}。FP8输出仅支持"rint"。默认"rint"。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>量化后的输出，公式中的y。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为x.shape[:-1]+[H]，其中H=x.shape[-1]/2。</li></ul></td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>MX量化的scale（E8M0格式）。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为x.shape[:-1]+[ceil(H/64), 2]，其中H=x.shape[-1]/2。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-9</td>
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
    <td>ACLNN_ERR_PARAM_NULLPTR</td>
    <td>161001</td>
    <td>传入的x、yOut、yScaleOut是空指针。</td>
  </tr>
  <tr>
    <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
    <td rowspan="10">161002</td>
    <td>x、yOut、yScaleOut为空tensor。</td>
  </tr>
  <tr>
    <td>x的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>axis不为-1。</td>
  </tr>
  <tr>
    <td>dstType不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>roundModeOptional不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x的最后一维不是2的倍数。</td>
  </tr>
  <tr>
    <td>beta不大于0。</td>
  </tr>
  <tr>
    <td>yOut的数据类型与dstType不匹配，或yScaleOut的数据类型不为FLOAT8_E8M0。</td>
  </tr>
  <tr>
    <td>yOut或yScaleOut的shape与推导结果不一致。</td>
  </tr>
  <tr>
    <td>x、yOut、yScaleOut的数据格式不为ND。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_RUNTIME_ERROR</td>
    <td>361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
</tbody></table>

## aclnnSituMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSituMxQuantGetWorkspaceSize获取。</td>
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
  - aclnnSituMxQuant默认确定性实现。

- x的最后一维需要是2的倍数。
- x的维数必须大于等于1维。
- axis当前仅支持-1（尾轴量化）。
- dstType支持36（FLOAT8_E4M3FN）或35（FLOAT8_E5M2）。
- roundModeOptional必须为"rint"。
- yOut的数据类型必须与dstType匹配，yScaleOut的数据类型必须为FLOAT8_E8M0。
- yOut、yScaleOut的shape需要与推导结果一致（见参数说明）。
- 关于yScaleOut的shape约束说明如下：
  - H = x.shape[-1] / 2
  - scaleNum = ceil(H / 64)
  - yScaleOut.shape = x.shape[:-1] + [scaleNum, 2]

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_situ_mx_quant.h"

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

uint16_t FloatToBf16Bits(float val)
{
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7FFF + lsb;
    return static_cast<uint16_t>(bits >> 16);
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
    std::vector<int64_t> xShape = {4, 128};
    std::vector<int64_t> yShape = {4, 64};
    std::vector<int64_t> yScaleShape = {4, 1, 2};

    // 生成BF16输入数据
    std::vector<uint16_t> xHostData(GetShapeSize(xShape));
    for (int64_t i = 0; i < GetShapeSize(xShape); i++) {
        float val = ((i * 17 + 3) % 200 - 100) / 50.0f;
        xHostData[i] = FloatToBf16Bits(val);
    }
    std::vector<uint8_t> yHostData(GetShapeSize(yShape), 0);
    std::vector<uint8_t> yScaleHostData(GetShapeSize(yScaleShape), 0);

    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* yScaleDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* yScale = nullptr;

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yScaleHostData, yScaleShape, &yScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &yScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    double beta = 1.0;
    double linearBeta = 0.0;
    bool activateLeft = false;
    int64_t axis = -1;
    int64_t dstType = 36;

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnSituMxQuant第一段接口
    ret = aclnnSituMxQuantGetWorkspaceSize(x, beta, linearBeta, activateLeft, axis, dstType,
                                           const_cast<char*>("rint"),
                                           y, yScale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnSituMxQuant第二段接口
    ret = aclnnSituMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSituMxQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto yResultSize = GetShapeSize(yShape);
    std::vector<uint8_t> yResult(yResultSize, 0);
    ret = aclrtMemcpy(yResult.data(), yResult.size(), yDeviceAddr,
                      yResultSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yResult from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < yResultSize; i++) {
        LOG_PRINT("y[%ld] is: %d\n", i, yResult[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(yScale);

    // 7. 释放device资源，需要根据具体API的接口定义修改
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
