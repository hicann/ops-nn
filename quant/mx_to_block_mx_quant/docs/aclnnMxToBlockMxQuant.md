# aclnnMxToBlockMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/mx_to_block_mx_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：将调用`aclnnDynamicMxQuantV2`量化得到的FLOAT4的Tensor结合FLOAT8_E8M0缩放系数，转换为FLOAT8分块量化格式，同时输出-1轴和-2轴方向的量化尺度。

- 计算公式：

  $$
  \begin{aligned}
  scale_{fp8} &= max(mxscale_{fp4\_block}) / MAX\_OFFSET \\
  offset &= mxscale_{fp4\_block} / scale_{fp8} \\
  x_{fp8} &= x_{fp4} \times offset
  \end{aligned}
  $$

  - 其中$mxscale_{fp4\_block}$是输入mxscale提供的FP8_E8M0缩放系数；$MAX\_OFFSET$是输入和输出数据类型之间的最大偏移量；$x_{fp4}$是量化得到的FLOAT4张量；$x_{fp8}$是转换得到的FLOAT8张量。

  - MAX_OFFSET对照表：

    | 输入     | 输出       | MAX_OFFSET |
    | -------- | ---------- | ---------- |
    | FP4_E2M1 | FP8_E5M2   | 13         |
    | FP4_E1M2 | FP8_E5M2   | 15         |
    | FP4_E2M1 | FP8_E4M3FN | 6          |
    | FP4_E1M2 | FP8_E4M3FN | 8          |

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnMxToBlockMxQuantGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnMxToBlockMxQuant`接口执行计算。

```cpp
aclnnStatus aclnnMxToBlockMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *mxscale,
  int64_t          dstType,
  const aclTensor *y,
  const aclTensor *scale1,
  const aclTensor *scale2,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnMxToBlockMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnMxToBlockMxQuantGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1600px"><colgroup>
  <col style="width: 300px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 250px">
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
      <td>表示算子输入的量化Tensor。对应公式中的x_fp4。</td>
      <td><ul><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>输入</td>
      <td>调用DynamicMxQuant计算得到的量化尺度。对应公式中的mxscale_fp4_block。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需满足约束说明中的公式。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>指定输出y的数据类型。</td>
      <td>输入范围为{35, 36}，分别对应输出y的数据类型为{35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>输出</td>
      <td>表示量化输出Tensor。对应公式中的x_fp8。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape维度与x保持一致。</li><li>数据类型需与dstType对应。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN</td>
      <td>ND</td>
      <td>2-3</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale1 (aclTensor*)</td>
      <td>输出</td>
      <td>表示-1轴每个分组对应的量化尺度。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需满足约束说明中的公式</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale2 (aclTensor*)</td>
      <td>输出</td>
      <td>表示-2轴每个分组对应的量化尺度。输出需要对每两行数据进行交织处理。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需满足约束说明中的公式。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-4</td>
      <td>×</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 319px">
  <col style="width: 108px">
  <col style="width: 621px">
  </colgroup>
  <thead>
  <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
  </tr></thead>
  <tbody>
  <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入参数是必选输入、输出，且是空指针。</td>
  </tr>
  <tr>
  </tr>
  <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>x、mxscale、y、scale1、scale2的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x、mxscale、y、scale1、scale2的shape不满足校验条件。</td>
  </tr>
  <tr>
    <td>x、mxscale、y、scale1、scale2的维度不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>dstType不符合当前支持的值。</td>
  </tr>
  <tr>
    <td>y的数据类型和dstType不符合对应关系。</td>
  </tr>
  <tr>
    <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="1">361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
  </tbody></table>

## aclnnMxToBlockMxQuant

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
  <col style="width: 173px">
  <col style="width: 127px">
  <col style="width: 748px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMxToBlockMxQuantGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码,具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - `aclnnMxToBlockMxQuant`默认确定性实现。
- x只支持2维或3维输入，且-2轴是64的倍数, -1轴是2的倍数,不支持非连续Tensor,不支持空Tensor。
- 关于x、mxscale、scale1、scale2的shape约束说明如下：
  - rank(mxscale) = rank(x) + 1。
  - mxscale.shape[-2] = (Ceil(x.shape[-1], 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - 其它维度与输入x一致。
- 关于输出scale1的shape约束说明如下：
  - rank(scale1) = rank(x) + 1。
  - scale1.shape[-2] = (Ceil(x.shape[-1], 32) + 2 - 1) / 2。
  - scale1.shape[-1] = 2。
  - 其它维度和输入x保持一致。
- 关于输出scale2的shape约束说明如下：
  - rank(scale2) = rank(x) + 1。
  - scale2.shape[-3] = ((Ceil(x.shape[-2], 32) + 2 - 1) / 2) * 2 / 2。
  - scale2.shape[-2] = x.shape[-1]。
  - scale2.shape[-1] = 2。
  - 其它维度和输入x保持一致。
- dstType仅支持{35, 36}，对应{FLOAT8_E5M2, FLOAT8_E4M3FN}。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_mx_to_block_mx_quant.h"

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

int aclnnMxToBlockMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {64, 2};
    std::vector<int64_t> scaleShape = {64, 1, 2};
    std::vector<int64_t> yOutShape = {64, 2};
    std::vector<int64_t> scale1Shape = {64, 1, 2};
    std::vector<int64_t> scale2Shape = {1, 2, 2};

    void* xDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* scale1DeviceAddr = nullptr;
    void* scale2DeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* scale1Out = nullptr;
    aclTensor* scale2Out = nullptr;

    std::vector<uint8_t> xHostData(128, 0x12);
    std::vector<uint8_t> scaleHostData(128, 128);
    std::vector<uint8_t> yOutHostData(128, 0);
    std::vector<uint8_t> scale1HostData(128, 0);
    std::vector<uint8_t> scale2HostData(4, 0);
    int64_t dstType = 36;

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT4_E2M1, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(mxscale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleDeviceAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建scale1Out aclTensor
    ret = CreateAclTensor(scale1HostData, scale1Shape, &scale1DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1TensorPtr(scale1Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale1DeviceAddrPtr(scale1DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建scale2Out aclTensor
    ret = CreateAclTensor(scale2HostData, scale2Shape, &scale2DeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Out);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2TensorPtr(scale2Out, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scale2DeviceAddrPtr(scale2DeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnMxToBlockMxQuant第一段接口
    ret = aclnnMxToBlockMxQuantGetWorkspaceSize(x, mxscale, dstType, yOut, scale1Out, scale2Out,
                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnMxToBlockMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnMxToBlockMxQuant第二段接口
    ret = aclnnMxToBlockMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMxToBlockMxQuant failed. ERROR: %d\n", ret); return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出y的值，将device侧内存上的结果拷贝至host侧
    auto ySize = GetShapeSize(yOutShape);
    std::vector<uint8_t> yOutData(ySize, 0);
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      ySize * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < ySize; i++) {
        LOG_PRINT("y[%ld] is: %u\n", i, yOutData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnMxToBlockMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMxToBlockMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
