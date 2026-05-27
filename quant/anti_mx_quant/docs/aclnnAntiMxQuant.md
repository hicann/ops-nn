# aclnnAntiMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/anti_mx_quant)

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

- 接口功能：将调用 `aclnnDynamicMxQuant/aclnnDynamicMxQuantV2` 量化得到的 FLOAT4/FLOAT8 的 Tensor 反量化为 FLOAT16/BFLOAT16/FLOAT32 格式。

- 计算公式：

  $$
  X_{dq} = X_q \times 2^{sf - bias}
  $$

  - 其中 $sf$ 是缩放因子，由输入 mxscale 提供；$bias$ 是指数位的偏移，对于 FLOAT8_E8M0 格式，$bias=127$；$X_q$ 是量化得到的 FLOAT4/FLOAT8 张量；$X_{dq}$ 是反量化得到的 FLOAT16/BFLOAT16/FLOAT32 张量。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 `aclnnAntiMxQuantGetWorkspaceSize` 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 `aclnnAntiMxQuant` 接口执行计算。

```cpp
aclnnStatus aclnnAntiMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *mxscale,
  int64_t          axis,
  int64_t          dstType,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAntiMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAntiMxQuantGetWorkspaceSize

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
      <td>表示算子输入的量化 Tensor。对应公式中的 Xq。</td>
      <td>支持空 Tensor。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7，需要和DynamicMxQuant的输出保持一致</td>
      <td>FLOAT8支持非连续，FLOAT4不支持非连续</td>
    </tr>
    <tr>
      <td>mxscale (aclTensor*)</td>
      <td>输入</td>
      <td>调用 DynamicMxQuant 计算得到的量化尺度。对应公式中的 sf。</td>
      <td><ul><li>支持空 Tensor。</li><li>shape 需满足约束说明中的公式。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8，需要和DynamicMxQuant的输出保持一致</td>
      <td>√</td>
    </tr>
    <tr>
      <td>axis (int64_t)</td>
      <td>输入</td>
      <td>指定反量化轴。</td>
      <td>输入范围为[-D, D-1]，其中 D 为输入 x 的维度数。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>指定输出 y 的数据类型。</td>
      <td>输入范围为{0, 1, 27}，分别对应输出 y 的数据类型为 {0: FLOAT32, 1: FLOAT16, 27: BFLOAT16}。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y (aclTensor*)</td>
      <td>输出</td>
      <td>表示反量化后的输出 Tensor。对应公式中的 Xdq。</td>
      <td><ul><li>支持空 Tensor。</li><li>shape 维度与 x 保持一致。</li><li>数据类型需与 dstType 对应。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>FLOAT8支持非连续，FLOAT4不支持非连续</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
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
      <td>x、mxscale、y 的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>x、mxscale、y 的 shape 不满足校验条件。</td>
  </tr>
  <tr>
    <td>x、mxscale、y 的维度不在支持的范围之内。</td>
  </tr>
  <tr>
    <td>axis、dstType 不符合当前支持的值。</td>
  </tr>
  <tr>
    <td>y 的数据类型和 dstType 不符合对应关系。</td>
  </tr>
  <tr>
    <td rowspan="1">ACLNN_ERR_PARAM_NULLPTR</td>
    <td rowspan="1">361001</td>
    <td>当前平台不在支持的平台范围内。</td>
  </tr>
  </tbody></table>

## aclnnAntiMxQuant

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
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnAntiMxQuantGetWorkspaceSize 获取。</td>
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
  - `aclnnAntiMxQuant` 默认确定性实现。
- 关于 x、mxscale 的 shape 约束说明如下：
  - 如果输入x的数据类型是float4_e2m1或float4_e1m2，x.shape[-1]必须是偶数。
  - axis_change = axis if axis >= 0 else axis + rank(x)。
  - mxscale.shape[axis_change] = (ceil(x.shape[axis], 32) + 2 - 1) / 2。
  - mxscale.shape[-1] = 2。
  - rank(mxscale) = rank(x) + 1。
  - 其它维度与输入 x 一致。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_anti_mx_quant.h"

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

int aclnnAntiMxQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // x: FP8_E4M3FN, shape {1, 4}
    std::vector<int64_t> xShape = {1, 4};
    // mxscale: FP8_E8M0, shape {1, 1, 2}
    // ceil(4/32) = 1, (1+2-1)/2 = 1, append 2 -> {1, 1, 2}
    std::vector<int64_t> mxscaleShape = {1, 1, 2};
    // y: BF16, shape {1, 4}
    std::vector<int64_t> yOutShape = {1, 4};

    void* xDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* yOut = nullptr;

    std::vector<uint8_t> xHostData = {0, 72, 96, 120};
    std::vector<uint8_t> mxscaleHostData = {128, 0};
    std::vector<uint16_t> yOutHostData = {0, 16640, 17024, 17408};

    int64_t axis = -1;
    int64_t dstType = 27;  // BFLOAT16

    // 创建x aclTensor(FP8_E4M3FN)
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建mxscale aclTensor(FP8_E8M0)
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleTensorPtr(mxscale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> mxscaleDeviceAddrPtr(mxscaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建yOut aclTensor(BF16)
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_BF16, &yOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnAntiMxQuant第一段接口
    ret = aclnnAntiMxQuantGetWorkspaceSize(x, mxscale, axis, dstType, yOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnAntiMxQuant第二段接口
    ret = aclnnAntiMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuant failed. ERROR: %d\n", ret); return ret);

    // 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出y的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(yOutShape);
    std::vector<uint16_t> yOutData(size, 0);
    ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                      size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
              return ret);
    for (int64_t i = 0; i < size; i++) {
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
    auto ret = aclnnAntiMxQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAntiMxQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
