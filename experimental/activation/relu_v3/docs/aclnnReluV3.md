# aclnnReluV3

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/activation/relu_v3)

## 产品支持情况

| 产品                                           | 是否支持 |
|:---------------------------------------------|:----:|
| <term>Ascend 950PR/Ascend 950DT</term>       |  ×   |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |  ×   |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |  √   |
| <term>Atlas 200I/500 A2 推理产品</term>          |  ×   |
| <term>Atlas 推理系列产品</term>                    |  ×   |
| <term>Atlas 训练系列产品</term>                    |  ×   |

## 功能说明

- 接口功能：对输入tensor逐元素计算修正线性单元，输出计算结果，并同时返回一个掩码用于指示输入元素中哪些值大于0。
- 计算公式：
  $$y_i = \max(x_i, 0)$$
  $$mask_i = (x_i > 0)$$

## 函数原型

- 每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnReluV3GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnReluV3”接口执行计算。

```Cpp
aclnnStatus aclnnReluV3GetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *yOut,
    const aclTensor *maskOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnReluV3(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
```

## aclnnReluV3GetWorkspaceSize

- **参数说明：**

  | 参数名                        | 输入/输出 | 描述                          | 使用说明         | 数据类型                                    | 数据格式    | 维度(shape) | 非连续Tensor |
  |----------------------------|-------|-----------------------------|--------------|-----------------------------------------|---------|-----------|-----------|
  | x（aclTensor *）             | 输入    | 待进行ReluV3计算的入参，公式中的$x$。     | -            | FLOAT16、FLOAT、INT32、INT8、UINT8、BFLOAT16 | NC1HWC0 | 5         | √         |
  | yOut（aclTensor *）          | 输出    | 待进行ReluV3计算的出参，公式中的$y$。     | shape需要与x一致。 | FLOAT16、FLOAT、INT32、INT8、UINT8、BFLOAT16 | NC1HWC0 | 5         | √         |
  | maskOut（aclTensor *）       | 输出    | 待进行ReluV3计算的出参，公式中的$mask$。  | shape需要与x一致。 | UINT8                                   | ND      | 5         | √         |
  | workspaceSize（uint64_t *）  | 输出    | 返回需要在Device侧申请的workspace大小。 | -            | -                                       | -       | -         | -         |
  | executor（aclOpExecutor **） | 输出    | 返回op执行器，包含了算子计算流程。          | -            | -                                       | -       | -         | -         |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table>
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
          <td>传入的x、yOut、maskOut是空指针。</td>
      </tr>
      <tr>
          <td rowspan="0">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="0">161002</td>
          <td>x、yOut、maskOut的数据类型或数据格式不在支持的范围之内。</td>
      </tr>
      <tr>
          <td>x和yOut的数据类型不相同。</td>
      </tr>
      <tr>
          <td>x、yOut、maskOut的shape不是5维，或者shape不一致。</td>
      </tr>
      </tbody>
  </table>

## aclnnReluV3

- **参数说明：**

  | 参数名           | 输入/输出 | 描述                                                          |
  |---------------|-------|-------------------------------------------------------------|
  | workspace     | 输入    | 在Device侧申请的workspace内存地址。                                   |
  | workspaceSize | 输入    | 在Device侧申请的workspace大小，由第一段接口aclnnReluV3GetWorkspaceSize获取。 |
  | executor      | 输入    | op执行器，包含了算子计算流程。                                            |
  | stream        | 输入    | 指定执行任务的Stream。                                              |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnReluV3默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

**aclnnReluV3接口调用示例：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_relu_v3.h"

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
int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}
int Init(int32_t deviceId, aclrtStream* stream) {
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
                                        aclDataType dataType, aclTensor** tensor) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NC1HWC0,
                                                        shape.data(), shape.size(), *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorND(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                                        aclDataType dataType, aclTensor** tensor) {
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
int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 1, 1, 1, 16};
    std::vector<int64_t> yOutShape = {1, 1, 1, 1, 16};
    std::vector<int64_t> maskOutShape = {1, 1, 1, 1, 2};
    void* xDeviceAddr = nullptr;
    void* yOutDeviceAddr = nullptr;
    void* maskOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* yOut = nullptr;
    aclTensor* maskOut = nullptr;
    std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    std::vector<float> yOutHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<uint8_t> maskOutHostData = {0, 0};
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yOut aclTensor
    ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT, &yOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建maskOut aclTensor
    ret = CreateAclTensorND(maskOutHostData, maskOutShape, &maskOutDeviceAddr, aclDataType::ACL_UINT8, &maskOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 3. 调用CANN算子库API，需要修改为具体的API名称
    // aclnnReluV3接口调用示例
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnReluV3第一段接口
    ret = aclnnReluV3GetWorkspaceSize(x, yOut, maskOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnReluV3第二段接口
    ret = aclnnReluV3(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnReluV3 failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yOutShape);
    std::vector<float> yOutResultData(size, 0);
    ret = aclrtMemcpy(yOutResultData.data(), yOutResultData.size() * sizeof(yOutResultData[0]), yOutDeviceAddr,
                                        size * sizeof(yOutResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("aclnnReluV3 yOut result[%ld] is: %f\n", i, yOutResultData[i]);
    }
    size = GetShapeSize(maskOutShape);
    std::vector<uint8_t> maskOutResultData(size, 0);
    ret = aclrtMemcpy(maskOutResultData.data(), maskOutResultData.size() * sizeof(maskOutResultData[0]), maskOutDeviceAddr,
                                        size * sizeof(maskOutResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy maskOut result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size * 8; i++) {
        LOG_PRINT("aclnnReluV3 maskOut result[%ld] is: %d\n", i, maskOutResultData[i / 8] >> i % 8 & 1);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(yOut);
    aclDestroyTensor(maskOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(yOutDeviceAddr);
    aclrtFree(maskOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
