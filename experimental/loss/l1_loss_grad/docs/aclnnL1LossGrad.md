# aclnnL1LossBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss/l1_loss_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

计算[aclnnL1LossBackward](../../../../loss/l1_loss/docs/aclnnL1Loss.md)平均绝对误差损失函数的反向传播梯度。reduction指定损失函数的计算方式，支持 'none'、'mean'、'sum'。'none' 表示不应用reduction，'mean' 表示输出的总和将除以输出中的元素数，'sum' 表示输出将被求和。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnL1LossBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnL1LossBackward”接口执行计算。

```Cpp
aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(
    const aclTensor* gradOutput, 
    const aclTensor* self,
    const aclTensor* target, 
    int64_t          reduction,
    aclTensor* gradInput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)
```

```Cpp
aclnnStatus aclnnL1LossBackward(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnL1LossBackwardGetWorkspaceSize

- **参数说明**：

    </style>
    <table class="tg" style="undefined;table-layout: fixed; width: 1547px"><colgroup>
    <col style="width: 217px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 350px">
    <col style="width: 200px">
    <col style="width: 115px">
    <col style="width: 120px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">参数名</th>
        <th class="tg-0pky">输入/输出</th>
        <th class="tg-0pky">描述</th>
        <th class="tg-0pky">使用说明</th>
        <th class="tg-0pky">数据类型</th>
        <th class="tg-0pky">数据格式</th>
        <th class="tg-0pky">维度(shape)</th>
        <th class="tg-0pky">非连续Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">gradOutput（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">梯度反向输入。</td>
        <td class="tg-0pky">shape需要与self和target满足<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。<br>数据类型与self、target的数据类型需满足数据类型推导规则（参见<a href="../../../../docs/zh/context/互转换关系.md" target="_blank">互转换关系</a>）。</td>
        <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">self（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">前向传播的预测值输入。</td>
        <td class="tg-0pky">shape需要与gradOutput、target满足<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。<br>数据类型与gradOutput、target的数据类型需满足数据类型推导规则。</td>
        <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">target（aclTensor*）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">真实的标签输入。</td>
        <td class="tg-0pky">shape需要与gradOutput、self满足<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast关系</a>。<br>数据类型与gradOutput、self的数据类型需满足数据类型推导规则。</td>
        <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">reduction（int64_t）</td>
        <td class="tg-0pky">输入</td>
        <td class="tg-0pky">指定要应用到输出的缩减。</td>
        <td class="tg-0pky">支持0(none)|1(mean)|2(sum)。<br>'none'表示不应用缩减<br>'mean'表示输出的总和将除以输出中的元素数<br>'sum'表示输出将被求和</td>
        <td class="tg-0pky">INT64</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradInput（aclTensor*）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">计算输出的梯度。</td>
        <td class="tg-0pky">shape为gradOutput，self，target做<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>后的结果。</td>
        <td class="tg-0pky">FLOAT、FLOAT16、BFLOAT16</td>
        <td class="tg-0pky">ND</td>
        <td class="tg-0pky">1-8</td>
        <td class="tg-0pky">√</td>
      </tr>
      <tr>
        <td class="tg-0pky">workspaceSize（uint64_t*）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
      <tr>
        <td class="tg-0pky">executor（aclOpExecutor**）</td>
        <td class="tg-0pky">输出</td>
        <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
        <td class="tg-0pky">-</td>
      </tr>
    </tbody></table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
    </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 269px">
  <col style="width: 135px">
  <col style="width: 746px">
    </colgroup>
    <thead>
      <tr>
        <th class="tg-0pky">返回值</th>
        <th class="tg-0pky">错误码</th>
        <th class="tg-0pky">描述</th>
      </tr></thead>
    <tbody>
      <tr>
        <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
        <td class="tg-0pky">161001</td>
        <td class="tg-0pky">传入的gradOutput、self、target或gradInput是空指针。</td>
      </tr>
      <tr>
        <td class="tg-0pky" rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td class="tg-0pky" rowspan="3">161002</td>
        <td class="tg-0pky">gradOutput、self、target或gradInput的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td class="tg-0lax">gradOutput、self和target、gradInput的shape不满足<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>规则或维度超过限制（8维）。</td>
      </tr>
      <tr>
        <td class="tg-0pky">gradOutput、self和target做<a href="../../../../docs/zh/context/broadcast关系.md" target="_blank">broadcast</a>后的shape与gradInput不一致。</td>
      </tr>
    </tbody>
    </table>

## aclnnL1LossBackward

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 200px">
      <col style="width: 150px">
      <col style="width: 800px">
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
          <td>在Device侧申请的workspace大小，由第一段接口aclnnL1LossBackwardGetWorkspaceSize获取。</td>
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


- **返回值**：

  **aclnnStatus**：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnL1LossBackward默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_l1_loss_backward.h"

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

    // 调用 aclrtMalloc 申请 device 侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续 tensor 的 strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用 aclCreateTensor 接口创建 aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    std::vector<int64_t> commonShape = {2, 2};

    void* gradsDeviceAddr = nullptr;
    void* predictDeviceAddr = nullptr;
    void* labelDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;

    aclTensor* grads = nullptr;
    aclTensor* predict = nullptr;
    aclTensor* label = nullptr;
    aclTensor* y = nullptr;

    // 构造测试数据
    std::vector<float> gradsHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<float> predictHostData = {2.0, 3.0, 0.0, 1.0};
    std::vector<float> labelHostData = {1.0, 1.0, 1.0, 1.0};
    std::vector<float> yHostData = {0.0, 0.0, 0.0, 0.0};

    // 创建输入 Tensors
    ret = CreateAclTensor(gradsHostData, commonShape, &gradsDeviceAddr, aclDataType::ACL_FLOAT, &grads);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(predictHostData, commonShape, &predictDeviceAddr, aclDataType::ACL_FLOAT, &predict);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(labelHostData, commonShape, &labelDeviceAddr, aclDataType::ACL_FLOAT, &label);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建输出 Tensor
    ret = CreateAclTensor(yHostData, commonShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 准备 Attribute
    int64_t reduction = 1; // 0: none, 1: mean, 2: sum

    // 3. 调用 CANN 算子库 API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 获取 Workspace 大小
    ret = aclnnL1LossBackwardGetWorkspaceSize(grads, predict, label, reduction, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 申请 workspace
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 执行算子
    ret = aclnnL1LossBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnL1LossBackward failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出结果
    auto size = GetShapeSize(commonShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("\nResult Y: \n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("Index[%ld]: %f\n", i, resultData[i]);
    }

    // 6. 释放资源
    aclDestroyTensor(grads);
    aclDestroyTensor(predict);
    aclDestroyTensor(label);
    aclDestroyTensor(y);

    aclrtFree(gradsDeviceAddr);
    aclrtFree(predictDeviceAddr);
    aclrtFree(labelDeviceAddr);
    aclrtFree(yDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
```