# aclnnForeachZeroInplace

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

- 接口功能：原地更新输入张量列表，输入张量列表的每个张量置为0。

- 计算公式：

  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  $$

  $$
  x_i = 0 (i=0,1,...n-1)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnForeachZeroInplaceGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnForeachZeroInplace”接口执行计算。

```Cpp
aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(
  const aclTensorList *x,
  uint64_t *workspaceSize,
  aclOpExecutor **executor)
```

```Cpp
aclnnStatus aclnnForeachZeroInplace(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnForeachZeroInplaceGetWorkspaceSize

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
      <td>x（aclTensorList*）</td>
      <td>输入/输出</td>
      <td>表示进行置0运算的输入和输出张量列表，对应公式中的`x`。</td>
      <td><ul><li>支持空Tensor。</li><li>Tensor列表的Tensor数量取值范围为0~256，数量为0（空列表）时接口直接返回成功、不执行计算。</li><li>该参数中所有Tensor的数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16、INT16、INT32</td>
      <td>ND</td>
      <td>0-8</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>x的数据类型不在支持的范围之内，或TensorList中存在空指针元素，或Tensor使用私有格式。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="3">561002</td>
      <td>x中的Tensor的数据类型不一致。</td>
    </tr>
    <tr>
      <td>x中的Tensor维度超过8维。</td>
    </tr>
    <tr>
      <td>x的长度超过256个。</td>
    </tr>
  </tbody></table>

## aclnnForeachZeroInplace

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnForeachZeroInplaceGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- TensorList中的所有Tensor必须使用相同的数据类型。
- aclnnForeachZeroInplace在Atlas A2/A3系列产品上支持非连续Tensor的原地清零，仅更新各Tensor逻辑view对应的元素，shape、stride和offset保持不变，view范围之外的storage数据保持不变。
- x为空TensorList或所有Tensor均为空时，执行成功空操作。
- 调用方需保证可写Tensor的逻辑元素以及列表中各Tensor的可写区域不重叠；共享storage的多个互不重叠view可以使用。
- 确定性计算：
  - aclnnForeachZeroInplace默认确定性实现。

## 调用示例

以下代码展示初始化、输入构造、两段式调用、同步、结果回传和资源释放。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_foreach_zero_inplace.h"

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
    for (auto dim : shape) {
        shapeSize *= dim;
    }
    return shapeSize;
}

void LogCleanupFailure(const char* operation, aclError status)
{
    if (status != ACL_SUCCESS) {
        LOG_PRINT("%s failed. ERROR: %d\n", operation, status);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LogCleanupFailure("aclrtResetDevice", aclrtResetDevice(deviceId));
              LogCleanupFailure("aclFinalize", aclFinalize()); return ret);
    return ACL_SUCCESS;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    *deviceAddr = nullptr;
    *tensor = nullptr;
    const size_t size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return *tensor == nullptr ? ACL_ERROR_FAILURE : ACL_SUCCESS;
}

int main()
{
    constexpr int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<int64_t> shape1 = {2, 3};
    std::vector<int64_t> shape2 = {1, 3};
    std::vector<float> host1 = {1, 2, 3, 4, 5, 6};
    std::vector<float> host2 = {7, 8, 9};
    void* inputAddr1 = nullptr;
    void* inputAddr2 = nullptr;
    void* workspace = nullptr;
    aclTensor* input1 = nullptr;
    aclTensor* input2 = nullptr;
    aclTensorList* inputList = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t workspaceSize = 0;
    bool streamHasPendingWork = false;

    auto cleanup = [&](int status) {
        bool deviceWasReset = false;
        auto record = [&](const char* operation, aclError cleanupStatus) {
            if (cleanupStatus != ACL_SUCCESS) {
                LOG_PRINT("%s failed. ERROR: %d\n", operation, cleanupStatus);
                if (status == ACL_SUCCESS) {
                    status = cleanupStatus;
                }
            }
        };
        if (stream != nullptr) {
            const aclError streamStatus = streamHasPendingWork ? aclrtDestroyStreamForce(stream) :
                                                                 aclrtDestroyStream(stream);
            record(streamHasPendingWork ? "aclrtDestroyStreamForce" : "aclrtDestroyStream", streamStatus);
            if (streamHasPendingWork && streamStatus != ACL_SUCCESS) {
                const aclError resetStatus = aclrtResetDeviceForce(deviceId);
                record("aclrtResetDeviceForce", resetStatus);
                if (resetStatus != ACL_SUCCESS) {
                    LOG_PRINT("Device reset failed; retain in-flight resources until process exit.\n");
                    return status;
                }
                deviceWasReset = true;
                workspace = nullptr;
                inputAddr1 = nullptr;
                inputAddr2 = nullptr;
            }
            stream = nullptr;
            streamHasPendingWork = false;
        }
        if (inputList != nullptr) {
            record("aclDestroyTensorList", aclDestroyTensorList(inputList));
        } else {
            if (input1 != nullptr) {
                record("aclDestroyTensor(input1)", aclDestroyTensor(input1));
            }
            if (input2 != nullptr) {
                record("aclDestroyTensor(input2)", aclDestroyTensor(input2));
            }
        }
        if (!deviceWasReset && workspace != nullptr) {
            record("aclrtFree(workspace)", aclrtFree(workspace));
        }
        if (!deviceWasReset && inputAddr1 != nullptr) {
            record("aclrtFree(input1)", aclrtFree(inputAddr1));
        }
        if (!deviceWasReset && inputAddr2 != nullptr) {
            record("aclrtFree(input2)", aclrtFree(inputAddr2));
        }
        if (!deviceWasReset) {
            record("aclrtResetDevice", aclrtResetDevice(deviceId));
        }
        record("aclFinalize", aclFinalize());
        return status;
    };

    ret = CreateAclTensor(host1, shape1, &inputAddr1, ACL_FLOAT, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    ret = CreateAclTensor(host2, shape2, &inputAddr2, ACL_FLOAT, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    std::vector<aclTensor*> tensors{input1, input2};
    inputList = aclCreateTensorList(tensors.data(), tensors.size());
    CHECK_RET(inputList != nullptr, return cleanup(ACL_ERROR_FAILURE));

    ret = aclnnForeachZeroInplaceGetWorkspaceSize(inputList, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    }
    ret = aclnnForeachZeroInplace(workspace, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    streamHasPendingWork = true;

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    streamHasPendingWork = false;

    std::vector<float> result1(host1.size());
    std::vector<float> result2(host2.size());
    ret = aclrtMemcpy(result1.data(), result1.size() * sizeof(float), inputAddr1,
                      result1.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));
    ret = aclrtMemcpy(result2.data(), result2.size() * sizeof(float), inputAddr2,
                      result2.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return cleanup(ret));

    for (size_t i = 0; i < result1.size(); ++i) {
        LOG_PRINT("result1[%zu]=%f\n", i, result1[i]);
    }
    for (size_t i = 0; i < result2.size(); ++i) {
        LOG_PRINT("result2[%zu]=%f\n", i, result2[i]);
    }
    return cleanup(ACL_SUCCESS);
}
```
