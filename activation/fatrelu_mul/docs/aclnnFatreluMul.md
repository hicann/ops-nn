# aclnnFatreluMul

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

- 接口功能：

  将输入Tensor按照最后一个维度分为左右两个Tensor：x1和x2，对左边的x1进行Threshold计算，将计算结果与x2相乘。

- 计算公式：

  给定输入张量x，最后一维的长度为2d，进行以下计算：

  1. 将x分割为两部分：

     $$
     x_1 = x[..., :d], \quad x_2 = x[..., d:]
     $$
  2. 对x1应用Threshold激活函数，定义如下：

     $$
     \text{Threshold}(x, \text{threshold}) =
        \begin{cases}
        0 & \text{if } x \leq \text{threshold} \\
        x & \text{if } x > \text{threshold}
        \end{cases}
     $$

     因此，计算：

     $$
     x_1 = \text{Threshold}(x_1, \text{threshold})
     $$
  3. 最终输出是x1和x2的逐元素乘积：

     $$
     \text{out} = x_1 \times x_2
     $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnFatreluMulGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFatreluMul”接口执行计算。

```Cpp
aclnnStatus aclnnFatreluMulGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *threshold,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnFatreluMul(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnFatreluMulGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 280px">
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
      <td>输入的张量，公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>最后一维值为偶数。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>不小于2</td>
      <td>√</td>
    </tr>
     <tr>
      <td>threshold（aclTensor*）</td>
      <td>输入</td>
      <td>Threshold计算的边界值，公式中的threshold。</td>
      <td><ul><li>数据类型和数据格式与x保持一致。</li><li>要求tensor中元素只有一个。</li><li>不支持NaN和Inf。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>不限制</td>
      <td>√</td>
    </tr>
      <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出的张量，公式中的out。</td>
      <td><ul><li>输出的数据类型与输入x的数据类型保持一致。</li><li>输出shape最后一维的值为输入shape最后一维值的二分之一。</li><li>其他维度的值和输入x的shape其他维度的值保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>不小于2</td>
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
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
<!-- npu="A3,910b" id7 -->
- <term>Atlas A2系列产品</term>、<term>Atlas A3系列产品</term>：最后一维值不大于1024，其他维度的乘积不大于200000。
<!-- end id7 -->
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的x、threshold或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>x的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>threshold的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>out的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="4">561002</td>
      <td>x的维度（rank）小于2。</td>
    </tr>
    <tr>
      <td>x最后一维的长度不是偶数。</td>
    </tr>
    <tr>
      <td>threshold的元素个数不为1。</td>
    </tr>
    <tr>
      <td>out的shape与x的shape不匹配（前导维度不一致或最后一维不为x最后一维的二分之一）。</td>
    </tr>
  </tbody></table>

## aclnnFatreluMul

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFatreluMulGetWorkspaceSize获取。</td>
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
  - aclnnFatreluMul默认确定性实现。

- 输入x中gate段（前半）为NaN的元素，输出对应位置为0。

- 典型场景尾轴为16的倍数，当尾轴为非32Byte对齐时，建议走小算子拼接逻辑。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fatrelu_mul.h"

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); aclrtResetDevice(deviceId);
              aclFinalize(); return ret);
    return 0;
}

// 错误返回前释放已申请的 device/acl 资源（DEMO.04：与main正常路径的释放顺序一致）
void FinalizeAclResources(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

void ReleaseAclTensor(aclTensor** tensor, void** deviceAddr)
{
    if (*tensor != nullptr) {
        aclDestroyTensor(*tensor);
        *tensor = nullptr;
    }
    if (*deviceAddr != nullptr) {
        aclrtFree(*deviceAddr);
        *deviceAddr = nullptr;
    }
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
              *deviceAddr = nullptr; return ret);

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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {2, 4};

    std::vector<float> inputHostData = {0, 1, 2, 3, 4, 5, 6, 7};

    void* inputDeviceAddr = nullptr;

    aclTensor* input = nullptr;
    // 创建threshold aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, FinalizeAclResources(deviceId, stream); return ret);

    // 创建threshold aclTensor
    std::vector<int64_t> threShape = {1};

    std::vector<float> threData = {0.01};

    void* threDeviceAddr = nullptr;

    aclTensor* threshold = nullptr;
    // 创建threshold aclTensor
    ret = CreateAclTensor(threData, threShape, &threDeviceAddr, aclDataType::ACL_FLOAT, &threshold);
    CHECK_RET(ret == ACL_SUCCESS, ReleaseAclTensor(&input, &inputDeviceAddr); FinalizeAclResources(deviceId, stream);
              return ret);

    std::vector<int64_t> outShape = {2, 2};
    std::vector<float> outHostData(2 * 2, 1);
    aclTensor* out = nullptr;
    void* outDeviceAddr = nullptr;
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, ReleaseAclTensor(&input, &inputDeviceAddr);
              ReleaseAclTensor(&threshold, &threDeviceAddr); FinalizeAclResources(deviceId, stream); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // 调用aclnnFatreluMul第一段接口
    ret = aclnnFatreluMulGetWorkspaceSize(input, threshold, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFatreluMulGetWorkspaceSize failed. ERROR: %d\n", ret);
              ReleaseAclTensor(&input, &inputDeviceAddr); ReleaseAclTensor(&threshold, &threDeviceAddr);
              ReleaseAclTensor(&out, &outDeviceAddr); FinalizeAclResources(deviceId, stream); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  ReleaseAclTensor(&input, &inputDeviceAddr); ReleaseAclTensor(&threshold, &threDeviceAddr);
                  ReleaseAclTensor(&out, &outDeviceAddr); FinalizeAclResources(deviceId, stream); return ret);
    }

    // 调用aclnnFatreluMul第二段接口
    ret = aclnnFatreluMul(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnFatreluMul failed. ERROR: %d\n", ret); aclDestroyAclOpExecutor(executor);
        ReleaseAclTensor(&input, &inputDeviceAddr); ReleaseAclTensor(&threshold, &threDeviceAddr);
        ReleaseAclTensor(&out, &outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } FinalizeAclResources(deviceId, stream); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
        aclDestroyAclOpExecutor(executor); ReleaseAclTensor(&input, &inputDeviceAddr);
        ReleaseAclTensor(&threshold, &threDeviceAddr); ReleaseAclTensor(&out, &outDeviceAddr);
        if (workspaceAddr != nullptr) { aclrtFree(workspaceAddr); } FinalizeAclResources(deviceId, stream); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(outShape, &outDeviceAddr);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(threshold);
    aclDestroyTensor(out);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
    aclrtFree(threDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
