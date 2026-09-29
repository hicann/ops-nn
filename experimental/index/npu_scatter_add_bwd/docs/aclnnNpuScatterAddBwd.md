# aclnnNpuScatterAddBwd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
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

- 算子功能：NpuScatterAddBwd 是 NpuScatterAdd（带缩放因子散射累加，MoE token 聚合）的反向算子，根据索引向量 `indices` 从上游梯度 `y_grad` 中取出对应行，分别计算对源张量 `x` 和缩放因子 `s` 的梯度。

- 计算公式（对每个 $i = 0, 1, \ldots, N-1$）：

  $$
  x\_grad[i, :] = y\_grad[\text{indices}[i], :] \times s[i]
  $$

  $$
  s\_grad[i] = \sum_{j=0}^{H-1} y\_grad[\text{indices}[i], j] \times x[i, j]
  $$

  其中 `y_grad` 形状为 `(D, H)`，`x` 形状为 `(N, H)`，`s` 形状为 `(N,)`，`indices` 形状为 `(N,)` 且取值范围为 `[0, D)`；输出 `x_grad` 形状为 `(N, H)`，`s_grad` 形状为 `(N,)`。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnNpuScatterAddBwdGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNpuScatterAddBwd”接口执行计算。

```Cpp
aclnnStatus aclnnNpuScatterAddBwdGetWorkspaceSize(
  const aclTensor*     yGrad,
  const aclTensor*     x,
  const aclTensor*     s,
  const aclTensor*     indices,
  const aclTensor*     xGrad,
  const aclTensor*     sGrad,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnNpuScatterAddBwd(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnNpuScatterAddBwdGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 175px">
  <col style="width: 160px">
  <col style="width: 150px">
  <col style="width: 300px">
  <col style="width: 280px">
  <col style="width: 115px">
  <col style="width: 130px">
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
      <td class="tg-0pky">yGrad（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的y_grad，上游梯度张量。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与x的数据类型一致。</li>
          <li>必须为2维张量，dim[1]与x的dim[1]一致。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">x（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的x，前向中的源张量。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与y_grad的数据类型一致。</li>
          <li>必须为2维张量。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">s（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的s，前向中的逐行缩放因子。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与y_grad的数据类型一致。</li>
          <li>必须为1维张量，dim[0]与x的dim[0]一致。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">indices（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的indices，目标行索引。</td>
      <td class="tg-0pky">
        <ul>
          <li>必须为1维张量，dim[0]与x的dim[0]一致。</li>
          <li>元素取值范围为[0, y_grad的dim[0])。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">xGrad（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">公式中的x_grad，x的梯度。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与x的数据类型一致。</li>
          <li>shape与x一致，必须为2维张量。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">sGrad（aclTensor*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">公式中的s_grad，s的梯度。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与s的数据类型一致。</li>
          <li>shape与s一致，必须为1维张量。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
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

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1147px"><colgroup>
  <col style="width: 286px">
  <col style="width: 123px">
  <col style="width: 738px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的y_grad、x、s、indices、x_grad或s_grad是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>y_grad、x、s的数据类型不在支持的范围之内（仅支持BFLOAT16、FLOAT16）。</td>
    </tr>
    <tr>
      <td>y_grad、x、s的数据类型不一致，或x_grad、s_grad的数据类型与对应输入不一致。</td>
    </tr>
    <tr>
      <td>indices的数据类型不为INT32。</td>
    </tr>
    <tr>
      <td>y_grad或x不是2维张量，s或indices不是1维张量。</td>
    </tr>
    <tr>
      <td>y_grad、x的dim[1]不一致，或x、s、indices的dim[0]不一致。</td>
    </tr>
    <tr>
      <td>x_grad、s_grad的shape与x、s不一致。</td>
    </tr>
  </tbody>
  </table>

## aclnnNpuScatterAddBwd

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 167px">
  <col style="width: 134px">
  <col style="width: 848px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNpuScatterAddBwdGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `y_grad`、`x`、`s` 的 dtype 必须一致，且仅支持 BFLOAT16、FLOAT16；`indices` 的 dtype 必须为 INT32。
- 隐藏维度 H 需满足：`align(H) * (element_size * 4 + 4 * 4) + 32 <= 180KB`（受 UB 容量限制，其中 `align(H)` 为 H 按 32 字节对齐后的元素个数，`element_size` 为 2 字节）。
- 非连续 Tensor 会在接口内部转换为连续 Tensor 后参与计算，输出非连续时结果会自动拷回。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_npu_scatter_add_bwd.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)     \
    do {                            \
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

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    int64_t D = 4;
    int64_t N = 8;
    int64_t H = 4;
    std::vector<int64_t> yGradShape = {D, H};
    std::vector<int64_t> xShape = {N, H};
    std::vector<int64_t> sShape = {N};
    std::vector<int64_t> indicesShape = {N};
    void* yGradDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* sDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* xGradDeviceAddr = nullptr;
    void* sGradDeviceAddr = nullptr;
    aclTensor* yGrad = nullptr;
    aclTensor* x = nullptr;
    aclTensor* s = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* xGrad = nullptr;
    aclTensor* sGrad = nullptr;

    std::vector<aclFloat16> yGradHostData;
    for (int64_t i = 0; i < D * H; i++) {
        yGradHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> xHostData;
    for (int64_t i = 0; i < N * H; i++) {
        xHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> sHostData(N, aclFloatToFloat16(1));
    std::vector<int32_t> indicesHostData = {3, 0, 2, 1, 3, 0, 1, 2};

    ret = CreateAclTensor(yGradHostData, yGradShape, &yGradDeviceAddr, aclDataType::ACL_FLOAT16, &yGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sHostData, sShape, &sDeviceAddr, aclDataType::ACL_FLOAT16, &s);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<aclFloat16> xGradHostData(N * H, aclFloatToFloat16(0));
    ret = CreateAclTensor(xGradHostData, xShape, &xGradDeviceAddr, aclDataType::ACL_FLOAT16, &xGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::vector<aclFloat16> sGradHostData(N, aclFloatToFloat16(0));
    ret = CreateAclTensor(sGradHostData, sShape, &sGradDeviceAddr, aclDataType::ACL_FLOAT16, &sGrad);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnNpuScatterAddBwdGetWorkspaceSize(yGrad, x, s, indices, xGrad, sGrad, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddBwdGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNpuScatterAddBwd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddBwd failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    // 预期结果：x_grad[i] = indices[i] + 1, s_grad[i] = 4 * (i + 1) * (indices[i] + 1)
    auto xGradSize = GetShapeSize(xShape);
    std::vector<aclFloat16> xGradResult(xGradSize, 0);
    ret = aclrtMemcpy(xGradResult.data(), xGradResult.size() * sizeof(xGradResult[0]), xGradDeviceAddr,
                      xGradSize * sizeof(xGradResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy x_grad from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < xGradSize; i++) {
        LOG_PRINT("x_grad[%ld][%ld] is: %f\n", i / H, i % H, aclFloat16ToFloat(xGradResult[i]));
    }
    auto sGradSize = GetShapeSize(sShape);
    std::vector<aclFloat16> sGradResult(sGradSize, 0);
    ret = aclrtMemcpy(sGradResult.data(), sGradResult.size() * sizeof(sGradResult[0]), sGradDeviceAddr,
                      sGradSize * sizeof(sGradResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy s_grad from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < sGradSize; i++) {
        LOG_PRINT("s_grad[%ld] is: %f\n", i, aclFloat16ToFloat(sGradResult[i]));
    }

    // 6. 释放aclTensor
    aclDestroyTensor(yGrad);
    aclDestroyTensor(x);
    aclDestroyTensor(s);
    aclDestroyTensor(indices);
    aclDestroyTensor(xGrad);
    aclDestroyTensor(sGrad);

    // 7. 释放device资源
    aclrtFree(yGradDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(sDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(xGradDeviceAddr);
    aclrtFree(sGradDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
