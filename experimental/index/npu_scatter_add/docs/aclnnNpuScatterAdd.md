# aclnnNpuScatterAdd

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

- 算子功能：基于预排序索引的带缩放因子散射累加（MoE token 聚合）。根据索引向量 `indices` 将源张量 `x` 的行按缩放因子 `s` 加权后累加到目标张量 `y` 的对应行上，计算结果 in-place 写入 `y`。

- 计算公式（提供缩放因子 `s` 时）：

  $$
  y[\text{indices}[i], :] \mathrel{+}= x[i, :] \times s[i], \quad i = 0, 1, \ldots, S-1
  $$

  不提供缩放因子时：

  $$
  y[\text{indices}[i], :] \mathrel{+}= x[i, :], \quad i = 0, 1, \ldots, S-1
  $$

  其中 `x` 形状为 `(S, H)`，`y` 形状为 `(D, H)` 且同时为输出，`s` 形状为 `(S,)`，`indices` 形状为 `(S,)` 且取值范围为 `[0, D)`。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnNpuScatterAddGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNpuScatterAdd”接口执行计算。

```Cpp
aclnnStatus aclnnNpuScatterAddGetWorkspaceSize(
  const aclTensor*     x,
  const aclTensor*     y,
  const aclTensor*     s,
  const aclTensor*     indices,
  const aclTensor*     sortIdx,
  const aclTensor*     validTokenNum,
  bool                 useHighPrecision,
  uint64_t*            workspaceSize,
  aclOpExecutor**      executor)
```

```Cpp
aclnnStatus aclnnNpuScatterAdd(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnNpuScatterAddGetWorkspaceSize

- **参数说明：**

  <table class="tg" style="undefined;table-layout: fixed; width: 1445px"><colgroup>
  <col style="width: 165px">
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
      <td class="tg-0pky">x（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的x，源张量。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与y的数据类型一致。</li>
          <li>必须为2维张量。</li>
        </ul>
      </td>
      <td class="tg-0pky">BFLOAT16、FLOAT16</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">2</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">y（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">公式中的y，目标张量，计算结果in-place累加写入。</td>
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
      <td class="tg-0pky">s（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的s，逐行缩放因子。可选输入，不使用时传nullptr。</td>
      <td class="tg-0pky">
        <ul>
          <li>数据类型与x的数据类型一致。</li>
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
          <li>元素取值范围为[0, y的dim[0])。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">sortIdx（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的sort_idx，argsort(indices)的结果。</td>
      <td class="tg-0pky">
        <ul>
          <li>必须为1维张量，dim[0]与x的dim[0]一致。</li>
          <li>必须由调用方保证为indices的argsort结果。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">validTokenNum（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">有效token数量，提供时仅处理前validTokenNum个行。可选输入，不使用时传nullptr。</td>
      <td class="tg-0pky">
        <ul>
          <li>必须为shape是(1,)的1维张量。</li>
        </ul>
      </td>
      <td class="tg-0pky">INT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">√</td>
    </tr>
    <tr>
      <td class="tg-0pky">useHighPrecision（bool）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">是否使用高精度模式（在FP32下完成缩放和累加后再转回原精度）。默认为false。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
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
      <td>传入的x、y、indices或sortIdx是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>x、y的数据类型不在支持的范围之内（仅支持BFLOAT16、FLOAT16）。</td>
    </tr>
    <tr>
      <td>x、y的数据类型不一致，或s的数据类型与x不一致。</td>
    </tr>
    <tr>
      <td>indices、sortIdx或validTokenNum的数据类型不为INT32。</td>
    </tr>
    <tr>
      <td>x或y不是2维张量，indices、sortIdx或s不是1维张量。</td>
    </tr>
    <tr>
      <td>x、y的dim[1]不一致。</td>
    </tr>
    <tr>
      <td>x、indices、sortIdx的dim[0]不一致，或s的dim[0]与x的dim[0]不一致。</td>
    </tr>
    <tr>
      <td>validTokenNum的shape不为(1,)。</td>
    </tr>
    <tr>
      <td>隐藏维度H超出UB容量限制（align(H) * (element_size * 3 + 4 * 3) + 32 &gt; 180KB）。</td>
    </tr>
  </tbody>
  </table>

## aclnnNpuScatterAdd

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNpuScatterAddGetWorkspaceSize获取。</td>
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

- `sort_idx` 必须是 `indices` 的 argsort 结果，即 `sort_idx = argsort(indices)`，由调用方保证。
- 隐藏维度 H 需满足：`align(H) * (element_size * 3 + 4 * 3) + 32 <= 180KB`（受 UB 容量限制，其中 `align(H)` 为 H 按 32 字节对齐后的元素个数，`element_size` 为 2 字节）。
- 该算子为 in-place 算子，计算结果直接累加写入 `y`；`y` 与 `x`、`s`、`indices`、`sort_idx` 的存储区域不应重叠。
- 非连续 Tensor 会在接口内部转换为连续 Tensor 后参与计算，`y` 非连续时结果会自动拷回。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_npu_scatter_add.h"

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
    int64_t S = 8;
    int64_t D = 4;
    int64_t H = 4;
    std::vector<int64_t> xShape = {S, H};
    std::vector<int64_t> yShape = {D, H};
    std::vector<int64_t> sShape = {S};
    std::vector<int64_t> indicesShape = {S};
    void* xDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* sDeviceAddr = nullptr;
    void* indicesDeviceAddr = nullptr;
    void* sortIdxDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* s = nullptr;
    aclTensor* indices = nullptr;
    aclTensor* sortIdx = nullptr;

    // x的第i行所有元素为i+1, 缩放因子全1, y初始为0
    std::vector<aclFloat16> xHostData;
    for (int64_t i = 0; i < S * H; i++) {
        xHostData.push_back(aclFloatToFloat16(static_cast<float>(i / H + 1)));
    }
    std::vector<aclFloat16> yHostData(D * H, aclFloatToFloat16(0));
    std::vector<aclFloat16> sHostData(S, aclFloatToFloat16(1));
    std::vector<int32_t> indicesHostData = {3, 0, 2, 1, 3, 0, 1, 2};
    // sort_idx = argsort(indices), 必须由调用方保证
    std::vector<int32_t> sortIdxHostData = {1, 5, 3, 6, 2, 7, 0, 4};

    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sHostData, sShape, &sDeviceAddr, aclDataType::ACL_FLOAT16, &s);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT32, &indices);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sortIdxHostData, indicesShape, &sortIdxDeviceAddr, aclDataType::ACL_INT32, &sortIdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 可选输入valid_token_num不使用时传nullptr, 高精度模式不开
    const aclTensor* validTokenNum = nullptr;
    bool useHighPrecision = false;

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    ret = aclnnNpuScatterAddGetWorkspaceSize(x, y, s, indices, sortIdx, validTokenNum, useHighPrecision,
                                             &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnNpuScatterAdd(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNpuScatterAdd failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    // 预期结果: y[0]=2+6=8, y[1]=4+8=12, y[2]=3+7=10, y[3]=1+5=6 (每行所有元素相同)
    auto size = GetShapeSize(yShape);
    std::vector<aclFloat16> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("y[%ld][%ld] is: %f\n", i / H, i % H, aclFloat16ToFloat(resultData[i]));
    }

    // 6. 释放aclTensor
    aclDestroyTensor(x);
    aclDestroyTensor(y);
    aclDestroyTensor(s);
    aclDestroyTensor(indices);
    aclDestroyTensor(sortIdx);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(sDeviceAddr);
    aclrtFree(indicesDeviceAddr);
    aclrtFree(sortIdxDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
