# aclnnThnnFusedGruCell

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/rnn/thnn_fused_gru_cell)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->

<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->

<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：完成GRU（Gated Recurrent Unit）单个时间步的门控融合计算。输入两路门控预激活（输入侧与隐层侧）、上一步隐状态及可选的两路bias，一次调用产出新隐状态hy与反向计算复用的中间量storage，适用于循环神经网络逐步推理或训练中的高频单步计算。
- 计算公式：门控预激活input_gates与hidden_gates沿最后一维按门序[r, z, n]三等分为gi_r、gi_z、gi_n与gh_r、gh_z、gh_n；两路bias同步三等分为b1_r、b1_z、b1_n与b2_r、b2_z、b2_n，并沿batch维广播；hx为上一步隐状态。

  $$
  rg = \frac{1}{1 + e^{-(gi_r + gh_r + b1_r + b2_r)}}
  $$

  $$
  zg = \frac{1}{1 + e^{-(gi_z + gh_z + b1_z + b2_z)}}
  $$

  $$
  ng = \tanh(gi_n + b1_n + rg \times (gh_n + b2_n))
  $$

  $$
  hy = ng + zg \times (hx - ng)
  $$

  $$
  storage = [rg \mid zg \mid ng \mid hx \mid (gh_n + b2_n)]
  $$

- 公式变量：B为batch维大小，H为隐藏维大小；gi、gh、b1、b2分别为input_gates、hidden_gates、input_bias、hidden_bias按门序三等分后的分段；hx为上一步隐状态；hy为新隐状态；storage为反向复用中间量，五段[rg、zg、ng、hx、gh_n+b2_n]沿最后一维拼接，每段H列。input_bias与hidden_bias缺省（传入空指针）时等价于全零bias。
- 精度说明：FLOAT16、BFLOAT16输入在float32中间精度下完成sigmoid、tanh与乘加计算，计算结果舍回原数据类型；输出数据类型与输入一致。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnThnnFusedGruCellGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThnnFusedGruCell”接口执行计算。

```cpp
aclnnStatus aclnnThnnFusedGruCellGetWorkspaceSize(
  const aclTensor  *inputGates,
  const aclTensor  *hiddenGates,
  const aclTensor  *hx,
  const aclTensor  *inputBiasOptional,
  const aclTensor  *hiddenBiasOptional,
  aclTensor        *hyOut,
  aclTensor        *storageOut,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnThnnFusedGruCell(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnThnnFusedGruCellGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 320px">
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
      <td>inputGates（aclTensor*）</td>
      <td>输入</td>
      <td>输入侧门控预激活，对应公式中的input_gates，三等分后为gi_r、gi_z、gi_n。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>数据类型需为BFLOAT16、FLOAT16或FLOAT，且与全部输入输出一致。</li><li>shape需为（B, 3H），且与hiddenGates一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(B, 3H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hiddenGates（aclTensor*）</td>
      <td>输入</td>
      <td>隐层侧门控预激活，对应公式中的hidden_gates，三等分后为gh_r、gh_z、gh_n。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>数据类型与inputGates保持一致。</li><li>shape需与inputGates相同。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(B, 3H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hx（aclTensor*）</td>
      <td>输入</td>
      <td>上一步隐状态，对应公式中的hx。</td>
      <td><ul><li>不能传入空指针，支持空Tensor。</li><li>数据类型与inputGates保持一致。</li><li>shape需为（B, H），且满足inputGates.shape[1] == 3 × H。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputBiasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>输入侧bias，对应公式中的b1_r、b1_z、b1_n。</td>
      <td><ul><li>可选输入，传入空指针表示缺省，等价于全零bias。</li><li>非空时数据类型与inputGates保持一致。</li><li>非空时shape需为（3H,）。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(3H,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hiddenBiasOptional（aclTensor*）</td>
      <td>可选输入</td>
      <td>隐层侧bias，对应公式中的b2_r、b2_z、b2_n。</td>
      <td><ul><li>可选输入，传入空指针表示缺省，等价于全零bias。</li><li>非空时数据类型与inputGates保持一致。</li><li>非空时shape需为（3H,），且与inputBiasOptional同size。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(3H,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hyOut（aclTensor*）</td>
      <td>输出</td>
      <td>新隐状态，对应公式中的hy。</td>
      <td><ul><li>不能传入空指针，支持空Tensor（输入为空Tensor时输出为空Tensor）。</li><li>数据类型与输入保持一致。</li><li>shape需为（B, H），与hx相同。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storageOut（aclTensor*）</td>
      <td>输出</td>
      <td>反向复用中间量，对应公式中的storage。</td>
      <td><ul><li>不能传入空指针，支持空Tensor（输入为空Tensor时输出为空Tensor）。</li><li>数据类型与输入保持一致。</li><li>shape需为（B, 5H）。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>(B, 5H)</td>
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
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>inputGates、hiddenGates、hx、hyOut、storageOut或executor存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>输入的数据类型不在支持范围内，如数据类型不是BFLOAT16、FLOAT16、FLOAT之一，或各张量数据类型不一致。</td>
    </tr>
    <tr>
      <td>输入shape的rank不满足约束，如门控矩阵、hx、hyOut、storageOut的rank不等于2，或bias的rank不等于1。</td>
    </tr>
    <tr>
      <td>门控矩阵（inputGates与hiddenGates）的第二维不等于3H。</td>
    </tr>
    <tr>
      <td>bias（inputBiasOptional与hiddenBiasOptional）非空时元素个数不等于3H。</td>
    </tr>
  </tbody></table>

## aclnnThnnFusedGruCell

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
  </colgroup>
  <thead>
    <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnThnnFusedGruCellGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码。

## 约束说明

- 确定性说明：aclnnThnnFusedGruCell默认确定性实现。
- 输入与输出的数据类型必须一致，仅支持BFLOAT16、FLOAT16、FLOAT；不支持跨数据类型组合，也不支持DOUBLE、INT64等其它数据类型。
- inputGates与hiddenGates的shape必须相同且为（B, 3H），hx的shape为（B, H），需满足inputGates.shape[1] == 3 × hx.shape[1]；inputBiasOptional与hiddenBiasOptional非空时shape为（3H,）且两者同size。
- 输入与输出的数据格式仅支持ND。
- B=0或H=0（numel为0的空Tensor）为合法输入，直接返回空输出。
- 输入支持非连续Tensor（由框架自动连续化处理）；输出支持非连续Tensor（算子内部计算完成后由框架按输出布局自动拷贝写回）。

## 调用示例

调用示例代码如下，示例为FLOAT输入、可选bias传入空指针的场景，其余数据类型与bias非空场景的调用方式相同。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_thnn_fused_gru_cell.h"

#define CHECK_ACL(call)                                                          \
    do {                                                                         \
        aclError aclRet = (call);                                                \
        if (aclRet != ACL_SUCCESS) {                                             \
            fprintf(stderr, "ACL error %d at file %s line %d\n",                 \
                    aclRet, __FILE__, __LINE__);                                 \
            return 1;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_ACLNN(call)                                                        \
    do {                                                                         \
        aclnnStatus aclnnRet = (call);                                           \
        if (aclnnRet != OK) {                                                    \
            fprintf(stderr, "ACLNN error %d at file %s line %d\n",               \
                    aclnnRet, __FILE__, __LINE__);                               \
            return 1;                                                            \
        }                                                                        \
    } while (0)

int main()
{
    // 1. 初始化设备与Stream。
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. 构造输入输出：B=4，H=8，数据类型为FLOAT；
    //    门控矩阵shape为(B, 3H)，隐状态为(B, H)，storage为(B, 5H)。
    constexpr int64_t B = 4;
    constexpr int64_t H = 8;
    constexpr int64_t threeH = 3 * H;
    constexpr int64_t fiveH = 5 * H;
    const int64_t gateNum = B * threeH;
    const int64_t hNum = B * H;
    const int64_t stNum = B * fiveH;
    const size_t gateBytes = static_cast<size_t>(gateNum) * sizeof(float);
    const size_t hBytes = static_cast<size_t>(hNum) * sizeof(float);
    const size_t stBytes = static_cast<size_t>(stNum) * sizeof(float);

    std::vector<float> inputGatesHost(static_cast<size_t>(gateNum), 0.1f);
    std::vector<float> hiddenGatesHost(static_cast<size_t>(gateNum), 0.2f);
    std::vector<float> hxHost(static_cast<size_t>(hNum), 0.3f);
    std::vector<float> hyHost(static_cast<size_t>(hNum), 0.0f);
    std::vector<float> stHost(static_cast<size_t>(stNum), 0.0f);

    // 3. 在Device侧申请输入输出内存，并将输入拷贝到Device。
    void *devInputGates = nullptr;
    void *devHiddenGates = nullptr;
    void *devHx = nullptr;
    void *devHy = nullptr;
    void *devStorage = nullptr;
    CHECK_ACL(aclrtMalloc(&devInputGates, gateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devHiddenGates, gateBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devHx, hBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devHy, hBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devStorage, stBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(devInputGates, gateBytes, inputGatesHost.data(),
                          gateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devHiddenGates, gateBytes, hiddenGatesHost.data(),
                          gateBytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(devHx, hBytes, hxHost.data(), hBytes, ACL_MEMCPY_HOST_TO_DEVICE));

    // 4. 构造aclTensor描述符；可选bias传入空指针表示缺省（等价于全零bias）。
    const int64_t gateDims[2] = {B, threeH};
    const int64_t gateStrides[2] = {threeH, 1};
    const int64_t hxDims[2] = {B, H};
    const int64_t hxStrides[2] = {H, 1};
    const int64_t stDims[2] = {B, fiveH};
    const int64_t stStrides[2] = {fiveH, 1};

    aclTensor *inputGates = aclCreateTensor(gateDims, 2, ACL_FLOAT, gateStrides, 0,
                                            ACL_FORMAT_ND, gateDims, 2, devInputGates);
    aclTensor *hiddenGates = aclCreateTensor(gateDims, 2, ACL_FLOAT, gateStrides, 0,
                                             ACL_FORMAT_ND, gateDims, 2, devHiddenGates);
    aclTensor *hx = aclCreateTensor(hxDims, 2, ACL_FLOAT, hxStrides, 0,
                                    ACL_FORMAT_ND, hxDims, 2, devHx);
    aclTensor *inputBiasOptional = nullptr;
    aclTensor *hiddenBiasOptional = nullptr;
    aclTensor *hyOut = aclCreateTensor(hxDims, 2, ACL_FLOAT, hxStrides, 0,
                                    ACL_FORMAT_ND, hxDims, 2, devHy);
    aclTensor *storageOut = aclCreateTensor(stDims, 2, ACL_FLOAT, stStrides, 0,
                                          ACL_FORMAT_ND, stDims, 2, devStorage);

    // 5. 第一段接口：获取计算所需workspace大小与执行器。
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    CHECK_ACLNN(aclnnThnnFusedGruCellGetWorkspaceSize(inputGates, hiddenGates, hx,
                                                      inputBiasOptional, hiddenBiasOptional,
                                                      hyOut, storageOut, &workspaceSize,
                                                      &executor));

    // 6. 按需申请workspace内存，并调用第二段接口执行计算。
    void *workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACLNN(aclnnThnnFusedGruCell(workspace, workspaceSize, executor, stream));

    // 7. 同步Stream，并将计算结果拷回Host。
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(hyHost.data(), hBytes, devHy, hBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(stHost.data(), stBytes, devStorage, stBytes, ACL_MEMCPY_DEVICE_TO_HOST));
    printf("hyOut[0] = %f\n", hyHost[0]);

    // 8. 释放申请的资源。
    if (workspace != nullptr) {
        CHECK_ACL(aclrtFree(workspace));
    }
    CHECK_ACLNN(aclDestroyTensor(storageOut));
    CHECK_ACLNN(aclDestroyTensor(hyOut));
    CHECK_ACLNN(aclDestroyTensor(hx));
    CHECK_ACLNN(aclDestroyTensor(hiddenGates));
    CHECK_ACLNN(aclDestroyTensor(inputGates));
    CHECK_ACL(aclrtFree(devStorage));
    CHECK_ACL(aclrtFree(devHy));
    CHECK_ACL(aclrtFree(devHx));
    CHECK_ACL(aclrtFree(devHiddenGates));
    CHECK_ACL(aclrtFree(devInputGates));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(0));
    CHECK_ACL(aclFinalize());
    return 0;
}
```
