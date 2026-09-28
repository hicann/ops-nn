# aclnnThnnFusedGruCellBackward

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

- 接口功能：完成 GRU（门控循环单元）单时间步的融合反向梯度计算。输入上游梯度 `gradHy (B, H)` 与前向 `_thnn_fused_gru_cell` 保存的 `storage (B, 5H)`（行内按 `[r, z, n, hx, hn]` 五个 H 宽平面顺序存放），单 kernel 融合计算 5 路梯度输出。对标 PyTorch `aten::_thnn_fused_gru_cell_backward`。
- 计算公式：

  对每个 batch 行 $b \in [0, B)$、hidden 列 $j \in [0, H)$，从 `storage` 行内拆分五个平面（$h = H$）：

  $$r_{b,j} = \mathrm{storage}_{b,\ j},\quad z_{b,j} = \mathrm{storage}_{b,\ h+j},\quad n_{b,j} = \mathrm{storage}_{b,\ 2h+j},\quad hx_{b,j} = \mathrm{storage}_{b,\ 3h+j},\quad hn_{b,j} = \mathrm{storage}_{b,\ 4h+j}$$

  记 $go_{b,j} = \mathrm{grad\_hy}_{b,j}$，五条梯度链（$\sigma'(o)=o(1-o)$、$\tanh'(o)=1-o^2$）：

  $$\mathrm{gin}_{b,j} = go_{b,j} \cdot (1 - z_{b,j}) \cdot (1 - n_{b,j}^2) \tag{tanh\_backward}$$

  $$\mathrm{gig}_{b,j} = go_{b,j} \cdot (hx_{b,j} - n_{b,j}) \cdot (1 - z_{b,j}) \cdot z_{b,j} \tag{sigmoid\_backward}$$

  $$\mathrm{grg}_{b,j} = \mathrm{gin}_{b,j} \cdot hn_{b,j} \cdot (1 - r_{b,j}) \cdot r_{b,j} \tag{sigmoid\_backward}$$

  $$\mathrm{ghn}_{b,j} = \mathrm{gin}_{b,j} \cdot r_{b,j}$$

  $$\mathrm{ghx}_{b,j} = go_{b,j} \cdot z_{b,j}$$

  输出拼接（$\big\|$ 表沿 dim=1 拼接）与归约：

  $$\mathrm{grad\_input\_gates} = [\,\mathrm{grg} \;\big\|\; \mathrm{gig} \;\big\|\; \mathrm{gin}\,] \in \mathbb{R}^{B \times 3H}$$

  $$\mathrm{grad\_hidden\_gates} = [\,\mathrm{grg} \;\big\|\; \mathrm{gig} \;\big\|\; \mathrm{ghn}\,] \in \mathbb{R}^{B \times 3H}$$

  $$\mathrm{grad\_hx} = \mathrm{ghx} \in \mathbb{R}^{B \times H}$$

  $$\mathrm{grad\_input\_bias}_{k} = \sum_{b=0}^{B-1} \mathrm{grad\_input\_gates}_{b,k},\qquad \mathrm{grad\_hidden\_bias}_{k} = \sum_{b=0}^{B-1} \mathrm{grad\_hidden\_gates}_{b,k}$$

  `hasBias = false` 时两路 bias 输出为空张量 `(0,)`；此时两槽亦可传入空指针（表示不索取该输出）或任意张量（原样返回，不读不写）。

## 函数原型

每个算子分为两段式接口，必须先调用“aclnnThnnFusedGruCellBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnThnnFusedGruCellBackward”接口执行计算。

```cpp
aclnnStatus aclnnThnnFusedGruCellBackwardGetWorkspaceSize(
  const aclTensor *gradHy,
  const aclTensor *storage,
  bool            hasBias,
  const aclTensor *gradInputGatesOut,
  const aclTensor *gradHiddenGatesOut,
  const aclTensor *gradHxOut,
  const aclTensor *gradInputBiasOutOptional,
  const aclTensor *gradHiddenBiasOutOptional,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnThnnFusedGruCellBackward(
  void          *workspace,
  uint64_t      workspaceSize,
  aclOpExecutor *executor,
  aclrtStream   stream)
```

## aclnnThnnFusedGruCellBackwardGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>gradHy（const aclTensor*）</td>
      <td>输入</td>
      <td>上游梯度 ∂L/∂hy（前向输出 hy 的梯度），对应计算公式中 go。</td>
      <td><li>支持空Tensor。</li><li>数据类型须为FLOAT、FLOAT16或BFLOAT16，与storage数据类型一致。</li><li>支持非连续Tensor（框架自动归一为连续后计算）。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2，(B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>storage（const aclTensor*）</td>
      <td>输入</td>
      <td>前向 _thnn_fused_gru_cell 保存的 storage，行内按 [r, z, n, hx, hn] 五个 H 宽平面顺序存放，对应计算公式中 r、z、n、hx、hn。</td>
      <td><li>支持空Tensor。</li><li>数据类型与gradHy保持一致。</li><li>shape须满足 (B, 5H) 约束，即第二维为 gradHy 第二维（H）的 5 倍，B 维与 gradHy 一致。</li><li>支持非连续Tensor（框架自动归一为连续后计算）。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>2，(B, 5H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias（bool）</td>
      <td>输入</td>
      <td>表示前向 input_bias 是否定义，对应属性 has_bias。true 时输出两路 bias 梯度 (3H,)；false 时输出空张量 (0,)。</td>
      <td>取值为 true 或 false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInputGatesOut（const aclTensor*）</td>
      <td>输出</td>
      <td>input 侧门预激活梯度，行内 [grg, gig, gin] 三段拼接，对应计算公式中 grad_input_gates。</td>
      <td><li>支持空Tensor。</li><li>数据类型与gradHy保持一致。</li><li>shape为 (B, 3H)。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>2，(B, 3H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHiddenGatesOut（const aclTensor*）</td>
      <td>输出</td>
      <td>hidden 侧门预激活梯度，行内 [grg, gig, ghn] 三段拼接，对应计算公式中 grad_hidden_gates。</td>
      <td><li>支持空Tensor。</li><li>数据类型与gradHy保持一致。</li><li>shape为 (B, 3H)。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>2，(B, 3H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHxOut（const aclTensor*）</td>
      <td>输出</td>
      <td>隐状态直连梯度，对应计算公式中 grad_hx。</td>
      <td><li>支持空Tensor。</li><li>数据类型与gradHy保持一致。</li><li>shape为 (B, H)。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>2，(B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInputBiasOutOptional（const aclTensor*）</td>
      <td>输出</td>
      <td>grad_input_gates 沿 batch（axis 0）归约结果，对应计算公式中 grad_input_bias。hasBias=false 时为空张量 (0,)。</td>
      <td><li>hasBias=true 时必传且不能为空指针，shape为 (3H,)；B=0 时为全零 (3H,)。</li><li>hasBias=false 时可传空指针（推荐，表示不索取该输出）或任意张量（原样返回，不读不写）；张量形态时 shape为 (0,)。</li><li>数据类型与gradHy保持一致。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>1，(3H,) 或 (0,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradHiddenBiasOutOptional（const aclTensor*）</td>
      <td>输出</td>
      <td>grad_hidden_gates 沿 batch（axis 0）独立归约结果，对应计算公式中 grad_hidden_bias。hasBias=false 时为空张量 (0,)。</td>
      <td><li>hasBias=true 时必传且不能为空指针，shape为 (3H,)；B=0 时为全零 (3H,)。</li><li>hasBias=false 时可传空指针（推荐，表示不索取该输出）或任意张量（原样返回，不读不写）；张量形态时 shape为 (0,)。</li><li>数据类型与gradHy保持一致。</li></td>
      <td>数据类型与gradHy保持一致。</td>
      <td>ND</td>
      <td>1，(3H,) 或 (0,)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>UINT64</td>
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

    aclnnStatus：返回状态码，具体参见aclnn返回码。

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
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>gradHy、storage、gradInputGatesOut、gradHiddenGatesOut、gradHxOut、workspaceSize、executor存在空指针；或 hasBias=true 时 gradInputBiasOutOptional、gradHiddenBiasOutOptional 存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="4"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="4"> 161002 </td>
      <td>gradHy或storage的数据类型不在支持的范围之内（仅支持FLOAT、FLOAT16、BFLOAT16）。</td>
    </tr>
    <tr>
      <td>gradHy与storage的数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradHy或storage的维度不为2。</td>
    </tr>
    <tr>
      <td>storage的shape不满足 (B, 5H) 约束（B维与gradHy不一致或第二维不为5*H）。</td>
    </tr>
    <tr>
      <td> ACLNN_ERR_INNER </td>
      <td> 561000 </td>
      <td>当前芯片非 ascend950 系列（IsRegbase 门控拒绝：本算子仅支持 ascend950）。</td>
    </tr>
    </tbody></table>

## aclnnThnnFusedGruCellBackward

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
            <tr><td>workspaceSize </td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnThnnFusedGruCellBackwardGetWorkspaceSize获取。</td></tr>
            <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
            <tr><td>stream </td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
        </tbody>
    </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见aclnn返回码。

## 约束说明

- 确定性说明：aclnnThnnFusedGruCellBackward默认确定性实现。

- 数据类型：gradHy与storage的数据类型须一致，仅支持FLOAT、FLOAT16、BFLOAT16。所有输出数据类型与gradHy保持一致。

- 数据格式：输入和输出仅支持ND。

- 维度约束：gradHy与storage须为2维，且storage的shape须满足 (B, 5H)，即storage第二维为gradHy第二维（H）的5倍，B维与gradHy一致。

- 非连续Tensor：输入与输出均支持非连续Tensor——输入经Contiguous归一为连续buffer后交给kernel计算；输出经ViewCopy按用户提供的视图散射写回（空输出为no-op跳过）。

- 空Tensor：支持空Tensor输入。B=0（空batch）或H=0（空hidden）为合法退化——门梯度与gradHx输出为空；hasBias=true且B=0时两路bias输出为全零 (3H,)；hasBias=false时两路bias输出为空张量 (0,)（该形态下两槽亦可传空指针或任意张量，原样返回不读不写）。

- <term>Ascend 950PR&950DT系列产品</term>：仅支持该产品型号，不支持其他产品。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnn_thnn_fused_gru_cell_backward.h"

#define CHECK_ACL(x)                                                          \
    do {                                                                      \
        aclError __ret = (x);                                                 \
        if (__ret != ACL_SUCCESS) {                                           \
            fprintf(stderr, "ACL error %d at %s:%d\n", __ret, __FILE__, __LINE__); \
            return -1;                                                        \
        }                                                                     \
    } while (0)

#define CHECK_ACLNN(x)                                                        \
    do {                                                                      \
        aclnnStatus __ret = (x);                                              \
        if (__ret != 0) {                                                     \
            fprintf(stderr, "ACLNN error %d at %s:%d\n", __ret, __FILE__, __LINE__); \
            return -1;                                                        \
        }                                                                     \
    } while (0)

int main() {
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateStream(&stream));

    // 2. 构造输入与输出
    //    grad_hy (B, H) = (4, 8)，storage (B, 5H) = (4, 40)
    //    grad_input_gates/grad_hidden_gates (B, 3H) = (4, 24)
    //    grad_hx (B, H) = (4, 8)，bias (3H,) = (24,)
    int64_t B = 4, H = 8;
    int64_t gradHyDims[2] = {B, H};
    int64_t gradHyStrides[2] = {H, 1};
    int64_t storageDims[2] = {B, 5 * H};
    int64_t storageStrides[2] = {5 * H, 1};
    int64_t gatesDims[2] = {B, 3 * H};
    int64_t gatesStrides[2] = {3 * H, 1};
    int64_t biasDims[1] = {3 * H};
    int64_t biasStrides[1] = {1};
    size_t elemBytes = 4;  // FLOAT = 4 bytes

    void* devGradHy = nullptr;
    void* devStorage = nullptr;
    void* devGradInputGates = nullptr;
    void* devGradHiddenGates = nullptr;
    void* devGradHx = nullptr;
    void* devGradInputBias = nullptr;
    void* devGradHiddenBias = nullptr;
    CHECK_ACL(aclrtMalloc(&devGradHy, B * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devStorage, B * 5 * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradInputGates, B * 3 * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHiddenGates, B * 3 * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHx, B * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradInputBias, 3 * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc(&devGradHiddenBias, 3 * H * elemBytes, ACL_MEM_MALLOC_HUGE_FIRST));

    // 输入清零（实际使用时填入真实数据）
    CHECK_ACL(aclrtMemset(devGradHy, B * H * elemBytes, 0, B * H * elemBytes));
    CHECK_ACL(aclrtMemset(devStorage, B * 5 * H * elemBytes, 0, B * 5 * H * elemBytes));

    // 创建 aclTensor 描述符（2D/1D 连续 ND 布局）
    aclTensor* gradHy = aclCreateTensor(gradHyDims, 2, ACL_FLOAT, gradHyStrides, 0,
                                        ACL_FORMAT_ND, gradHyDims, 2, devGradHy);
    aclTensor* storage = aclCreateTensor(storageDims, 2, ACL_FLOAT, storageStrides, 0,
                                         ACL_FORMAT_ND, storageDims, 2, devStorage);
    aclTensor* gradInputGates = aclCreateTensor(gatesDims, 2, ACL_FLOAT, gatesStrides, 0,
                                                ACL_FORMAT_ND, gatesDims, 2, devGradInputGates);
    aclTensor* gradHiddenGates = aclCreateTensor(gatesDims, 2, ACL_FLOAT, gatesStrides, 0,
                                                 ACL_FORMAT_ND, gatesDims, 2, devGradHiddenGates);
    aclTensor* gradHx = aclCreateTensor(gradHyDims, 2, ACL_FLOAT, gradHyStrides, 0,
                                        ACL_FORMAT_ND, gradHyDims, 2, devGradHx);
    aclTensor* gradInputBias = aclCreateTensor(biasDims, 1, ACL_FLOAT, biasStrides, 0,
                                               ACL_FORMAT_ND, biasDims, 1, devGradInputBias);
    aclTensor* gradHiddenBias = aclCreateTensor(biasDims, 1, ACL_FLOAT, biasStrides, 0,
                                                ACL_FORMAT_ND, biasDims, 1, devGradHiddenBias);

    // 3. 调用第一段接口获取 workspace 大小和 executor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACLNN(aclnnThnnFusedGruCellBackwardGetWorkspaceSize(
        gradHy, storage, true, gradInputGates, gradHiddenGates, gradHx,
        gradInputBias, gradHiddenBias, &workspaceSize, &executor));

    // 根据 workspaceSize 申请 device 内存
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // 4. 调用第二段接口执行计算
    CHECK_ACLNN(aclnnThnnFusedGruCellBackward(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // 5. 释放资源
    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    aclDestroyTensor(gradHy);
    aclDestroyTensor(storage);
    aclDestroyTensor(gradInputGates);
    aclDestroyTensor(gradHiddenGates);
    aclDestroyTensor(gradHx);
    aclDestroyTensor(gradInputBias);
    aclDestroyTensor(gradHiddenBias);
    aclrtFree(devGradHy);
    aclrtFree(devStorage);
    aclrtFree(devGradInputGates);
    aclrtFree(devGradHiddenGates);
    aclrtFree(devGradHx);
    aclrtFree(devGradInputBias);
    aclrtFree(devGradHiddenBias);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
