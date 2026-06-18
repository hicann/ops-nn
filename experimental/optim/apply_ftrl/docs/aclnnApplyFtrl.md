# aclnnApplyFtrl

> **接口状态声明（务必先读）**
> `ApplyFtrl` 是 **GE 图模式 / TensorFlow 算子**（对标 `tf.raw_ops.ApplyFtrl`），**CANN 当前未部署对外的 aclnn 接口**。其接口真值源为 `op_graph/apply_ftrl_proto.h`（GE IR `REG_OP(ApplyFtrl)`）。
> 本文档：
> - 「GE 图模式接口（真值源）」章节记录**实际下发签名**（输入/输出/属性），以 proto 为准；
> - 「派生 aclnn 两段式接口」章节为**本任务（experimental/optim/apply_ftrl，registry-invoke 自带 kernel）新增的派生封装原型**，**非 CANN 已部署接口**，最终签名以 1.3 方案设计 + L0 实现为准。

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

> **本任务（experimental）范围**：仅新增 **ascend910b（Atlas A2/A3，DAV_2201）** 原生 AscendC kernel。ascend950（arch35）路径由 mainline `optim/apply_ftrl` 已提供。上表反映 `ApplyFtrl` 算子整体（mainline + 本扩展）的产品支持。

## 功能说明

- 接口功能：根据 FTRL-proximal（Follow-The-Regularized-Leader）方案**原地更新**参数张量 `var`，并同步原地更新梯度平方累加器 `accum` 与校正项 `linear`。常用于带 L1/L2 正则的在线学习 / 大规模稀疏特征（推荐、广告 CTR）模型的参数更新。功能对标 `tf.raw_ops.ApplyFtrl`。
- 计算公式（逐元素；下式右侧 `accum/linear/var` 为更新前值，`var` 的更新使用更新后的 `linear`）：

  $$
  accum\_new = accum + grad \times grad
  $$

  $$
  linear = linear + grad - \frac{accum\_new^{-lr\_power} - accum^{-lr\_power}}{lr} \times var
  $$

  $$
  quadratic = \frac{accum\_new^{-lr\_power}}{lr} + 2 \times l2
  $$

  $$
  var =
  \begin{cases}
  \dfrac{sign(linear) \times l1 - linear}{quadratic}, & |linear| > l1 \\[2ex]
  0, & |linear| \le l1
  \end{cases}
  $$

  $$
  accum = accum\_new
  $$

  其中 $sign(x) \in \{-1, 0, +1\}$；$var$、$accum$、$linear$ 三者均原地写回。

## GE 图模式接口（真值源）

ApplyFtrl 通过 GE IR 构图下发。算子原型（`op_graph/apply_ftrl_proto.h`）：

```cpp
REG_OP(ApplyFtrl)
    .INPUT(var,      TensorType::NumberType())
    .INPUT(accum,    TensorType::NumberType())
    .INPUT(linear,   TensorType::NumberType())
    .INPUT(grad,     TensorType::NumberType())
    .INPUT(lr,       TensorType::NumberType())
    .INPUT(l1,       TensorType::NumberType())
    .INPUT(l2,       TensorType::NumberType())
    .INPUT(lr_power, TensorType::NumberType())
    .OUTPUT(var,     TensorType::NumberType())
    .ATTR(use_locking, Bool, false)
    .OP_END_FACTORY_REG(ApplyFtrl)
```

- **参数说明**

  <table style="table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 140px">
  <col style="width: 140px">
  <col style="width: 400px">
  <col style="width: 280px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr><td>var</td><td>输入（Ref，原地更新）</td><td>待更新参数张量，应来自Variable。对应公式var。shape与accum/linear/grad一致。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>accum</td><td>输入（Ref，原地更新）</td><td>梯度平方累加器，应来自Variable。对应公式accum，要求各元素≥0。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>linear</td><td>输入（Ref，原地更新）</td><td>校正项，应来自Variable。对应公式linear。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>grad</td><td>输入</td><td>当前步梯度张量。对应公式grad。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>lr</td><td>输入</td><td>学习率（缩放因子），标量（0-D或1元素）。对应公式lr，要求≠0。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>l1</td><td>输入</td><td>L1正则化系数，标量。对应公式l1。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>l2</td><td>输入</td><td>L2正则化系数，标量。对应公式l2。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>lr_power</td><td>输入</td><td>缩放因子的幂次，标量。对应公式lr_power。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
    <tr><td>use_locking</td><td>属性</td><td>是否使用锁机制保护更新操作，默认False。仅支持False。</td><td>Bool</td><td>-</td></tr>
    <tr><td>var</td><td>输出</td><td>更新后的参数张量，与输入var共享存储。算子仅声明此1个输出端口；accum、linear经各自input ref端口原地更新。</td><td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td></tr>
  </tbody></table>

- 图模式构图调用样例：[test_geir_apply_ftrl.cpp](../../../../optim/apply_ftrl/examples/test_geir_apply_ftrl.cpp)（按 `var → accum → linear → grad → lr → l1 → l2 → lr_power` 串接 8 个 `Data` 输入，scalar 输入 shape `{1}`，`set_attr_use_locking(false)`）。

## 派生 aclnn 两段式接口（本任务新增，标准 CANN 未部署）

> 该派生封装**已在本实验扩展落地**：手写 op_api（`op_api/aclnn_apply_ftrl.{h,cpp}` + L0 `op_api/apply_ftrl.{h,cpp}`，`ACLNNTYPE=aclnn_exclude`），随 `cann-ops-nn-custom_*.run` 自定义算子包部署后，`aclnnApplyFtrlGetWorkspaceSize` / `aclnnApplyFtrl` 符号在 `${ASCEND_CUSTOM_OPP_PATH}/op_api/lib/libcust_opapi.so` 中以 `extern "C"` 默认可见性导出，可被 ATK pyaclnn / 链接调用（标准 CANN OPP 仍不含该接口）。下列签名即最终实现签名（与 L0 实现一致）。算子分为两段式接口：先调用 `aclnnApplyFtrlGetWorkspaceSize` 获取 workspace 大小及执行器，再调用 `aclnnApplyFtrl` 执行计算。

```cpp
aclnnStatus aclnnApplyFtrlGetWorkspaceSize(
  aclTensor        *varRef,
  aclTensor        *accumRef,
  aclTensor        *linearRef,
  const aclTensor  *grad,
  const aclTensor  *lr,
  const aclTensor  *l1,
  const aclTensor  *l2,
  const aclTensor  *lrPower,
  bool              useLocking,
  uint64_t         *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnApplyFtrl(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

### aclnnApplyFtrlGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 220px">
  <col style="width: 80px">
  <col style="width: 80px">
  <col style="width: 90px">
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
      <td>varRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>待更新参数，原地更新，对应公式var。</td>
      <td><ul><li>支持空Tensor。</li><li>与accumRef/linearRef/grad数据类型、shape需完全一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-8</td><td>×</td>
    </tr>
    <tr>
      <td>accumRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>梯度平方累加器，原地更新，对应公式accum（要求≥0）。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-8</td><td>×</td>
    </tr>
    <tr>
      <td>linearRef（aclTensor*）</td>
      <td>输入/输出</td>
      <td>校正项，原地更新，对应公式linear。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-8</td><td>×</td>
    </tr>
    <tr>
      <td>grad（aclTensor*）</td>
      <td>输入</td>
      <td>当前步梯度，对应公式grad。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型、shape与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-8</td><td>×</td>
    </tr>
    <tr>
      <td>lr（aclTensor*）</td>
      <td>输入</td>
      <td>学习率标量，对应公式lr（要求≠0）。</td>
      <td><ul><li>0-D或1元素1-D。</li><li>数据类型与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-1</td><td>-</td>
    </tr>
    <tr>
      <td>l1（aclTensor*）</td>
      <td>输入</td>
      <td>L1正则系数标量，对应公式l1。</td>
      <td><ul><li>0-D或1元素1-D。</li><li>数据类型与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-1</td><td>-</td>
    </tr>
    <tr>
      <td>l2（aclTensor*）</td>
      <td>输入</td>
      <td>L2正则系数标量，对应公式l2。</td>
      <td><ul><li>0-D或1元素1-D。</li><li>数据类型与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-1</td><td>-</td>
    </tr>
    <tr>
      <td>lrPower（aclTensor*）</td>
      <td>输入</td>
      <td>缩放因子幂次标量，对应公式lr_power。</td>
      <td><ul><li>0-D或1元素1-D。</li><li>数据类型与varRef一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td><td>ND</td><td>0-1</td><td>-</td>
    </tr>
    <tr>
      <td>useLocking（bool）</td>
      <td>输入</td>
      <td>是否加锁，对应属性use_locking。仅支持false。</td>
      <td>仅支持传入false。</td>
      <td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需在Device侧申请的workspace大小。</td>
      <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程。</td>
      <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>：是否支持 BFLOAT16 以最终实现及配套为准。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
  </colgroup>
  <thead>
    <tr><th>返回值</th><th>错误码</th><th>描述</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>varRef、accumRef、linearRef、grad、lr、l1、l2、lrPower存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>输入数据类型不在支持范围（BFLOAT16/FLOAT16/FLOAT）之内。</td>
    </tr>
    <tr><td>各输入数据类型不一致，或lr/l1/l2/lrPower非标量（0-D或1元素）。</td></tr>
    <tr><td>var/accum/linear/grad的shape不一致。</td></tr>
  </tbody></table>

### aclnnApplyFtrl

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
    <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由aclnnApplyFtrlGetWorkspaceSize获取。</td></tr>
    <tr><td>executor</td><td>输入</td><td>op执行器，包含算子计算流程。</td></tr>
    <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
  </tbody></table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnApplyFtrl 默认确定性实现（Elementwise 逐元素独立计算，无跨元素/跨核归约）。
- **数据类型**：var/accum/linear/grad/lr/l1/l2/lrPower 与输出 var 的数据类型必须完全一致，取值 ∈ {BFLOAT16, FLOAT16, FLOAT}；无类型推导，无混合精度入参。
- **shape**：var/accum/linear/grad 的 shape 必须完全一致；lr/l1/l2/lrPower 必须为 0-D 或 1 元素 1-D 标量。
- **原地更新**：var、accum、linear 均为 Ref Tensor，算子执行后原地更新；调用方需将其视为可被算子修改的存储。
- **值域前置条件**（算子内不做运行时值域校验，由调用方保证）：`accum ≥ 0`、`lr ≠ 0`。当 `accum_new = 0` 且 `lr_power > 0`、或 `quadratic = 0`、或 `lr = 0` 时，幂/除法会产生 Inf/NaN，行为与 TensorFlow 原生 `ApplyFtrl` 一致，需由上游调用方规避。
- **属性**：use_locking 仅支持 false。
- 内部计算：FLOAT16 / BFLOAT16 输入在算子内部 Cast 为 FLOAT32 完成中间计算，再 Cast 回原数据类型输出。

## 调用示例

> **标准 CANN OPP 未部署 ApplyFtrl 的 aclnn 接口**（ACLNNTYPE=`aclnn_exclude`），也无 PyTorch 等价；但本实验扩展通过手写 op_api 提供了派生 aclnn 封装，安装本算子的自定义算子包（`cann-ops-nn-custom_*.run`）后，`aclnnApplyFtrl` / `aclnnApplyFtrlGetWorkspaceSize` 符号即在 `libcust_opapi.so` 中可链接调用（ATK pyaclnn 测试经此使能）。本节给出本实验算子目录 `examples/` 下**真实可运行**的两种调用样例，以及上文「派生 aclnn 两段式接口」的调用骨架。

### 可运行样例（指向 examples/）

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| GE 图模式（静态/动态 shape） | [../examples/test_geir_apply_ftrl.cpp](../examples/test_geir_apply_ftrl.cpp) | 通过[算子 IR](../../../../optim/apply_ftrl/op_graph/apply_ftrl_proto.h) 构图下发 `ApplyFtrl`：按 `var → accum → linear → grad → lr → l1 → l2 → lr_power` 串接 8 个 `Data` 输入（scalar 输入 shape `{1}`），`set_attr_use_locking(false)`。运行需先把算子编译安装进 OPP 包：`bash build.sh --pkg --experimental --soc=ascend910b --ops=apply_ftrl` → `bash build.sh --run_example apply_ftrl graph`。 |
| Kernel 直调（`<<<>>>`，**已在 ascend910b 真实 NPU 验证 PASS**） | [../examples/test_kernel_direct_apply_ftrl.asc](../examples/test_kernel_direct_apply_ftrl.asc) | 直接复用本算子 `op_kernel/apply_ftrl_kernel.h` 的 `NsApplyFtrl::ApplyFtrl<T, PAD_TAIL, HAS_L1>` 计算类（与正式构建同一份 kernel 源码），经 CUDA 风格 `<<<>>>` 在真实 NPU 上运行并与 fp64 CPU golden 比对三输出 var/accum/linear。一键编译 + 运行见 [../examples/run.sh](../examples/run.sh)。更完整的多 dtype / 多 shape 真实 NPU 验证见 [../tests/st/aclnnApplyFtrl/atk_ApplyFtrl.json](../tests/st/aclnnApplyFtrl/atk_ApplyFtrl.json)。 |

### 派生 aclnn 两段式调用骨架（本任务新增派生，标准 CANN 未部署）

> 「派生 aclnn 两段式接口」章节签名的**调用顺序示意**（参照同族 `experimental/optim/apply_proximal_adagrad/examples/arch35/test_aclnn_apply_proximal_adagrad.cpp` 模式）。安装本扩展自定义算子包后符号可链接（`libcust_opapi.so`，需 `export LD_LIBRARY_PATH=${ASCEND_CUSTOM_OPP_PATH}/op_api/lib:$LD_LIBRARY_PATH`）；未安装时不可链接，请用上表 GE 图模式 / Kernel 直调样例。

```cpp
// 注意：aclnnApplyFtrl / aclnnApplyFtrlGetWorkspaceSize 为派生原型，非 CANN 已部署接口，不保证可链接。

// 1. 第一段：计算 workspace 大小并构造执行器
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
auto ret = aclnnApplyFtrlGetWorkspaceSize(
    varRef, accumRef, linearRef,   // 3 个 Ref Tensor（原地更新）
    grad, lr, l1, l2, lrPower,     // grad 张量 + 4 个标量（0-D 或 1 元素 1-D）
    /*useLocking=*/false,          // 仅支持 false
    &workspaceSize, &executor);
// CHECK_RET(ret == ACL_SUCCESS, ...);

// 2. 按需申请 device workspace
void* workspace = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

// 3. 第二段：执行计算（var/accum/linear 原地更新）
ret = aclnnApplyFtrl(workspace, workspaceSize, executor, stream);
// CHECK_RET(ret == ACL_SUCCESS, ...);
aclrtSynchronizeStream(stream);
```
