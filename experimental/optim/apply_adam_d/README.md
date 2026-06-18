# ApplyAdamD

> 本算子提供 `aclnnApplyAdamDGetWorkspaceSize` / `aclnnApplyAdamD` 两段式 aclnn 调用接口，同时保留
> `REG_OP(ApplyAdamD)` 图模式调用形态。工程使用 `ACLNNTYPE aclnn_exclude`，公共头由
> `op_api/aclnn_apply_adam_d.h` 手写提供。
>
> 算子原型已标注 **DEPRECATED**（建议使用 `ApplyAdam`）。本 `experimental` 任务补 **ascend910b**
> （Atlas A2 系列，DAV_2201）端侧原生 AscendC kernel + tiling + aclnn 调用入口，沿用既有 IR
> （10 输入 / 3 输出 / 2 属性）的张量标量契约。

## 产品支持情况

| 产品 | 是否支持 |
|:-----|:--------:|
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>（ascend910b，DAV_2201） | √ |

> 本 `experimental` 任务新增并验证的是 **ascend910b（Atlas A2 系列）** 原生开源 AscendC kernel；
> 其余产品（Ascend 950PR/950DT、Atlas A3、Atlas 推理/训练系列等）由仓内既有实现（`optim/apply_adam_d` arch35 路径）支持，不在本任务范围。

## 功能说明

- **算子功能**：根据 Adam 算法对参数张量 `var` 做单步更新。本算子与 `ApplyAdam` 计算逻辑相同，但 "_d" 变体
  **除原地更新 `var` 外，额外输出更新后的 `m`、`v`**（`var`/`m`/`v` 为 ref 输入输出，与对应输入同址原地更新）。
- **第三方框架兼容**：TensorFlow `ApplyAdam`。
- **计算公式**（内部统一升精度到 FP32 计算，两分支分母均使用**更新后**的 `v`）：

  $$
  lr' = learning\_rate \times \frac{\sqrt{1 - beta2\_power}}{1 - beta1\_power}
  $$

  $$
  m_{new} = m + (1 - beta1) \times (grad - m) \quad\equiv\quad beta1 \times m + (1 - beta1) \times grad
  $$

  $$
  v_{new} = v + (1 - beta2) \times (grad \times grad - v) \quad\equiv\quad beta2 \times v + (1 - beta2) \times grad \times grad
  $$

  - 若 `use_nesterov = False`：

    $$
    var_{new} = var - lr' \times \frac{m_{new}}{\epsilon + \sqrt{v_{new}}}
    $$

  - 若 `use_nesterov = True`：

    $$
    var_{new} = var - lr' \times \frac{m_{new} \times beta1 + (1 - beta1) \times grad}{\epsilon + \sqrt{v_{new}}}
    $$

  其中 $\epsilon$ 为数值稳定小常数。BFLOAT16 / FLOAT16 输入在算子内部统一 Cast 升精度到 FLOAT32 计算，结果按 RINT 降回原数据类型。

## 参数说明

调用形态：`OpType = ApplyAdamD`，**10 个输入 / 3 个输出 / 2 个属性**，数据格式均为 ND，支持静态 / 动态 shape、动态 rank。

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
<col style="width: 110px">
<col style="width: 130px">
<col style="width: 300px">
<col style="width: 320px">
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
    <tr>
    <td>var</td>
    <td>输入（ref）</td>
    <td>待更新的参数张量，应来自Variable。与m/v/grad同shape。对应公式中var。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>m</td>
    <td>输入（ref）</td>
    <td>一阶矩估计，应来自Variable。与var同shape、同数据类型。对应公式中m。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>v</td>
    <td>输入（ref）</td>
    <td>二阶矩估计，应来自Variable。与var同shape、同数据类型。对应公式中v。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta1_power</td>
    <td>输入</td>
    <td>beta1的幂次，标量（shape为[1]或标量）。对应公式中beta1_power。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2_power</td>
    <td>输入</td>
    <td>beta2的幂次，标量。对应公式中beta2_power。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>lr</td>
    <td>输入</td>
    <td>学习率learning_rate，标量。对应公式中learning_rate。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta1</td>
    <td>输入</td>
    <td>一阶矩估计的衰减率，标量。对应公式中beta1。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>beta2</td>
    <td>输入</td>
    <td>二阶矩估计的衰减率，标量。对应公式中beta2。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>epsilon</td>
    <td>输入</td>
    <td>用于数值稳定性的小常数，标量。对应公式中ε。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>grad</td>
    <td>输入</td>
    <td>梯度张量，与var同shape、同数据类型。对应公式中grad。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>use_locking</td>
    <td>属性</td>
    <td>是否使用锁机制保护var、m、v的更新操作，默认为False。端侧实现不影响数值结果（仅图层语义）。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>use_nesterov</td>
    <td>属性</td>
    <td>是否使用Nesterov加速梯度，默认为False。决定var更新分支。</td>
    <td>Bool</td>
    <td>-</td>
    </tr>
    <tr>
    <td>var</td>
    <td>输出</td>
    <td>更新后的参数张量，与输入var同地址原地更新。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>m</td>
    <td>输出</td>
    <td>更新后的一阶矩估计，与输入m同地址原地更新。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
    <tr>
    <td>v</td>
    <td>输出</td>
    <td>更新后的二阶矩估计，与输入v同地址原地更新。</td>
    <td>BFLOAT16、FLOAT16、FLOAT</td>
    <td>ND</td>
    </tr>
</tbody></table>

## 约束说明

- **同形约束**：`var`、`m`、`v`、`grad` 必须具有**相同的 shape**（任意 rank；infershape 中输出 var/m/v 各自拷贝对应输入 shape）。
- **标量约束**：`beta1_power`、`beta2_power`、`lr`、`beta1`、`beta2`、`epsilon` 必须为**标量**（shape 为 `[1]` 或标量，`GetShapeSize() == 1`）。
- **同 dtype 约束**：全部 10 个输入与 3 个输出的**数据类型必须一致**（与 `var` 相同），取值范围 {BFLOAT16, FLOAT16, FLOAT}。
- **数据格式**：均为 ND。
- **数值约束**：`1 - beta1_power` 作分母须非 0（由调用方保证，算子不额外做零分母兜底）；`v` 在 Adam 语义下为非负，分母 `ε + sqrt(v_new)` 由 epsilon 提供数值稳定。
- **确定性**：纯 Elementwise 计算，无跨元素归约 / 原子累加，**默认确定性实现**（相同输入产生相同输出）。
- **废弃说明**：算子原型标注 deprecated（建议使用 `ApplyAdam`）；本任务仅补 ascend910b 端侧原生实现，不改变 IR 与功能定位。

## 调用说明

本算子提供 aclnn 两段式接口，并保留图模式和 kernel 直调示例（示例入口见 [`examples/run.sh`](./examples/run.sh)）：

| 调用方式 | 调用样例 | 说明 |
|----------|----------|------|
| aclnn 调用 | [test_aclnn_apply_adam_d.cpp](./examples/test_aclnn_apply_adam_d.cpp) | 通过 `aclnnApplyAdamDGetWorkspaceSize` / `aclnnApplyAdamD` 两段式接口调用，输入保持 10 个 Tensor（其中 6 个为 shape size 1 的标量 Tensor），输出 `varOut/mOut/vOut` 可与输入 `var/m/v` 复用 Device 地址实现原地更新。 |
| 图模式调用（标准形态） | [test_geir_apply_adam_d.cpp](./examples/test_geir_apply_adam_d.cpp) | 通过[算子 IR](./op_graph/experimental_apply_adam_d_proto.h)（`REG_OP(ApplyAdamD)`）构图（GE/IR）下发调用，演示 OpType=ApplyAdamD / 10 输入 / 3 输出 / 2 属性 的标准调用形态。端到端 GE 建图运行需先将 ApplyAdamD 自定义算子包部署到 OPP。 |
| kernel 直调（真机验证） | [test_kernel_direct_apply_adam_d.asc](./examples/test_kernel_direct_apply_adam_d.asc) | 通过 `<<<>>>` 直接调用真实内核 `NsApplyAdamD::ApplyAdamD<T, NES>`，在真实 NPU（ascend910b / dav-2201）上执行并与 host(FP32) golden 比对，覆盖 `use_nesterov=false/true` 两条分支。本环境下「示例编译运行通过」的真机验证路径。 |

构建 / 运行示例：

```bash
# 编译并运行 aclnn 示例 + 编译 test_geir（标准调用形态，编译检查）+ 编译并真机运行 kernel 直调示例
bash examples/run.sh

# 若已部署 ApplyAdamD 自定义算子包（设置 ASCEND_CUSTOM_OPP_PATH），可额外尝试端到端运行图模式示例
bash examples/run.sh --run-geir
```

> 运行环境记录：`test_kernel_direct_apply_adam_d.asc` 已在 **Atlas 910B3（NPU）** 上编译并跑通（var/m/v 均达 FP32 浮点社区精度阈值，max_rel < 1.22e-4）；
> `test_geir_apply_adam_d.cpp` 在本环境编译通过（标准调用形态），端到端 GE 建图运行依赖自定义算子包部署，默认仅做编译检查。

## 实现要点

- **范式**：Elementwise（同形张量 + 标量参数，逐元素独立计算）。
- **单一 tiling**：元素总数多核切分（`GetCoreNumAiv` 动态核数）+ UB 切分 + 双缓冲流水（CopyIn/Compute/CopyOut）。
- **TilingKey**：`D_T(fp16/bf16/fp32) × USE_NESTEROV(0/1)` 共 6 组合（模板编程 + `ASCENDC_TPL_SEL`）。
- **标量读取**：6 个标量为运行期 GM `[1]` 张量，kernel 在 `Init` 中 `DataCopyPad`→UB→`GetValue` 一次性读取（fp16/bf16 经 `Cast` 升精度），不入热路径。
- **边界处理**：尾块 / 非对齐由 `DataCopyPad`（按字节 blockLen）统一处理；空 Tensor / rank0 由 tiling + kernel `Process` 早退处理。
- **内部计算**：三种 dtype 均统一升精度到 FP32（fp16/bf16 经 Cast-up→FP32→Cast-down(RINT)），满足社区精度阈值。

## 目录结构

```
experimental/optim/apply_adam_d/
├── CMakeLists.txt              # SUPPORT_COMPUTE_UNIT=ascend910b, ACLNNTYPE aclnn_exclude
├── op_api/                     # aclnn_apply_adam_d.{h,cpp} + L0 apply_adam_d.{h,cpp}
├── op_host/                    # def / infershape / tiling
├── op_kernel/                  # apply_adam_d.cpp + apply_adam_d_kernel.h + tiling_data.h + tiling_key.h + CMakeLists
├── op_graph/                   # experimental_apply_adam_d_proto.h（REG_OP，承接现有 IR）
├── examples/                   # test_aclnn_*.cpp + test_geir_*.cpp + test_kernel_direct_*.asc + run.sh
└── docs/                       # aclnnApplyAdamD.md 接口文档
```

## 构建

```bash
bash build.sh --pkg --experimental --soc=ascend910b --ops=apply_adam_d -j16
```
