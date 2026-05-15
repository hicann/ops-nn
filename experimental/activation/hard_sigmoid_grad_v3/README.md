# HardsigmoidBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：对 `gradOutput` 和 `self` 执行 HardSigmoid 反向梯度计算。

计算公式：

$$
\operatorname{hardsigmoid\_backward}(gradOutput, self)=
\begin{cases}
\frac{gradOutput}{6}, & -3 < self < 3 \\
0, & \text{otherwise}
\end{cases}
$$

- 目录 `experimental/activation/hard_sigmoid_grad_v3` 对外导出 `aclnnHardsigmoidBackward` 两段式 ACLNN 接口。
- `op_host/op_api/aclnn_hardsigmoid_backward.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/hard_sigmoid_grad_v3.h` 和 `op_host/op_api/hard_sigmoid_grad_v3.cpp` 提供内部 `l0op::HardSigmoidGradV3` 封装，当前由 ACLNN 接口直接调用。
- `gradOutput` 和 `self` 允许混合精度输入，接口内部先按 `PromoteType(gradOutput, self)` 对齐计算 dtype。
- 当提升后的 dtype 为 `FLOAT16` 时，接口内部会进一步提升到 `FLOAT32` 计算，再 cast 回输出 dtype。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental HardsigmoidBackward 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnHardsigmoidBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnHardsigmoidBackward.md](docs/aclnnHardsigmoidBackward.md)。

## 参数说明

<table style="undefined;table-layout: fixed; width: 1393px"><colgroup>
<col style="width: 171px">
<col style="width: 115px">
<col style="width: 260px">
<col style="width: 220px">
<col style="width: 200px">
<col style="width: 104px">
</colgroup>
<thead>
  <tr>
    <th>参数名</th>
    <th>输入/输出</th>
    <th>描述</th>
    <th>数据类型</th>
    <th>数据格式</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>gradOutput</td>
    <td>输入</td>
    <td>上游梯度张量。</td>
    <td>FLOAT、FLOAT16；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>前向输入张量，用于生成 HardSigmoid 反向掩码。</td>
    <td>FLOAT、FLOAT16；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>计算得到的输入梯度张量。</td>
    <td>FLOAT、FLOAT16；BFLOAT16 仅支持 Ascend910B 及后续同代 SoC</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`。
- `BFLOAT16` 仅在 `Ascend910B` 及后续同代 SoC 上支持。
- `gradOutput` 和 `self` 允许混合精度输入，内部按 `PromoteType(gradOutput, self)` 选择计算 dtype。
- 当提升后的 dtype 为 `FLOAT16` 时，内部会进一步提升为 `FLOAT32` 计算。
- `out` 的 dtype 需要支持从内部计算 dtype cast 回写。
- `gradOutput`、`self` 和 `out` 的 shape 必须完全一致。
- 支持 0 到 8 维 Tensor。
- 支持标量输入，host 侧会通过 `EnsureNotScalar` 做适配。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。

## 当前实现要点

- host tiling 采用 512B cache line 对齐的核间切分，并限制 `ubFactor <= blockFactor`，避免中等 shape 整块退化到 tail 标量路径。
- `float16/float32` full-tile 路径使用 `Abs(x) < 3`，将双 compare 收敛为单次 `Abs + CompareScalar + Select`。
- `bfloat16` full-tile 路径保持升精度到 `float32` 计算；tail 路径继续使用标量安全实现。
- mixed precision 语义只在 ACLNN 封装层处理，AscendC kernel 始终接收对齐后的单一 dtype 输入。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_hard_sigmoid_grad_v3.cpp](examples/test_aclnn_hard_sigmoid_grad_v3.cpp) | `aclnnHardsigmoidBackward` 两段式调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnHardsigmoidBackward.md](docs/aclnnHardsigmoidBackward.md) | `aclnnHardsigmoidBackward` 接口文档。 |
| [tests/ut/op_api/test_aclnn_hardsigmoid_backward.cpp](tests/ut/op_api/test_aclnn_hardsigmoid_backward.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnHardsigmoidBackward/all_aclnnHardsigmoidBackward.json](tests/st/aclnnHardsigmoidBackward/all_aclnnHardsigmoidBackward.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnHardsigmoidBackward/executor_aclnnHardsigmoidBackward.py](tests/st/aclnnHardsigmoidBackward/executor_aclnnHardsigmoidBackward.py) | ATK CPU benchmark 执行器。 |

## Example 运行

先确保 custom run 包已经安装，并加载 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd <ops-nn-repo>/experimental/activation/hard_sigmoid_grad_v3/examples
bash run.sh
```

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=hard_sigmoid_grad_v3 -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd <testcase-repo>
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c <ops-nn-repo>/experimental/activation/hard_sigmoid_grad_v3/tests/st/aclnnHardsigmoidBackward/all_aclnnHardsigmoidBackward.json \
  -p <ops-nn-repo>/experimental/activation/hard_sigmoid_grad_v3/tests/st/aclnnHardsigmoidBackward/executor_aclnnHardsigmoidBackward.py
```
