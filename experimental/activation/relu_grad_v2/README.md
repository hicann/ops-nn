# ThresholdBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | √ |
| <term>Atlas 训练系列产品</term> | √ |

## 功能说明

- 算子功能：对 `gradOutput` 和 `self` 执行 ReLU 反向梯度计算。

计算公式：

$$
\operatorname{threshold\_backward}(gradOutput, self, threshold) =
\begin{cases}
gradOutput, & self > 0 \\
gradOutput, & self = \mathrm{NaN}\ \text{且 dtype 为浮点类型} \\
0, & self \le 0
\end{cases}
$$

- 目录 `experimental/activation/relu_grad_v2` 对外导出 `aclnnThresholdBackward` 两段式 ACLNN 接口。
- `op_host/op_api/aclnn_threshold_backward.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/relu_grad_v2.h` 和 `op_host/op_api/relu_grad_v2.cpp` 提供内部 `l0op::ReluGradV2` 封装，当前由 ACLNN 接口直接调用。
- 当前实现仅接受 `threshold == 0`，与 ReLU backward 语义保持一致。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental ThresholdBackward 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclScalar *threshold,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnThresholdBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnThresholdBackward.md](docs/aclnnThresholdBackward.md)。

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
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT8、UINT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>前向输入张量，用于生成 ReLU 掩码。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT8、UINT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>threshold</td>
    <td>输入</td>
    <td>阈值标量。当前实现仅接受值为 `0` 的 `INT32` 标量。</td>
    <td>INT32</td>
    <td>Scalar</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>计算得到的输出梯度张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT8、UINT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `gradOutput`、`self` 和 `out` 的 dtype 必须完全一致。
- `gradOutput`、`self` 和 `out` 的 shape 必须完全一致。
- `threshold` 必须是值为 `0` 的标量。
- 输入仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`、`INT8`、`UINT8`、`INT32`、`INT64`。
- 支持 0 到 8 维 Tensor。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。
- `FLOAT` 路径遵循 PyTorch `threshold_backward` 的标量语义，当 `self` 为 `NaN` 时保留 `gradOutput`。
- `FLOAT16` 和 `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。
- `INT8`、`UINT8`、`INT32` 和 `INT64` 路径遵循 `self > 0 ? gradOutput : 0`。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_relu_grad_v2.cpp](examples/test_aclnn_relu_grad_v2.cpp) | `aclnnThresholdBackward` 两段式调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnThresholdBackward.md](docs/aclnnThresholdBackward.md) | `aclnnThresholdBackward` 接口文档。 |
| [tests/ut/op_api/test_aclnn_threshold_backward.cpp](tests/ut/op_api/test_aclnn_threshold_backward.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnThresholdBackward/all_aclnnThresholdBackward.json](tests/st/aclnnThresholdBackward/all_aclnnThresholdBackward.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnThresholdBackward/executor_aclnnThresholdBackward.py](tests/st/aclnnThresholdBackward/executor_aclnnThresholdBackward.py) | ATK CPU benchmark 执行器。 |

## Example 运行

先确保 custom run 包已经安装，并加载 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd <ops-nn-repo>/experimental/activation/relu_grad_v2/examples
bash run.sh
```

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=relu_grad_v2 -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd /root/src/testcase
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c ./experimental/activation/relu_grad_v2/tests/st/aclnnThresholdBackward/all_aclnnThresholdBackward.json \
  -p ./experimental/activation/relu_grad_v2/tests/st/aclnnThresholdBackward/executor_aclnnThresholdBackward.py
```
