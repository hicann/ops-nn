# ReluGradV4

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 算子功能：对 `gradOutput` 和 `self` 执行 ReLU 反向梯度计算。

计算公式：

$$
\operatorname{threshold\_backward}(gradOutput, self, threshold) =
\begin{cases}
gradOutput, & self \ne 0 \\
0, & self = 0
\end{cases}
$$

其中 `self` 为 uint8 的 ReLU 前向激活掩码（取值 {0, 1}）。

- 目录 `experimental/activation/relu_grad_v4` 对外导出 `aclnnReluGradV4` 两段式 ACLNN 接口。
- `op_host/op_api/aclnn_relu_grad_v4.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/relu_grad_v4.h` 和 `op_host/op_api/relu_grad_v4.cpp` 提供内部 `l0op::ReluGradV4` 封装，当前由 ACLNN 接口直接调用。
- 当前实现仅接受 `threshold == 0`，与 ReLU backward 语义保持一致。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental ReluGradV4 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnReluGradV4GetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    const aclScalar *threshold,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnReluGradV4(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnReluGradV4.md](docs/aclnnReluGradV4.md)。

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
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT8、UINT8、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>ReLU 前向激活掩码（取值 {0, 1}），固定为 uint8。</td>
    <td>UINT8</td>
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
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT8、UINT8、INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `self`（mask）的 dtype 必须为 `UINT8`；`gradOutput` 和 `out` 的 dtype 必须完全一致。
- `gradOutput`、`self` 和 `out` 的 shape 必须完全一致。
- `threshold` 必须是值为 `0` 的标量。
- `gradOutput` / `out` 仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`、`INT8`、`UINT8`、`INT32`。
- 支持 0 到 8 维 Tensor。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。
- 所有 dtype 均按 `mask != 0 ? gradOutput : 0` 计算（`mask` 为 uint8 掩码）。
- `FLOAT16` 和 `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_relu_grad_v4.cpp](examples/test_aclnn_relu_grad_v4.cpp) | `aclnnReluGradV4` 两段式调用示例。 |
| [docs/aclnnReluGradV4.md](docs/aclnnReluGradV4.md) | `aclnnReluGradV4` 接口文档。 |
| [tests/ut/op_api/test_aclnn_relu_grad_v4.cpp](tests/ut/op_api/test_aclnn_relu_grad_v4.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnReluGradV4/all_aclnnReluGradV4.json](tests/st/aclnnReluGradV4/all_aclnnReluGradV4.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnReluGradV4/executor_aclnnReluGradV4.py](tests/st/aclnnReluGradV4/executor_aclnnReluGradV4.py) | ATK CPU benchmark 执行器。 |

## Example 运行

加载 CANN 环境后，使用代码仓标准方式编译并运行样例：

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --run_example relu_grad_v4 eager cust --vendor_name=custom --experimental
```

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=relu_grad_v4 -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd /root/src/testcase
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c ./experimental/activation/relu_grad_v4/tests/st/aclnnReluGradV4/all_aclnnReluGradV4.json \
  -p ./experimental/activation/relu_grad_v4/tests/st/aclnnReluGradV4/executor_aclnnReluGradV4.py
```

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| :--- | :--- | :--- | :--- | :--- |
| HKUST复仇者 | HKUST复仇者 | ReluGradV4 | 2026-07 | 新增 ReluGradV4 算子的 Ascend C 实现，支持 float32/float16/bfloat16/int8/int32/uint8。 |
