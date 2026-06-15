# HardSigmoidV2

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

- 算子功能：对输入 Tensor 完成 HardSigmoidV2 激活计算。

计算公式：

$$
\operatorname{hard\_sigmoid}(x)=
\begin{cases}
0, & x \le -3 \\
\frac{x}{6} + \frac{1}{2}, & -3 < x < 3 \\
1, & x \ge 3
\end{cases}
$$

- 目录 `experimental/activation/hard_sigmoid_v2` 对外导出 `aclnnHardsigmoidV2` 和 `aclnnInplaceHardsigmoidV2` 两段式 ACLNN 接口。
- `op_host/op_api/aclnn_hardsigmoid_v2.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/hard_sigmoid_v2.h` 和 `op_host/op_api/hard_sigmoid_v2.cpp` 提供内部 `l0op::HardSigmoidV2` 封装，当前由 ACLNN 接口直接调用。
- `INT32` 路径严格对齐当前 AscendC kernel 的已验证整数语义，输出只会是 `0/1`。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |
| ACLNN 原地调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental HardSigmoidV2 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnHardsigmoidV2GetWorkspaceSize(
    const aclTensor *self,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnHardsigmoidV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

原地版本接口如下：

```cpp
aclnnStatus aclnnInplaceHardsigmoidV2GetWorkspaceSize(
    const aclTensor *self,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnInplaceHardsigmoidV2(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnHardsigmoidV2&aclnnInplaceHardsigmoidV2.md](docs/aclnnHardsigmoidV2&aclnnInplaceHardsigmoidV2.md)。

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
    <td>self</td>
    <td>输入</td>
    <td>待进行 HardSigmoid 计算的输入张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>非原地接口的计算结果张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>输入/输出</td>
    <td>原地接口的输入输出张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）、INT32</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`、`INT32`。
- `BFLOAT16` 仅在 `Ascend910B` 及后续同代 SoC 上支持。
- 非原地接口中输入和输出的 dtype 必须一致。
- 非原地接口中输入和输出的 shape 必须完全一致。
- 支持 0 到 8 维 Tensor。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。
- `FLOAT16` 路径直接在半精度上计算。
- `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。
- `INT32` 路径按已验证整数语义执行，最终输出只会是 `0/1`。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_hard_sigmoid_v2.cpp](examples/test_aclnn_hard_sigmoid_v2.cpp) | `aclnnHardsigmoidV2` 两段式调用示例。 |
| [examples/test_aclnn_inplace_hard_sigmoid_v2.cpp](examples/test_aclnn_inplace_hard_sigmoid_v2.cpp) | `aclnnInplaceHardsigmoidV2` 两段式原地调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnHardsigmoidV2&aclnnInplaceHardsigmoidV2.md](docs/aclnnHardsigmoidV2&aclnnInplaceHardsigmoidV2.md) | `aclnnHardsigmoidV2` / `aclnnInplaceHardsigmoidV2` 接口文档。 |
| [tests/ut/op_api/test_aclnn_hardsigmoid_v2.cpp](tests/ut/op_api/test_aclnn_hardsigmoid_v2.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnHardsigmoidV2/all_aclnnHardsigmoidV2.json](tests/st/aclnnHardsigmoidV2/all_aclnnHardsigmoidV2.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnHardsigmoidV2/executor_aclnnHardsigmoidV2.py](tests/st/aclnnHardsigmoidV2/executor_aclnnHardsigmoidV2.py) | ATK CPU benchmark 执行器。 |

## Example 运行

先确保 custom run 包已经安装，并加载 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd <ops-nn-repo>/experimental/activation/hard_sigmoid_v2/examples
bash run.sh
```

`examples/run.sh` 会优先使用 `CANN_INCLUDE_DIR`，否则按当前主机架构自动选择 `${ASCEND_HOME_PATH:-/usr/local/Ascend/cann}/<arch>-linux/include`。如果本机的 CANN 安装目录布局不同，执行前显式导出 `CANN_INCLUDE_DIR` 即可。

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=hard_sigmoid_v2 -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd /root/src/testcase
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c <ops-nn-repo>/experimental/activation/hard_sigmoid_v2/tests/st/aclnnHardsigmoidV2/all_aclnnHardsigmoidV2.json \
  -p <ops-nn-repo>/experimental/activation/hard_sigmoid_v2/tests/st/aclnnHardsigmoidV2/executor_aclnnHardsigmoidV2.py
```
