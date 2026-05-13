# Swish

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

- 算子功能：对输入 Tensor 完成 Swish 激活计算。

计算公式：

$$
y = \frac{x}{1 + e^{-\text{scale} \times x}}
$$

- 目录 `experimental/activation/swish_v2` 对外导出 `aclnnSwish` 两段式 ACLNN 接口。
- 当前仅导出普通接口，未导出任何 inplace 接口。
- `op_host/op_api/aclnn_swish.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/swish.h` 和 `op_host/op_api/swish.cpp` 提供内部 `l0op::SwishV2` 封装，当前由 ACLNN 接口直接调用。
- `betaOptional` 为空时按 `scale = 1.0f` 执行。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental Swish 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnSwishGetWorkspaceSize(
    const aclTensor *self,
    const aclScalar *betaOptional,
    aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSwish(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnSwish.md](docs/aclnnSwish.md)。

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
    <td>待进行 Swish 计算的输入张量，公式中的 `x`。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>betaOptional</td>
    <td>输入</td>
    <td>可选缩放系数，公式中的 `scale`；为空时默认值为 `1.0`。</td>
    <td>可转换到 FLOAT 的标量类型</td>
    <td>Scalar</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>Swish 计算结果张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`。
- `BFLOAT16` 仅在 `Ascend910B` 及后续同代 SoC 上支持。
- 输入和输出的 dtype 必须一致。
- 输入和输出的 shape 必须完全一致。
- `betaOptional` 允许为空；非空时需要能转换为 `FLOAT`。
- 支持 0 到 8 维 Tensor。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。
- `FLOAT16` 和 `BFLOAT16` 路径在 kernel 中采用升精度计算，再回写到原 dtype。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_swish.cpp](examples/test_aclnn_swish.cpp) | `aclnnSwish` 两段式调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnSwish.md](docs/aclnnSwish.md) | `aclnnSwish` 接口文档。 |
| [tests/ut/op_api/test_aclnn_swish.cpp](tests/ut/op_api/test_aclnn_swish.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnSwish/all_aclnnSwish.json](tests/st/aclnnSwish/all_aclnnSwish.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnSwish/executor_aclnnSwish.py](tests/st/aclnnSwish/executor_aclnnSwish.py) | ATK CPU benchmark 执行器。 |

## Example 运行

先确保 custom run 包已经安装，并加载 CANN 环境：

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd <ops-nn-repo>/experimental/activation/swish_v2/examples
bash run.sh
```

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=swish_v2 -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd /root/src/testcase
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c <ops-nn-repo>/experimental/activation/swish_v2/tests/st/aclnnSwish/all_aclnnSwish.json \
  -p <ops-nn-repo>/experimental/activation/swish_v2/tests/st/aclnnSwish/executor_aclnnSwish.py
```
