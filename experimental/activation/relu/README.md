# Relu

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

- 算子功能：对输入 Tensor 完成 ReLU 运算。

计算公式：

$$
\operatorname{relu}(x) = \max(x, 0)
$$

- 目录 `experimental/activation/relu` 对外导出 `aclnnRelu` 和 `aclnnInplaceRelu` 两段式 ACLNN 接口。
- `op_host/op_api/aclnn_relu.cpp` 是对外 ACLNN 接口入口。
- `op_host/op_api/relu.h` 和 `op_host/op_api/relu.cpp` 提供内部 `l0op::Relu` 封装，当前由 ACLNN 接口直接调用。

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |
| ACLNN 原地调用 | 是 |

## ACLNN 接口

### 函数原型

当前 experimental Relu 提供两段式 ACLNN 接口：

```cpp
aclnnStatus aclnnReluGetWorkspaceSize(
    const aclTensor *self,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnRelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

原地版本接口如下：

```cpp
aclnnStatus aclnnInplaceReluGetWorkspaceSize(
    aclTensor *selfRef,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnInplaceRelu(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数说明见 [docs/aclnnRelu&aclnnInplaceRelu.md](docs/aclnnRelu&aclnnInplaceRelu.md)。

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
    <td>待进行 ReLU 计算的入参。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>out</td>
    <td>输出</td>
    <td>非原地接口的计算出参。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>selfRef</td>
    <td>输入/输出</td>
    <td>原地接口的输入输出张量。</td>
    <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT32、INT64</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- 输入仅支持 `FLOAT`、`FLOAT16`、`BFLOAT16`、`INT8`、`INT32`、`INT64`。
- `BFLOAT16` 仅在 `Ascend910B` 及后续同代 SoC 上支持。
- 输入和输出的 dtype 必须一致。
- 输入和输出的 shape 必须完全一致。
- 支持 0 到 8 维 Tensor。
- 支持空 Tensor。
- 支持非连续 Tensor，接口内部会在需要时做 `Contiguous` 和 `ViewCopy`。
- `FLOAT`、`FLOAT16`、`INT32` 路径直接使用 AscendC `Relu` 计算。
- `BFLOAT16` 路径在 kernel 中升精度到 `float32` 计算后回写。
- `INT8` 路径在 kernel 中升精度到 `float16` 计算后回写。
- `INT64` 路径按已验证基线走逐元素标量语义。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_relu.cpp](examples/test_aclnn_relu.cpp) | `aclnnRelu` 两段式调用示例。 |
| [examples/test_aclnn_inplace_relu.cpp](examples/test_aclnn_inplace_relu.cpp) | `aclnnInplaceRelu` 两段式调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnRelu&aclnnInplaceRelu.md](docs/aclnnRelu&aclnnInplaceRelu.md) | `aclnnRelu` / `aclnnInplaceRelu` 接口文档。 |
| [tests/ut/op_api/test_aclnn_relu.cpp](tests/ut/op_api/test_aclnn_relu.cpp) | `op_api` 单元测试。 |
| [tests/st/aclnnRelu/all_aclnnRelu.json](tests/st/aclnnRelu/all_aclnnRelu.json) | 适用于 ATK 的小规模标准化测试集。 |
| [tests/st/aclnnRelu/executor_aclnnRelu.py](tests/st/aclnnRelu/executor_aclnnRelu.py) | ATK CPU benchmark 执行器。 |

## Example 运行

先确保 custom run 包已经安装，并加载 CANN 环境：

```bash
source /usr/local/Ascend/cann-8.5.0-beta.1/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann-8.5.0-beta.1/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd /root/src/cann/ops-nn/experimental/activation/relu/examples
bash run.sh
```

示例会编译并运行 `test_aclnn_relu.cpp` 与 `test_aclnn_inplace_relu.cpp`。

## Tests 运行

### 1. op_api 单元测试

```bash
source /usr/local/Ascend/cann/set_env.sh
cd /root/src/cann/ops-nn
bash build.sh --experimental --ops=relu -u --opapi -j8 -O2
```

### 2. ATK 小规模标准化测试

```bash
export ATK_BIND_CPU_TYPE=2
source /usr/local/Ascend/cann/set_env.sh
source /root/src/kernel/ascend-kernel/.venv/bin/activate
cd /root/src/testcase
atk node --backend npu --devices 2 \
  node --backend cpu task --task accuracy \
  -c /root/src/cann/ops-nn/experimental/activation/relu/tests/st/aclnnRelu/all_aclnnRelu.json \
  -p /root/src/cann/ops-nn/experimental/activation/relu/tests/st/aclnnRelu/executor_aclnnRelu.py
```
