# MishGradV2

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

- 算子功能：计算 Mish 的反向梯度。
- 目录 `experimental/activation/mish_grad_v2` 复用 `ascend-kernel/csrc/ops/mish_grad` 中已经验证的 AscendC 实现，目录名保留 `v2` 仅用于区分新实现，对外导出的 ACLNN 接口仍为 `aclnnMishBackward`。
- `op_host/op_api/aclnn_mish_backward_v2.cpp` 是对外 ACLNN 两段式接口入口。
- `op_host/op_api/mish_grad_v2.h` 和 `op_host/op_api/mish_grad_v2.cpp` 提供内部 `l0op::MishGradV2` 封装。

计算公式：

$$
xGrad = grad * \left(tanhx + x * \frac{1 - tanhx^2}{1 + e^{-x}}\right)
$$

其中：

$$
tanhx = tanh(softplus(x)),\ softplus(x) = relu(x) + log(1 + e^{-|x|})
$$

## 调用方式

| 调用方式 | 是否支持 |
| :------- | :------: |
| ACLNN 调用 | 是 |

## ACLNN 接口

### 函数原型

```cpp
aclnnStatus aclnnMishBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    aclTensor *gradInput,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnMishBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    const aclrtStream stream);
```

详细参数和返回值说明见 [docs/aclnnMishBackward.md](docs/aclnnMishBackward.md)。

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
    <td>上游反向传播输入梯度。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>self</td>
    <td>输入</td>
    <td>Mish 正向输入。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>gradInput</td>
    <td>输出</td>
    <td>Mish 反向梯度计算结果。</td>
    <td>FLOAT、FLOAT16、BFLOAT16（仅 Ascend910B 及后续同代 SoC 支持）</td>
    <td>ND</td>
  </tr>
</tbody>
</table>

## 约束说明

- `gradOutput`、`self` 和 `gradInput` 的 shape 必须完全一致，不支持 broadcast。
- `gradOutput`、`self` 和 `gradInput` 的数据类型必须完全一致，不支持隐式类型提升。
- 支持非连续 Tensor，内部会执行连续化和 `ViewCopy`。
- 支持空 Tensor。

## 目录说明

| 路径 | 说明 |
| :--- | :--- |
| [examples/test_aclnn_mish_grad_v2.cpp](examples/test_aclnn_mish_grad_v2.cpp) | `aclnnMishBackward` 两段式调用示例。 |
| [examples/run.sh](examples/run.sh) | 编译并运行 example 的脚本。 |
| [docs/aclnnMishBackward.md](docs/aclnnMishBackward.md) | `aclnnMishBackward` 接口文档。 |
| [tests/ut/op_host/test_aclnn_mish_backward.cpp](tests/ut/op_host/test_aclnn_mish_backward.cpp) | `op_api` 单元测试。 |

## Example 运行

```bash
source /usr/local/Ascend/cann/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/cann/opp/vendors/customize_nn/op_api/lib:${LD_LIBRARY_PATH}
cd <ops-nn-repo>/experimental/activation/mish_grad_v2/examples
bash run.sh
```

## Tests 运行

```bash
source /usr/local/Ascend/cann/set_env.sh
cd <ops-nn-repo>
bash build.sh --experimental --ops=mish_grad_v2 -u --opapi -j8 -O2
```
