# aclnnGaussianNllLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

计算连续高斯分布的负对数似然损失。令 $d_i=input_i-target_i$、
$v_i=\max(var_i,eps)$，逐元素结果为：

$$
l_i=\frac{1}{2}\left(\log(v_i)+\frac{d_i^2}{v_i}\right)
+\begin{cases}
\frac{1}{2}\log(2\pi), & full=true \\
0, & full=false
\end{cases}
$$

支持 `none`、`sum`、`mean` 三种规约。FLOAT16 和 BFLOAT16 在 Kernel 内转换为 FLOAT 计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用
“aclnnGaussianNllLossGetWorkspaceSize”获取 workspace 大小和执行器，再调用
“aclnnGaussianNllLoss”执行计算。

```Cpp
aclnnStatus aclnnGaussianNllLossGetWorkspaceSize(
    const aclTensor* input,
    const aclTensor* target,
    const aclTensor* var,
    bool             full,
    double           eps,
    char*            reductionOptional,
    const aclTensor* out,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnGaussianNllLoss(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnGaussianNllLossGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 220px"><col style="width: 110px"><col style="width: 250px">
  <col style="width: 410px"><col style="width: 180px"><col style="width: 100px">
  <col style="width: 180px"></colgroup><thead><tr><th>参数名</th><th>输入/输出</th>
  <th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th>
  </tr></thead><tbody>
  <tr><td>input（const aclTensor*）</td><td>输入</td><td>预测均值。</td>
  <td>确定逐元素输出的逻辑 shape。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>任意维</td></tr>
  <tr><td>target（const aclTensor*）</td><td>输入</td><td>目标值。</td>
  <td>dtype 与 input 一致；支持相同 shape 或同 rank 单轴广播。</td>
  <td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>与 input 相同，或恰有一个维度为 1</td></tr>
  <tr><td>var（const aclTensor*）</td><td>输入</td><td>预测方差。</td>
  <td>dtype 与 input 一致；元素应非负。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td>
  <td>与 input 相同、最后一维为 1、少最后一维或标量</td></tr>
  <tr><td>full（bool）</td><td>输入</td><td>是否加入常数项。</td>
  <td>默认 false；ACLNN 调用时显式传入。</td><td>BOOL</td><td>-</td><td>-</td></tr>
  <tr><td>eps（double）</td><td>输入</td><td>方差下限。</td>
  <td>默认 1e-6，必须为有限正数；ACLNN 调用时显式传入。</td><td>FLOAT</td><td>-</td><td>-</td></tr>
  <tr><td>reductionOptional（char*）</td><td>输入</td><td>规约方式。</td>
  <td>支持 none、sum、mean，默认 mean；ACLNN 调用时显式传入。</td><td>STRING</td><td>-</td><td>-</td></tr>
  <tr><td>out（const aclTensor*）</td><td>输出</td><td>损失结果。</td>
  <td>none 时 shape 与 input 一致；sum/mean 时含一个元素。</td>
  <td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>任意维或单元素</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回 Device workspace 大小。</td>
  <td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回 op 执行器。</td>
  <td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

  常见失败场景：

  - 输入或输出为空指针。
  - 输入输出 dtype 不受支持或不一致。
  - `target` 或 `var` shape 不属于已支持的广播形式。
  - `eps` 不是有限正数。
  - `loss` shape 与 reduction 不匹配。
  - reduction 不属于 `none`、`sum`、`mean`。

## aclnnGaussianNllLoss

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 200px"><col style="width: 150px"><col style="width: 800px"></colgroup>
  <thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>Device workspace 地址，以第一段接口返回值为准。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>第一段接口返回的 workspace 大小。</td></tr>
  <tr><td>executor</td><td>输入</td><td>第一段接口返回的执行器。</td></tr>
  <tr><td>stream</td><td>输入</td><td>执行任务的 Stream。</td></tr>
  </tbody></table>

- **返回值**：

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `input`、`target`、`var`、`loss` 的 dtype 必须一致，仅支持 FLOAT、FLOAT16、BFLOAT16 和 ND。
- `target` 与 `input` shape 相同，或同 rank 且恰有一个不同维度为 1。
- `var` 与 `input` shape 相同、最后一维为 1、比 `input` 少最后一维，或为标量。
- `var` 元素应非负，该 Device 数据值域由调用者保证；计算使用 `max(var, eps)`。
- `eps` 必须为有限正数；reduction 仅支持 `none`、`sum`、`mean`。
- `sum`、`mean` 多核执行需要第一段接口返回的 workspace。
- 支持动态 rank 和动态 shape。
- 仅支持 Atlas A2 训练系列产品。

## 调用示例

具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)，完整样例见
[test_aclnn_gaussian_nll_loss.cpp](../examples/test_aclnn_gaussian_nll_loss.cpp)。

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
char reduction[] = "mean";
auto ret = aclnnGaussianNllLossGetWorkspaceSize(
    input, target, var, true, 1e-6, reduction, loss, &workspaceSize, &executor);
if (ret == ACL_SUCCESS) {
    ret = aclnnGaussianNllLoss(workspace, workspaceSize, executor, stream);
}
```
