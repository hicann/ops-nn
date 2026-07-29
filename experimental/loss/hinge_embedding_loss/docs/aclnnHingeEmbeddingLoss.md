# aclnnHingeEmbeddingLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

计算 `input` 与二值标签 `target` 的 Hinge embedding loss：

$$
l_i =
\begin{cases}
input_i, & target_i = 1 \\
\max(0, margin-input_i), & target_i = -1
\end{cases}
$$

支持 `none`、`sum`、`mean` 三种规约。FLOAT16 和 BFLOAT16 在 Kernel 内转换为 FLOAT 计算。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用
“aclnnHingeEmbeddingLossGetWorkspaceSize”获取 workspace 大小和执行器，再调用
“aclnnHingeEmbeddingLoss”执行计算。

```Cpp
aclnnStatus aclnnHingeEmbeddingLossGetWorkspaceSize(
    const aclTensor* input,
    const aclTensor* target,
    double           margin,
    char*            reduction,
    const aclTensor* loss,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnHingeEmbeddingLoss(
    void*           workspace,
    uint64_t        workspaceSize,
    aclOpExecutor*  executor,
    aclrtStream     stream)
```

## aclnnHingeEmbeddingLossGetWorkspaceSize

- **参数说明**：

  <table class="tg" style="undefined;table-layout: fixed; width: 1450px"><colgroup>
  <col style="width: 220px"><col style="width: 110px"><col style="width: 250px">
  <col style="width: 410px"><col style="width: 180px"><col style="width: 100px">
  <col style="width: 180px"></colgroup><thead><tr><th>参数名</th><th>输入/输出</th>
  <th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th>
  </tr></thead><tbody>
  <tr><td>input（aclTensor*）</td><td>输入</td><td>输入距离张量。</td>
  <td>shape 和 dtype 必须与 target 一致。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>任意维</td></tr>
  <tr><td>target（aclTensor*）</td><td>输入</td><td>二值标签张量。</td>
  <td>元素应为 1 或 -1。</td><td>FLOAT、FLOAT16、BFLOAT16</td><td>ND</td><td>与 input 一致</td></tr>
  <tr><td>margin（double）</td><td>输入</td><td>负标签分支的间隔。</td>
  <td>默认值为 1.0；ACLNN 调用时显式传入。</td><td>FLOAT</td><td>-</td><td>-</td></tr>
  <tr><td>reduction（char*）</td><td>输入</td><td>规约方式。</td>
  <td>支持 none、sum、mean，默认 mean；ACLNN 调用时显式传入。</td><td>STRING</td><td>-</td><td>-</td></tr>
  <tr><td>loss（aclTensor*）</td><td>输出</td><td>损失结果。</td>
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
  - `input`、`target` shape 不一致。
  - `loss` shape 与 reduction 不匹配。
  - reduction 不属于 `none`、`sum`、`mean`。

## aclnnHingeEmbeddingLoss

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

- `input`、`target`、`loss` 的 dtype 必须一致，仅支持 FLOAT、FLOAT16、BFLOAT16。
- `input` 与 `target` shape 必须一致，不支持广播。
- `target` 元素应为 `1` 或 `-1`，该 Device 数据值域由调用者保证。
- reduction 仅支持 `none`、`sum`、`mean`。
- 支持动态 rank 和动态 shape。
- 仅支持 Atlas A2 训练系列产品。

## 调用示例

具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)，完整样例见
[test_aclnn_hinge_embedding_loss.cpp](../examples/test_aclnn_hinge_embedding_loss.cpp)。

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
void* workspace = nullptr;
char reduction[] = "none";
auto ret = aclnnHingeEmbeddingLossGetWorkspaceSize(
    input, target, 1.0, reduction, loss, &workspaceSize, &executor);
if (ret == ACL_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
if (ret == ACL_SUCCESS) {
    ret = aclnnHingeEmbeddingLoss(workspace, workspaceSize, executor, stream);
}
if (workspace != nullptr) {
    aclrtFree(workspace);
}
```
