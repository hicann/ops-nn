# aclnnSoftshrinkBackward

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Atlas A3系列产品</term>| √ |
| <term>Atlas A2系列产品</term> | √ |

## 功能说明

- 接口功能：计算 Softshrink 激活函数的反向传播梯度。

- 计算公式：

  $$
  gradInput_i =
  \begin{cases}
  gradOutput_i, & self_i > lambd \\
  gradOutput_i, & self_i < -lambd \\
  0, & -lambd \le self_i \le lambd
  \end{cases}
  $$

  `gradOutput` 与 `self` 支持广播。当 `self` 等于 `lambd` 或 `-lambd` 时，输出为 0。

## 函数原型

每个算子分为两段式接口，必须先调用
`aclnnSoftshrinkBackwardGetWorkspaceSize` 接口获取计算所需的 workspace 大小和执行器，再调用
`aclnnSoftshrinkBackward` 接口执行计算。

```Cpp
aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(
  const aclTensor* gradOutput,
  const aclTensor* self,
  const aclScalar* lambd,
  aclTensor*       gradInput,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnSoftshrinkBackward(
  void*             workspace,
  uint64_t          workspaceSize,
  aclOpExecutor*    executor,
  const aclrtStream stream)
```

## aclnnSoftshrinkBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 180px">
  <col style="width: 115px">
  <col style="width: 220px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度（shape）</th>
      <th>非连续 Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>gradOutput（aclTensor*）</td>
      <td>输入</td>
      <td>反向传播的上游梯度，公式中的 gradOutput。</td>
      <td><ul><li>支持空 Tensor。</li><li>与 self 的 shape 满足 broadcast 关系。</li><li>数据类型可以与 self 不同；不同时内部转换为 FLOAT 计算。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self（aclTensor*）</td>
      <td>输入</td>
      <td>Softshrink 正向计算的输入，公式中的 self。</td>
      <td><ul><li>支持空 Tensor。</li><li>与 gradOutput 的 shape 满足 broadcast 关系。</li><li>数据类型可以与 gradOutput 不同；不同时内部转换为 FLOAT 计算。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>lambd（aclScalar*）</td>
      <td>输入</td>
      <td>Softshrink 计算的阈值，公式中的 lambd。</td>
      <td>取值必须大于或等于 0。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput（aclTensor*）</td>
      <td>输出</td>
      <td>Softshrink 反向传播的输出梯度，公式中的 gradInput。</td>
      <td><ul><li>支持空 Tensor。</li><li>shape 必须与 gradOutput 和 self 广播后的 shape 一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在 Device 侧申请的 workspace 大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回算子执行器，包含算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的 gradOutput、self、lambd 或 gradInput 是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>gradOutput、self 或 gradInput 的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>当前硬件平台不是 Atlas A2 或 Atlas A3。</td>
    </tr>
    <tr>
      <td>gradOutput 或 self 的维度大于 8。</td>
    </tr>
    <tr>
      <td>gradOutput 与 self 的 shape 不满足 broadcast 条件，或 gradInput 的 shape 与 broadcast 结果不一致。</td>
    </tr>
    <tr>
      <td>lambd 小于 0。</td>
    </tr>
  </tbody>
  </table>

## aclnnSoftshrinkBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnSoftshrinkBackwardGetWorkspaceSize 获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>算子执行器，包含算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的 Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见 aclnn 返回码。

## 约束说明

- `gradOutput`、`self` 和 `gradInput` 的维度范围为 0-8。
- `gradOutput` 与 `self` 必须满足 broadcast 关系，`gradInput` 的 shape 必须等于 broadcast 结果。
- `lambd` 必须大于或等于 0。
- 支持空 Tensor 和非连续 Tensor。
- 确定性计算：`aclnnSoftshrinkBackward` 默认确定性实现。

## 调用示例

完整示例代码参见
[test_aclnn_softshrink_grad.cpp](../examples/test_aclnn_softshrink_grad.cpp)。在仓库根目录执行以下命令编译并运行：

```bash
bash build.sh --run_example softshrink_grad eager cust --vendor_name=custom --experimental
```

两段式接口的核心调用方式如下：

```Cpp
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnStatus ret = aclnnSoftshrinkBackwardGetWorkspaceSize(
    gradOutput, self, lambd, gradInput, &workspaceSize, &executor);

void* workspace = nullptr;
if (ret == ACLNN_SUCCESS && workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}
if (ret == ACLNN_SUCCESS) {
    ret = aclnnSoftshrinkBackward(workspace, workspaceSize, executor, stream);
}
if (ret == ACLNN_SUCCESS) {
    ret = aclrtSynchronizeStream(stream);
}
if (workspace != nullptr) {
    aclrtFree(workspace);
}
```
