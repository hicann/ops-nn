# aclnnHingeLoss

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas A2 训练系列产品/Atlas 800I A2 推理产品 | √ |

## 功能说明

- 接口功能：计算逐元素 Hinge Loss 分类损失，用于 SVM 等间隔分类训练场景，并可与 HingeLossGrad 配套完成反向传播。

- 计算公式：对于每个元素 i，模型预测值为 predict_i、标签为 target_i，输出损失为：

  $$
  loss_i = max(0, 1 - target_i * predict_i)
  $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)。必须先调用 `aclnnHingeLossGetWorkspaceSize` 获取 workspace 大小和包含计算流程的执行器，再调用 `aclnnHingeLoss` 执行计算。

```Cpp
aclnnStatus aclnnHingeLossGetWorkspaceSize(
    const aclTensor*  predict,
    const aclTensor*  target,
    aclTensor*        loss,
    uint64_t*         workspaceSize,
    aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnHingeLoss(
    void*             workspace,
    uint64_t          workspaceSize,
    aclOpExecutor*    executor,
    aclrtStream       stream)
```

## aclnnHingeLossGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1400px"><colgroup>
  <col style="width: 190px"><col style="width: 115px"><col style="width: 260px"><col style="width: 310px">
  <col style="width: 170px"><col style="width: 105px"><col style="width: 140px"><col style="width: 145px">
  </colgroup><thead><tr>
  <th>参数名</th><th>输入/输出</th><th>描述</th><th>使用说明</th><th>数据类型</th><th>数据格式</th><th>维度(shape)</th><th>非连续Tensor</th>
  </tr></thead><tbody>
  <tr><td>predict（aclTensor*）</td><td>输入</td><td>模型预测值，对应公式中的 predict_i。</td><td>shape、dtype 必须与 target 一致。</td><td>FLOAT / FLOAT16 / BF16</td><td>ND</td><td>1-8 维</td><td>√</td></tr>
  <tr><td>target（aclTensor*）</td><td>输入</td><td>分类标签，对应公式中的 target_i，通常取值为 1 或 -1。</td><td>shape、dtype 必须与 predict 一致。</td><td>FLOAT / FLOAT16 / BF16</td><td>ND</td><td>与 predict 一致</td><td>√</td></tr>
  <tr><td>loss（aclTensor*）</td><td>输出</td><td>逐元素 Hinge Loss 结果。</td><td>shape、dtype 必须与 predict 一致。</td><td>FLOAT / FLOAT16 / BF16</td><td>ND</td><td>与 predict 一致</td><td>√</td></tr>
  <tr><td>workspaceSize（uint64_t*）</td><td>输出</td><td>返回需要在 Device 侧申请的 workspace 大小。</td><td>当前实现无需额外 workspace，返回 0。</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>executor（aclOpExecutor**）</td><td>输出</td><td>返回 op 执行器，包含算子计算流程。</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  </tbody></table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。第一段接口负责检查空指针、数据类型、格式以及 predict、target、loss 的 shape 一致性；不满足要求时返回参数错误。

## aclnnHingeLoss

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px"><col style="width: 112px"><col style="width: 668px">
  </colgroup><thead><tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr></thead><tbody>
  <tr><td>workspace</td><td>输入</td><td>Device 侧 workspace 地址。当前实现无需 workspace，可传入 nullptr。</td></tr>
  <tr><td>workspaceSize</td><td>输入</td><td>Device 侧 workspace 大小，由 aclnnHingeLossGetWorkspaceSize 获取。</td></tr>
  <tr><td>executor</td><td>输入</td><td>op 执行器，包含算子计算流程。</td></tr>
  <tr><td>stream</td><td>输入</td><td>指定执行任务的 Stream。</td></tr>
  </tbody></table>

- **返回值：**

  `aclnnStatus`：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- predict、target 和 loss 的 shape 必须完全一致；首版不支持广播。
- target 通常取值为 1 或 -1；接口不额外校验 target 数值。
- 算子输出逐元素 loss，不支持 mean 或 sum reduction。

## 调用示例

示例代码见[测试 aclnnHingeLoss](../examples/test_aclnn_hinge_loss.cpp)。编译和运行方法请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。
