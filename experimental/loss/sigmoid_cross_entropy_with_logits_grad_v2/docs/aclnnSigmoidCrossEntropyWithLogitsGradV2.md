# aclnnSigmoidCrossEntropyWithLogitsGradV2

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/loss/sigmoid_cross_entropy_with_logits_grad_v2)

## 产品支持情况

|产品|是否支持|
|:---|:---:|
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|√|

## 功能说明

- 接口功能：计算 SigmoidCrossEntropyWithLogitsGradV2 的梯度输出。
- 算子对应的实际 ACLNN 两段式接口为：
  - `aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize`
  - `aclnnBinaryCrossEntropyWithLogitsBackward`

参数映射关系：
- `self` 对应算子输入 `predict`（logits）
- `gradOutput` 对应算子输入 `dout`
- `out` 对应算子输出 `gradient`

设输入 `self`（即 logits）、`target`、`gradOutput`（即上游梯度），定义：

$$
p = sigmoid(self) = \frac{1}{1 + e^{-self}}
$$

当 `posWeightOptional` 不存在时：

$$
grad\_base = p - target
$$

当 `posWeightOptional` 存在时：

$$
log\_weight = posWeightOptional \cdot target
$$

$$
grad\_base = (log\_weight + 1 - target) \cdot p - log\_weight
$$

输出计算：

$$
out = grad\_base \cdot gradOutput
$$

若 `weightOptional` 非空：

$$
out = out \cdot weightOptional
$$

`reduction` 语义：0 表示 none，1 表示 mean，2 表示 sum。

当 `reduction = mean` 时，输出按元素总数做均值缩放。

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize”接口获取计算所需 workspace 大小以及执行器，再调用“aclnnBinaryCrossEntropyWithLogitsBackward”接口执行计算。

```Cpp
aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* self,
    const aclTensor* target,
    const aclTensor* weightOptional,
    const aclTensor* posWeightOptional,
    int64_t reduction,
    aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor)

aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    const aclrtStream stream)
```

## aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize

- 参数说明：

<table style="undefined;table-layout: fixed; width: 1475px"><colgroup>
<col style="width: 210px">
<col style="width: 120px">
<col style="width: 320px">
<col style="width: 320px">
<col style="width: 150px">
<col style="width: 110px">
<col style="width: 120px">
<col style="width: 130px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出</th>
<th>描述</th>
<th>使用说明</th>
<th>数据类型</th>
<th>数据格式</th>
<th>维度(shape)</th>
<th>非连续Tensor</th>
</tr></thead>
<tbody>
<tr>
<td>gradOutput(aclTensor*)</td>
<td>输入</td>
<td>上游梯度输入。</td>
<td>shape 需与 self 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>self(aclTensor*)</td>
<td>输入</td>
<td>输入 logits。</td>
<td>shape 需与 target 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>target(aclTensor*)</td>
<td>输入</td>
<td>标签输入。</td>
<td>shape 需与 self 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>weightOptional(aclTensor*)</td>
<td>可选输入</td>
<td>样本权重输入。</td>
<td>若存在需与 self shape 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>posWeightOptional(aclTensor*)</td>
<td>可选输入</td>
<td>正样本权重输入。</td>
<td>若存在需与 self shape 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>reduction(int64_t)</td>
<td>输入</td>
<td>输出缩减方式。</td>
<td>支持 0/1/2：0-none，1-mean，2-sum。</td>
<td>INT64</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>out(aclTensor*)</td>
<td>输出</td>
<td>梯度输出。</td>
<td>shape 与 self 一致。</td>
<td>FLOAT、FLOAT16、BFLOAT16</td>
<td>ND</td>
<td>1-8</td>
<td>√</td>
</tr>
<tr>
<td>workspaceSize(uint64_t*)</td>
<td>输出</td>
<td>返回需要在 Device 侧申请的 workspace 大小。</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>executor(aclOpExecutor**)</td>
<td>输出</td>
<td>返回 op 执行器，包含了算子计算流程。</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
</tbody></table>

- 返回值：

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

第一段接口完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
<col style="width: 269px">
<col style="width: 120px">
<col style="width: 761px">
</colgroup>
<thead>
<tr>
<th>返回值</th>
<th>错误码</th>
<th>描述</th>
</tr></thead>
<tbody>
<tr>
<td>ACLNN_ERR_PARAM_NULLPTR</td>
<td>161001</td>
<td>`gradOutput`、`self`、`target` 或 `out` 是空指针。</td>
</tr>
<tr>
<td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
<td rowspan="7">161002</td>
<td>`gradOutput`、`self`、`target` 或 `out` 的数据类型不在支持范围内。</td>
</tr>
<tr>
<td>`out` 的数据类型与 `self` 不一致。</td>
</tr>
<tr>
<td>`gradOutput`、`self`、`target`、`out` 的 shape 不一致。</td>
</tr>
<tr>
<td>`weightOptional` 或 `posWeightOptional`（非空时）shape 与 `self` 不一致。</td>
</tr>
<tr>
<td>`reduction` 不在 0/1/2 取值范围内。</td>
</tr>
<tr>
<td>输入维度超过 8 维。</td>
</tr>
<tr>
<td>输入/输出 format 不在支持范围内。</td>
</tr>
</tbody>
</table>

## aclnnBinaryCrossEntropyWithLogitsBackward

- 参数说明：

<table style="undefined;table-layout: fixed; width: 1151px"><colgroup>
<col style="width: 184px">
<col style="width: 134px">
<col style="width: 833px">
</colgroup>
<thead>
<tr>
<th>参数名</th>
<th>输入/输出</th>
<th>描述</th>
</tr></thead>
<tbody>
<tr>
<td>workspace</td>
<td>输入</td>
<td>在 Device 侧申请的 workspace 内存地址。</td>
</tr>
<tr>
<td>workspaceSize</td>
<td>输入</td>
<td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize 获取。</td>
</tr>
<tr>
<td>executor</td>
<td>输入</td>
<td>op 执行器，包含了算子计算流程。</td>
</tr>
<tr>
<td>stream</td>
<td>输入</td>
<td>指定执行任务的 Stream。</td>
</tr>
</tbody>
</table>

- 返回值：

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - `aclnnBinaryCrossEntropyWithLogitsBackward` 默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "aclnn_binary_cross_entropy_with_logits_backward.h"

// 省略 aclInit / stream 初始化与 aclTensor 构造

uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
int64_t reduction = 1; // 0:none, 1:mean, 2:sum

aclnnStatus ret = aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(
    doutTensor,
    predictTensor,
    targetTensor,
    weightTensor,
    posWeightTensor,
    reduction,
    outputTensor,
    &workspaceSize,
    &executor);

void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
    aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
}

ret = aclnnBinaryCrossEntropyWithLogitsBackward(workspaceAddr, workspaceSize, executor, stream);
aclrtSynchronizeStream(stream);

// 省略输出回拷与资源释放
```

完整示例代码请参考：
- [test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2](../examples/test_aclnn_sigmoid_cross_entropy_with_logits_grad_v2.cpp)