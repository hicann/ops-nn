# aclnnMaxPool2dWithMaskBackward

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :------: |
| Atlas 800T A2 | √ |
| Atlas 300V Pro | √ |

## 功能说明

计算 `aclnnMaxPool2dWithMask` 的反向梯度。算子使用正向输出 `indices` 中记录的
最大值位置，将 `gradOutput` 按重叠窗口的累加语义写回 `gradInput`；未被任何
最大值位置选中的输入元素，其梯度保持为 0。

对于输出位置 $(n,c,oh,ow)$，设 `indices` 解码得到的最大值位置为
$(n,c,ih,iw)$，则：

$$
\operatorname{gradInput}(n,c,ih,iw)
\mathrel{+}= \operatorname{gradOutput}(n,c,oh,ow)
$$

当窗口发生重叠且同一输入位置被多个输出位置选中时，梯度会累加。

## 函数原型

本算子分为两段式接口。先调用
`aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize` 获取 workspace 大小和执行器，
再调用 `aclnnMaxPool2dWithMaskBackward` 执行计算。

```cpp
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(
    const aclTensor* gradOutput,
    const aclTensor* self,
    const aclTensor* indices,
    const aclIntArray* kernelSize,
    const aclIntArray* stride,
    const aclIntArray* padding,
    const aclIntArray* dilation,
    bool ceilMode,
    aclTensor* gradInput,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);
```

```cpp
aclnnStatus aclnnMaxPool2dWithMaskBackward(
    void* workspace,
    uint64_t workspaceSize,
    aclOpExecutor* executor,
    aclrtStream stream);
```

## aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize

| 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度 | 非连续 Tensor |
| ------ | :-------: | ---- | -------- | -------- | :------: | :--: | :-----------: |
| `gradOutput` | 输入 | 正向输出对应的上游梯度。 | shape 应与正向输出 `[N, C, H_out, W_out]` 一致，且 dtype 与 `self` 相同。 | FLOAT32、FLOAT16；910B 额外支持 BFLOAT16 | NCHW | 4 | √ |
| `self` | 输入 | 正向输入张量。 | shape 应与 `gradInput` 一致，dtype 应与 `gradOutput` 相同。 | FLOAT32、FLOAT16；910B 额外支持 BFLOAT16 | NCHW | 4 | √ |
| `indices` | 输入 | 正向 `aclnnMaxPool2dWithMask` 输出的 INT8 mask 容器。 | 必须直接使用对应正向输出，不应在 Host 侧改写；其 shape 为 `[N, C, kH*kW, (ceil(H_out*W_out/16)+1)*2*16]`。 | INT8 | NCHW | 4 | √ |
| `kernelSize` | 输入 | 池化窗口大小。 | 长度为 1 或 2，所有值必须大于 0。 | INT64 | - | - | - |
| `stride` | 输入 | 池化窗口移动步长。 | 长度为 0、1 或 2；长度为 0 时取 `kernelSize`，非空时所有值必须大于 0。 | INT64 | - | - | - |
| `padding` | 输入 | 输入的高度和宽度方向填充量。 | 长度为 1 或 2；值必须非负且不大于对应方向 `kernelSize / 2`。 | INT64 | - | - | - |
| `dilation` | 输入 | 窗口内元素间隔。 | 长度为 1 或 2，当前所有值仅支持 1。 | INT64 | - | - | - |
| `ceilMode` | 输入 | 是否使用向上取整规则计算正向输出 shape。 | `true` 表示 ceil 模式，`false` 表示 floor 模式。 | BOOL | - | - | - |
| `gradInput` | 输出 | 反向传播得到的输入梯度。 | shape 和 format 应与 `self` 一致，dtype 应与 `gradOutput` 相同。 | FLOAT32、FLOAT16；910B 额外支持 BFLOAT16 | NCHW | 4 | √ |
| `workspaceSize` | 输出 | 返回 Device 侧需要申请的 workspace 大小。 | - | `uint64_t*` | - | - | - |
| `executor` | 输出 | 返回包含算子计算流程的执行器。 | - | `aclOpExecutor**` | - | - | - |

返回值 `aclnnStatus` 为状态码。第一段接口完成空指针、dtype、format、shape
和属性校验；具体返回值可参考 ACLNN 返回码定义。常见错误如下：

| 返回码 | 错误码 | 触发场景 |
| ------ | :----: | -------- |
| `ACLNN_ERR_PARAM_NULLPTR` | 161001 | `gradOutput`、`self`、`indices`、`kernelSize`、`stride`、`padding`、`dilation` 或 `gradInput` 为空指针。 |
| `ACLNN_ERR_PARAM_INVALID` | 161002 | dtype、format 或 shape 不支持；`gradInput` 与 `self` shape 不一致；`indices` 与正向输出 shape 不匹配；属性长度或取值非法；当前硬件不支持对应计算分支。 |

## aclnnMaxPool2dWithMaskBackward

| 参数名 | 输入/输出 | 描述 |
| ------ | :-------: | ---- |
| `workspace` | 输入 | Device 侧申请的 workspace 地址。`workspaceSize` 为 0 时可以为空指针。 |
| `workspaceSize` | 输入 | 第一段接口返回的 workspace 大小。 |
| `executor` | 输入 | 第一段接口返回的执行器。 |
| `stream` | 输入 | 执行任务的 ACL Stream。 |

返回值 `aclnnStatus` 为状态码。调用后应检查返回值，并在读取结果前同步 Stream。

## indices 编码契约（indices encoding contract）

`indices` 的公开 dtype 为 INT8，shape 为 `[N, C, K, L]`，其中
`K = kH * kW`，`L = (ceil(H_out * W_out / 16) + 1) * 32`。
该 shape 描述存储容器，不能单独用于判断内部编码是否兼容。

- **K >= 2**：从容器存储起始地址开始，紧凑存放
  `N * C * H_out * W_out` 个 INT32 索引，按 `[N, C, H_out, W_out]`
  行优先顺序排列；不是每个 `[K, L]` 切片分别存放一个前缀。
  索引为同一输入通道平面中的 `h * W_in + w`，范围为
  `[0, H_in * W_in)`。该前缀占用的字节数必须不超过容器容量；
  其余容量不参与反向计算。ACLNN 内部只创建 INT32 视图，不将 bitmap 转换成索引。
- **K == 1 且为恒等池化**（stride 为 1、padding 为 0）：在参数校验后复制梯度，
  不读取索引内容。
- **K == 1 且不是恒等池化**：DAV_2201 使用系统 V1 bitmap 路径，
  INT8 容器承载配套 V1 前向产生的位图，ACLNN 内部将其作为
  UINT16 / NC1HWC0 mask 视图使用，不将其解释为 INT32 索引。
  DAV_2002（310P）的现有路径仍使用上述 INT32 前缀，调用者必须保证容器容量足够；
  不能将 V1 bitmap 直接交给该路径。

实际业务必须使用满足对应编码契约的配套前向输出，并保持其内容不变。
仅有相同 dtype 和 shape 的其他前向 bitmap 与 INT32 前缀并不互通。
调用示例中的固定索引用于说明和验证该契约，不要求业务侧自行重写前向输出。

## 工作区说明

Host 侧为 Kernel 申请的总工作区为用户临时缓冲区与系统工作区之和。
用户临时缓冲区大小为 0，不表示系统工作区也为 0；需要输出清零及核间同步的路径
仍保留系统工作区。普通 ACLNN 调用应严格按第一段返回的 `workspaceSize` 分配内存。
Kernel 直调时，只有无重叠、完整覆盖输出且不需要清零/同步的 Normal 路径允许空工作区；
不能把 ACLNN 的零大小规则直接套用于所有 Kernel 分支。

## 约束说明

- 当前公开契约支持 NCHW 4D 输入输出；`gradInput` 与 `self` 的 shape 和 format 必须一致。
- `gradOutput` 的 H/W 必须与 `self` 和池化属性推导出的正向输出 shape 一致。
- `indices` 必须使用匹配的正向算子输出，其 shape 和内容都不能由业务侧重新定义。
- `gradOutput`、`self` 和 `gradInput` 的 dtype 必须一致。Atlas 300V Pro 支持
  FLOAT32/FLOAT16；Atlas 800T A2 额外支持 BFLOAT16。
- 当前 `dilation` 仅支持 1；`padding` 必须为非负的对称填充。
- 输入数据暂不支持 NaN 和 -Inf。
- 重叠窗口的写回按累加语义处理。默认采用非确定性实现，支持通过
  `aclrtCtxSetSysParamOpt` 开启确定性计算。

## 调用示例

独立的反向接口调用示例见
[test_aclnn_max_pool2d_with_mask_backward.cpp](../examples/test_aclnn_max_pool2d_with_mask_backward.cpp)。
示例使用固定的输入、索引和预期梯度，覆盖非重叠、重叠累加和 padding，
只需构建和安装反向算子包。固定索引仅用于该测试；实际业务应将匹配的前向输出
`indices` 原样传入反向接口。编译和安装步骤见[算子 README](../README.md)。
