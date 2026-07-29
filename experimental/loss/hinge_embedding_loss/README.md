# HingeEmbeddingLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

HingeEmbeddingLoss 根据 `target` 的类别标记计算输入距离的损失。设输入元素为 $x_i$，标签为
$y_i\in\{-1,1\}$，则：

$$
l_i =
\begin{cases}
x_i, & y_i = 1 \\
\max(0, margin-x_i), & y_i = -1
\end{cases}
$$

`reduction` 为 `none` 时输出 $l_i$；为 `sum` 时输出元素和；为 `mean` 时输出元素均值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| input | 输入 | 输入距离或相似性张量。 | FLOAT、FLOAT16、BFLOAT16 | ND | 必须与 `target` 完全一致。 |
| target | 输入 | 标签张量，元素应为 `1` 或 `-1`。 | FLOAT、FLOAT16、BFLOAT16 | ND | 必须与 `input` 完全一致。 |
| margin | 可选属性 | 负标签分支的间隔，默认 `1.0`。 | FLOAT | - | 任意有限浮点数。 |
| reduction | 可选属性 | 规约方式，默认 `mean`。 | STRING | - | 支持 `none`、`sum`、`mean`。 |
| loss | 输出 | Hinge embedding loss。 | 与 `input` 一致 | ND | `none` 时与输入一致；其他模式含一个元素。 |

## 约束说明

- `input`、`target` 和 `loss` 的数据类型必须一致。
- `input` 与 `target` 的 shape 必须一致，不支持广播。
- `target` 元素应为 `1` 或 `-1`；Host 不读取 Device 数据，因此该值域由调用者保证。
- FLOAT16、BFLOAT16 在 Kernel 内转换为 FLOAT 计算。
- `none`、`sum`、`mean` 分别使用独立 tiling key；Kernel 在编译期选择直写、求和或均值路径。
- `sum`、`mean` 多核执行需要第一段 ACLNN 接口返回的 workspace。
- 支持动态 rank 和动态 shape；运行时 tiling 使用已确定的 storage shape。
- 仅支持 Atlas A2 训练系列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| ACLNN 调用 | [aclnnHingeEmbeddingLoss 接口文档](./docs/aclnnHingeEmbeddingLoss.md)、[test_aclnn_hinge_embedding_loss.cpp](./examples/test_aclnn_hinge_embedding_loss.cpp) | 两段式接口。 |
| GE | - | 暂不提供。 |
| PyTorch | - | 暂不提供。 |

## 本地编译运行 UT

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash build.sh -u --ophost --ops=hinge_embedding_loss --soc=ascend910b --experimental
bash build.sh -u --opkernel --ops=hinge_embedding_loss --soc=ascend910b --experimental
```

## 本地自测 UT 覆盖率

- Host InferShape：覆盖 `none/sum/mean`、动态 shape、非法 shape 和非法 reduction。
- Host Tiling：覆盖三种 dtype、空张量、非均匀多核、多 tile、workspace、非法 dtype/shape/reduction/UB。
- Kernel：覆盖正负标签分支、`margin-input` 的正值/零值/负值边界、三种 dtype、三种 reduction 和非均匀多核多 tile。
- Golden 脚本位于 [hinge_embedding_loss_data](./tests/ut/op_kernel/hinge_embedding_loss_data)。

## eager 调用前置条件

先 source 匹配的 CANN 环境，构建并安装包含本算子的 custom package；运行时使用 `cust`，且
`vendor_name=custom` 必须与安装包匹配：

```bash
bash build.sh --run_example hinge_embedding_loss eager cust \
  --example_name=hinge_embedding_loss --vendor_name=custom --experimental
```

## 参考资源

- [PyTorch HingeEmbeddingLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html)
