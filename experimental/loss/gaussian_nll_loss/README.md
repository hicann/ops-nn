# GaussianNllLoss

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A2 训练系列产品</term>     |     √    |

## 功能说明

GaussianNllLoss 计算连续高斯分布的负对数似然损失。设预测均值为 $input_i$、目标值为
$target_i$、预测方差为 $var_i$，并令 $v_i=\max(var_i, eps)$，则逐元素结果为：

$$
l_i=\frac{1}{2}\left(\log(v_i)+\frac{(input_i-target_i)^2}{v_i}\right)
+\begin{cases}
\frac{1}{2}\log(2\pi), & full=true \\
0, & full=false
\end{cases}
$$

`reduction` 为 `none` 时输出 $l_i$；为 `sum` 时输出元素和；为 `mean` 时输出元素均值。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 | Shape 规格 |
| --- | --- | --- | --- | --- | --- |
| input | 输入 | 高斯分布的预测均值。 | FLOAT、FLOAT16、BFLOAT16 | ND | 输出的逻辑 shape。 |
| target | 输入 | 目标值。 | 与 `input` 一致 | ND | 与 `input` 相同，或同 rank 且恰有一个广播维度为 1。 |
| var | 输入 | 高斯分布的预测方差，元素应非负。 | 与 `input` 一致 | ND | 与 `input` 相同、最后一维为 1、比 `input` 少最后一维，或为标量。 |
| full | 可选属性 | 是否加入常数项 $\frac{1}{2}\log(2\pi)$，默认 `false`。 | BOOL | - | `true` 或 `false`。 |
| eps | 可选属性 | 方差下限，默认 `1e-6`。 | FLOAT | - | 必须为有限正数。 |
| reduction | 可选属性 | 规约方式，默认 `mean`。 | STRING | - | 支持 `none`、`sum`、`mean`。 |
| loss | 输出 | Gaussian negative log-likelihood loss。 | 与 `input` 一致 | ND | `none` 时与 `input` 一致；其他模式含一个元素。 |

## 约束说明

- `input`、`target`、`var` 和 `loss` 的数据类型必须一致。
- `target` 只允许上述单轴广播；不支持同时广播两个或更多维度。
- `var` 元素应非负；Host 不读取 Device 数据，因此该值域由调用者保证。Kernel 对参与计算的方差执行 `max(var, eps)`。
- `var` 少最后一维和最后一维为 1 两种形式均沿 `input` 的最后一维广播；标量广播到全部元素。
- FLOAT16、BFLOAT16 在 Kernel 内转换为 FLOAT 计算，再转换回输入数据类型。
- `none`、`sum`、`mean` 分别使用独立 tiling key；dtype 差异通过 `DTYPE_INPUT` 自动实例化，不创建 dtype tiling key。
- `sum`、`mean` 多核执行使用第一段 ACLNN 接口返回的 workspace。
- 支持动态 rank 和动态 shape；运行时 tiling 使用已确定的 storage shape。
- 仅支持 Atlas A2 训练系列产品。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| --- | --- | --- |
| ACLNN 调用 | [aclnnGaussianNllLoss 接口文档](./docs/aclnnGaussianNllLoss.md)、[test_aclnn_gaussian_nll_loss.cpp](./examples/test_aclnn_gaussian_nll_loss.cpp) | 两段式接口。 |
| GE | - | 暂不提供。 |
| PyTorch | - | 暂不提供。 |

## 本地编译运行 UT

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
bash build.sh -u --ophost --ops=gaussian_nll_loss --soc=ascend910b --experimental
bash build.sh -u --opkernel --ops=gaussian_nll_loss --soc=ascend910b --experimental
```

## 本地自测 UT 覆盖率

- Host InferShape：覆盖三种 reduction、动态 shape、target 单轴广播、var 四种 shape 和非法属性/shape。
- Host Tiling：覆盖三种 dtype、广播分类、空张量、非均匀多核、多 tile、workspace、UB 和逻辑元素守恒。
- Kernel：覆盖 `eps` clamp、`full`、三种 dtype、三种 reduction、target/var 各广播模式、tail 和多核多 tile。
- Golden 脚本位于 [gaussian_nll_loss_data](./tests/ut/op_kernel/gaussian_nll_loss_data)。

## eager 调用前置条件

先 source 匹配的 CANN 环境，构建并安装包含本算子的 custom package；运行时使用 `cust`，且
`vendor_name=custom` 必须与安装包匹配：

```bash
bash build.sh --run_example gaussian_nll_loss eager cust \
  --example_name=gaussian_nll_loss --vendor_name=custom --experimental
```

## 参考资源

- [PyTorch GaussianNLLLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html)
