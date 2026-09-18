<!-- codespell:ignore Silu -->
<!-- cspell:ignore Silu -->

# FusedMatmulSilu

## 贡献说明

| 贡献者 | 贡献方 | 贡献算子 | 贡献时间 | 贡献内容 |
| --- | --- | --- | --- | --- |
| 社区开发者 | 个人开发者 | FusedMatmulSilu | 2026/06/22 | 新增FusedMatmulSilu算子 |

## 支持的产品型号

- <term>Ascend 910B</term>

产品形态详细说明请参见[昇腾产品形态说明](http://www.hiascend.com/document/redirect/CannCommunityProductForm)。

## 算子描述

- 功能描述

  `FusedMatmulSilu`算子融合Matmul与SiLU激活计算，输入矩阵`x`与`weight`的转置做矩阵乘，加上`bias`后执行SiLU激活。

- 计算公式

  $$
  y = silu(x @ weight^T + bias)
  $$

  $$
  silu(t) = \frac{t}{1 + e^{-t}}
  $$

- 原型信息

  <table>
    <tr><th align="center">算子类型(OpType)</th><th colspan="4" align="center">FusedMatmulSilu</th></tr>
    <tr><td align="center"> </td><td align="center">name</td><td align="center">Type</td><td align="center">data type</td><td align="center">format</td></tr>
    <tr><td rowspan="3" align="center">算子输入</td><td align="center">x</td><td align="center">tensor</td><td align="center">bfloat16</td><td align="center">ND</td></tr>
    <tr><td align="center">weight</td><td align="center">tensor</td><td align="center">bfloat16</td><td align="center">ND</td></tr>
    <tr><td align="center">bias</td><td align="center">tensor</td><td align="center">bfloat16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">tensor</td><td align="center">bfloat16</td><td align="center">ND</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">fused_matmul_silu</td></tr>
  </table>

## 约束与限制

- 支持Ascend 910B。
- 仅支持BF16数据类型和ND数据格式。
- `x`为二维Tensor，shape为`[M, K]`。
- `weight`为二维Tensor，shape为`[N, K]`，计算时按`weight^T`参与矩阵乘。
- `bias`为一维必选Tensor，shape为`[N]`。
- `y`为二维Tensor，shape为`[M, N]`。
- 支持动态shape，Tensor rank固定为2/2/1/2；运行时实际`K`需满足下述约束。
- 当前实现要求`K`为64的整数倍，且`K <= 4096`。
- 当前实现使用`MatmulImpl`执行Cube矩阵乘主路径，并使用`MultiCoreMatmulTiling`进行多核切分；bias与SiLU由AIV后处理完成。
- 精度路径：Cube完成FP32累加后将Matmul结果转换为BF16并写入GM；AIV读取该BF16中间结果，转换为FP32后执行bias Add和SiLU，最终转换为BF16写回输出。因此结果包含一次Matmul中间结果的BF16截断

## 运行验证

编译：

```bash
bash build.sh --pkg --soc=ascend910b --vendor_name=custom_nn --ops=fused_matmul_silu --experimental
```

安装自定义算子包后，使用框架提供的示例执行门禁运行样例：

```bash
bash build.sh --run_example fused_matmul_silu eager cust --vendor_name=custom_nn --soc=ascend910b --experimental
```

已验证参考shape：

| x shape | weight shape | bias shape | y shape |
| --- | --- | --- | --- |
| [2, 256] | [4096, 256] | [4096] | [2, 4096] |
| [2, 4096] | [4096, 4096] | [4096] | [2, 4096] |
| [2, 4096] | [2048, 4096] | [2048] | [2, 2048] |

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [examples](./examples) | 通过[aclnnFusedMatmulSilu](./docs/aclnnFusedMatmulSilu.md)接口方式调用FusedMatmulSilu算子。 |
