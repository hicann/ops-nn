# FusedMatmulGelu

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |
## 功能说明

- 算子功能：

  `FusedMatmulGelu` 算子用于实现全连接线性变换、可选偏置加法以及 GELU 激活函数的融合计算。

- 计算公式：

  $$
  y = GELU(x * weight^T + bias)
  $$

  当 `bias` 为空时，计算公式为：

  $$
  y = GELU(x * weight^T)
  $$

  其中，`x` 为输入张量，`weight` 为权重张量，`bias` 为可选偏置张量，`y` 为输出张量。

## 参数说明

<table>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>表示MatMul计算的输入张量。</td>
      <td>数据类型需要与weight、bias、y保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[..., K]</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示MatMul计算的权重张量。</td>
      <td>最后一维需要与x的最后一维保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[N, K]</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>输入</td>
      <td>表示MatMul计算后的可选偏置张量。</td>
      <td>可为空；不为空时shape需要与输出最后一维一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[N]</td>
    </tr>
    <tr>
      <td>approximate</td>
      <td>属性</td>
      <td>表示GELU计算模式。</td>
      <td>仅支持1，表示tanh近似模式。</td>
      <td>INT</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>表示融合计算后的输出张量。</td>
      <td>数据类型需要与x保持一致。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[..., N]</td>
    </tr>
  </tbody>
</table>

## 约束说明

- `x`、`weight`、`bias` 和 `y` 的数据类型需要保持一致。
- `x` 的最后一维大小需要与 `weight` 的最后一维大小保持一致。
- `bias` 为可选输入；当 `bias` 不为空时，其shape需要为 `[N]`。
- `approximate` 当前仅支持取值为1（tanh近似模式）。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn接口 | [test_aclnn_fused_matmul_gelu.cpp](examples/test_aclnn_fused_matmul_gelu.cpp) | 通过[aclnnFusedMatmulGelu](docs/aclnnFusedMatmulGelu.md)接口方式调用FusedMatmulGelu算子。 |
