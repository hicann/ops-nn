# MatmulEmuSplitWeight

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    ×     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 算子功能：使用双路BF16 GEMM融合模拟FP32精度矩阵乘法。离线将FP32权重拆分为高位BF16与低位残差BF16，推理阶段执行两次BF16 GEMM并做线性组合，激活值全程保持BF16，两路矩阵乘均运行在Cube上。

- 计算公式：

  $$
  Y = X \times W_{high} + scale \times (X \times W_{low})
  $$

  其中 $X$ 为BF16激活矩阵，$W_{high}$ 为FP32权重截断BF16得到的高位权重，$W_{low}$ 为FP32权重残差除以scale后截断BF16得到的低位残差权重，$scale$ 为缩放因子（典型值1/256=0.00390625）。

- 权重离线拆分公式：

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## 参数说明

  <table style="undefined;table-layout: fixed; width: 962px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>激活矩阵，对应公式中的X。transpose_x=false时shape为[M, K]，transpose_x=true时shape为[K, M]。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_high</td>
      <td>输入</td>
      <td>高位权重，对应公式中的W_high。transpose_w=false时shape为[K, N]，transpose_w=true时shape为[N, K]。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_low</td>
      <td>输入</td>
      <td>低位残差权重，对应公式中的W_low。shape必须与w_high完全一致。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>计算输出矩阵，对应公式中的Y。仅支持FP32输出（yDtype=0）。shape为[M, N]。</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_low_scale</td>
      <td>属性</td>
      <td>缩放因子，对应公式中的scale。仅支持1/256（0.00390625）。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_x</td>
      <td>属性</td>
      <td>指定是否对输入x进行转置。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_w</td>
      <td>属性</td>
      <td>指定是否对权重w_high/w_low进行转置。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>属性</td>
      <td>输出数据类型标识。仅支持0（FP32），其他值将报错。</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## 约束说明
- x、w_high、w_low、y均为2维。
- x的K维度必须与w_high/w_low的K维度一致。
- M、K、N取值范围为(0, INT32_MAX]。
- transpose_x和transpose_w默认为false。

## 调用说明

<table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 170px">
    <col style="width: 300px">
    <col style="width: 430px">
    </colgroup>
    <thead>
      <tr>
        <th>调用方式</th>
        <th>样例代码</th>
        <th>说明</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>aclnn接口</td>
        <td><a href="examples/arch35/test_aclnn_matmul_emu_split_weight.cpp">test_aclnn_matmul_emu_split_weight</a></td>
        <td>通过<a href="docs/aclnnMatmulEmuSplitWeight.md">aclnnMatmulEmuSplitWeight</a>接口方式调用MatmulEmuSplitWeight算子</td>
      </tr>
      <tr>
        <td>torch接口</td>
        <td><a href="torch_extension/matmul_emu_split_weight.py">matmul_emu_split_weight</a></td>
        <td>通过<a href="docs/torchapi_matmul_emu_split_weight.md">torchapi</a>接口方式调用MatmulEmuSplitWeight算子</td>
      </tr>
  </tbody></table>
