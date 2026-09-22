# SwigluMxQuant

## 产品支持情况

| 产品 | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |
| <term>kirin X90处理器系列产品</term> | × |
| <term>kirin 9030处理器系列产品</term> | × |

## 功能说明

- 算子功能：融合算子，实现SwiGLU激活函数与动态块量化的组合计算。先对输入计算SwiGLU激活函数，然后对结果进行基于块的动态量化，输出低精度的FP4/FP8张量和对应的缩放因子。

- 计算公式：

  **阶段1：SwiGLU激活函数**

  <p style="text-align: center">
  gate, hidden = split(x, dim=dim)
  </p>
  <p style="text-align: center">
  swish = sigmoid(gate) * gate
  </p>
  <p style="text-align: center">
  act = swish * hidden
  </p>

  **阶段2：动态块量化**

  沿axis维度按blocksize=32分块进行动态量化。scale_alg=0（OCP）时，每个块内的元素 {V_i} 按以下公式量化：

  <p style="text-align: center">
  shared_exp = floor(log2(max_i(|V_i|))) - emax
  </p>
  <p style="text-align: center">
  mxscale = 2^shared_exp
  </p>
  <p style="text-align: center">
  y_i = cast_to_dst_type(V_i / mxscale, round_mode)
  </p>

  其中，emax为目标数据类型最大正则数的指数位，取值如下：

  | 数据类型 | emax |
  | :------: | :--: |
  | FLOAT4_E2M1 | 2 |
  | FLOAT4_E1M2 | 0 |
  | FLOAT8_E4M3FN | 8 |
  | FLOAT8_E5M2 | 15 |

  scale_alg=1（cuBLAS，仅FP8类型）时采用块缩放因子推导，公式详见[aclnnSwigluMxQuant](./docs/aclnnSwigluMxQuant.md)。

## 参数说明

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 120px">
  <col style="width: 100px">
  <col style="width: 380px">
  <col style="width: 280px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出/属性</th>
      <th>描述</th>
      <th>数据类型</th>
      <th>数据格式</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>输入</td>
      <td>输入张量，在activate_dim指定的维度上的尺寸必须是2的倍数。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>group_index</td>
      <td>可选输入</td>
      <td>shape必须为1维，且shape[0]大于0且小于等于256；data的每个值必须为大于等于0的整数，且所有值的和必须小于等于需要量化的x的总行数。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>输出</td>
      <td>量化后的输出张量，形状与x相同，activate_dim维度为x的一半。</td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>mxscale</td>
      <td>输出</td>
      <td>每个量化块（32个元素一组）的缩放因子。shape在axis轴上为y对应轴除以32向上取整后按偶数对齐，最后一维固定为2，存放相邻两个量化块的scale；当axis=-2且group_index存在时，shape在axis轴上为y对应轴的值整除64再加group_num。</td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>activate_dim</td>
      <td>属性</td>
      <td>SwiGLU的分割维度，取值范围为[-1, -2, xDim-2, xDim-1]（xDim为输入x的维度，即最后两维）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>activate_left</td>
      <td>属性</td>
      <td>表示对输入的前半部分或后半部分做SwiGLU激活，false时激活后半部分；swiglu_mode=1时该参数不生效。</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>swiglu_mode</td>
      <td>属性</td>
      <td>SwiGLU计算模式：0=传统SwiGLU，1=奇偶交错变体，2=前后分半clamp变体，3=前后分半sigmoid-clamp变体，取值范围[0, 3]。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>clamp_limit</td>
      <td>属性</td>
      <td>变体SwiGLU（swiglu_mode=1/2/3）的clamp门限，需大于0。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_alpha</td>
      <td>属性</td>
      <td>变体SwiGLU（swiglu_mode=1/2）的sigmoid缩放系数。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>glu_bias</td>
      <td>属性</td>
      <td>变体SwiGLU（swiglu_mode=1/2）的线性部分偏置。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>group_mode</td>
      <td>属性</td>
      <td>group_index存在时生效，0=count模式，1=cumsum模式，当前仅支持0。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis</td>
      <td>属性</td>
      <td>量化轴，沿此维度进行分块量化，取值范围为[-1, -2, xDim-2, xDim-1]（xDim为输入x的维度，即最后两维）。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dst_type</td>
      <td>属性</td>
      <td>目标量化类型：40=FP4_E2M1, 41=FP4_E1M2, 36=FP8_E4M3FN, 35=FP8_E5M2。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>round_mode</td>
      <td>属性</td>
      <td>舍入模式，用于量化时的类型转换，取值为"rint"、"floor"、"round"。</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scale_alg</td>
      <td>属性</td>
      <td>缩放算法：0=OCP，1=cuBLAS，2=RNE（预留），当前仅支持0和1。</td>
      <td>INT64</td>
      <td>-</td>
    </tr>
    <tr>
      <td>max_dtype_value</td>
      <td>属性</td>
      <td>预留参数，当前版本未生效。</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
  </tbody></table>

## 约束说明

- 输入x支持2-7维张量，在activate_dim指定维度上的尺寸必须能被2整除。
- activate_dim和axis必须取输入x的最后两维，取值范围为[-1, -2, xDim-2, xDim-1]（xDim为输入x的维度）。
- activate_dim为-2时，swiglu_mode必须为0。
- swiglu_mode为2或3时，axis必须为-1。
- 当activate_dim = -2 或者axis = -2, group_index存在时，输入x必须为2维。
- 当dst_type为FP4类型时，输出shape的最后一维必须能被2整除。
- 当dst_type为FP4类型时，scale_alg必须为0。
- group_index存在时，必须为1维，且shape[0]大于0且小于等于256；data的每个值必须为大于等于0的整数，且所有值的和必须小于等于需要量化的x的总行数。
- FP8输出类型仅支持"rint"舍入模式。
- 不支持空Tensor：输入x、可选输入group_index及输出y、mxscale的shape中任一维度为0（元素数为0）时，算子校验失败并报错。
- 输入x必须为2-7维，不支持rank=0标量Tensor。
- 输出y/mxscale由调用方按接口推导的形状分配内存，算子将量化结果完整写回输出Tensor；超出group_index所有元素之和覆盖范围的输出部分不清理，为垃圾数据。
- 输入与输出的内存空间不能重叠，不支持原地修改（inplace）场景。
- 算子执行过程中不修改任何输入数据。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
| :------- | :------- | :--- |
| aclnn调用 | [test_aclnn_swiglu_mx_quant](./examples/arch35/test_aclnn_swiglu_mx_quant.cpp) | 通过[aclnnSwigluMxQuant](./docs/aclnnSwigluMxQuant.md)接口方式调用SwigluMxQuant算子。 |
