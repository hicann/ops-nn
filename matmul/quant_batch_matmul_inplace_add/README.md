# QuantBatchMatmulInplaceAdd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- 算子功能：在micro-batch训练场景中，需要做micro-batch的梯度累计，会存在大量QuantBatchMatmul后接InplaceAdd的融合场景。QuantBatchMatmulInplaceAdd算子将上述算子融合起来，提高网络性能。实现量化矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。

- 计算公式：

  - **MX量化：**

    $$
    y[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice)) * (scale1[m, j] * scale2[j, n])) + y[m,n]
    $$

    其中，$gsK$ 代表K轴的量化的block size即32，$x1Slice$ 代表 $x1$ 第m行长度为 $gsK$ 的向量，$x2Slice$ 代表 $x2$ 第n列长度为 $gsK$ 的向量，K轴均从 $j*gsK$ 起始切片，j的取值范围 [0, kLoops)，kLoops=ceil($K_i$ / $gsK$)，支持最后的切片长度不足 $gsK$。

  - **HIFLOAT8 T-T量化：**

    $$
    y[m,n] = (\sum_{k=0}^{K_i-1} (x1[m,k] * x2[k,n])) * (scale1 * scale2) + y[m,n]
    $$

    其中，$scale1$ 和 $scale2$ 分别对应 $x1Scale$ 和 $x2Scale$，均为标量（shape为 $(1)$）；当设置`transposeX1`/`transposeX2`时，公式中的 $x1$ 和 $x2$ 按转置后的视图参与计算。

## 参数说明

<table class="tg"><thead>
  <tr>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">参数名</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">输入/输出/属性</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">描述</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据类型</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据格式</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9">x1</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">矩阵乘运算中的左矩阵，公式中的输入x1。</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">矩阵乘运算中的右矩阵，公式中的输入x2。</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x1Scale</td>
    <td class="tg-22a9">可选输入</td>
    <td class="tg-22a9">量化参数中由x1量化引入的缩放因子，对应公式的scale1。</td>
    <td class="tg-22a9">FLOAT8_E8M0（MX）、FLOAT32（HIFLOAT8 TT）</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2Scale</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">量化参数中由x2量化引入的缩放因子，对应公式的scale2。</td>
    <td class="tg-22a9">FLOAT8_E8M0（MX）、FLOAT32（HIFLOAT8 TT）</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">yRef</td>
    <td class="tg-22a9">输入输出</td>
    <td class="tg-22a9">矩阵乘运算与加法累加后的结果，对应公式中的输入输出y。</td>
    <td class="tg-22a9">FLOAT32</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX1</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">表示x1的输入shape是否转置。</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX2</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">表示x2的输入shape是否转置。</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">groupSize</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">整数型参数，用于输入m、n、k方向上的量化分组大小。</td>
    <td class="tg-22a9">INT64</td>
    <td class="tg-22a9">-</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x1、x2支持FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8数据类型。
  - x1Scale、x2Scale支持FLOAT8_E8M0（MX）和FLOAT32（HIFLOAT8 TT）数据类型。
  - yRef只支持FLOAT32数据类型。
  - 当前仅支持transposeX1=true且transposeX2=false。
  - groupSize由groupSizeM、groupSizeN、groupSizeK拼接组成：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。

## 约束说明

- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)支持转置场景。
- groupSize相关约束：传入的groupSize内部会按公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0时，会根据x1/x2/x1Scale/x2Scale输入shape重新推断。
- 动态量化（MX量化）场景约束：
  - 输入和输出支持以下数据类型组合：x1、x2为FLOAT8_E5M2/FLOAT8_E4M3FN，x1Scale、x2Scale为FLOAT8_E8M0，yRef为FLOAT32。
  - x1 shape为(k, m)，x2 shape为(k, n)，x1Scale shape为(ceil(k/64), m, 2)，x2Scale shape为(ceil(k/64), n, 2)，yRef shape为(m, n)，[gsM, gsN, gsK] 为 [1, 1, 32]，groupSize为32。
- HIFLOAT8 T-T场景约束：
  - 输入和输出支持以下数据类型组合：x1、x2为HIFLOAT8，x1Scale、x2Scale为FLOAT32，yRef为FLOAT32。
  - x1 shape为(k, m)，x2 shape为(k, n)，yRef shape为(m, n)。
  - x1Scale shape为(1)，x2Scale shape为(1)，groupSize必须为0。
  - 不支持bias、offset参数。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_quant_batch_matmul_inplace_add_mxfp8](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp) | 展示mxFP8量化场景的调用方式，其中`x1/x2`为FLOAT8、`x1Scale/x2Scale`为FLOAT8_E8M0，`groupSize=32`。 |
| aclnn接口 | [test_aclnn_quant_batch_matmul_inplace_add_TT](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_TT.cpp) | 展示当前支持组合`transposeX1=true`、`transposeX2=false`下HIFLOAT8 TT场景的调用方式。 |
