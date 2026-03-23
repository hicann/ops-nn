# QuantBatchMatmulInplaceAdd

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas 200I/500 A2推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

## 功能说明

- 算子功能：在 micro-batch 训练场景中，需要做 micro-batch 的梯度累计，会存在大量 QuantBatchMatmul 后接 InplaceAdd 的融合场景。QuantBatchMatmulInplaceAdd 算子将上述算子融合起来，提高网络性能。实现量化矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。

- 计算公式：

  - **mx 量化：**

    $$
    y[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice)) * (scale1[m, j] * scale2[j, n])) + y[m,n]
    $$

    其中，$gsK$ 代表 K 轴的量化的 block size 即 32，$x1Slice$ 代表 $x1$ 第 m 行长度为 $gsK$ 的向量，$x2Slice$ 代表 $x2$ 第 n 列长度为 $gsK$ 的向量，K 轴均从 $j*gsK$ 起始切片，j 的取值范围 [0, kLoops)，kLoops=ceil($K_i$ / $gsK$)，支持最后的切片长度不足 $gsK$。

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
    <td class="tg-22a9">矩阵乘运算中的左矩阵，公式中的输入 x1。</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">矩阵乘运算中的右矩阵，公式中的输入 x2。</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x1Scale</td>
    <td class="tg-22a9">可选输入</td>
    <td class="tg-22a9">量化参数中由 x1 量化引入的缩放因子，对应公式的 scale1。</td>
    <td class="tg-22a9">FLOAT8_E8M0</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2Scale</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">量化参数中由 x2 量化引入的缩放因子，对应公式的 scale2。</td>
    <td class="tg-22a9">FLOAT8_E8M0</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">yRef</td>
    <td class="tg-22a9">输入输出</td>
    <td class="tg-22a9">矩阵乘运算与加法累加后的结果，对应公式中的输入输出 y。</td>
    <td class="tg-22a9">FLOAT32</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX1</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">表示 x1 的输入 shape 是否转置。</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX2</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">表示 x2 的输入 shape 是否转置。</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">groupSize</td>
    <td class="tg-22a9">输入</td>
    <td class="tg-22a9">整数型参数，用于输入 m、n、k 方向上的量化分组大小。</td>
    <td class="tg-22a9">INT64</td>
    <td class="tg-22a9">-</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x1、x2 只支持 FLOAT8_E4M3FN、FLOAT8_E5M2 数据类型。
  - x1Scale、x2Scale 只支持 FLOAT8_E8M0 数据类型。
  - yRef 只支持 FLOAT32 数据类型。
  - 当前仅支持 transposeX1 为 true，transposeX2 为 false。
  - groupSize 由 groupSizeM、groupSizeN、groupSizeK 拼接组成：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。

## 约束说明

- 支持连续 tensor，[非连续 tensor](../../docs/zh/context/非连续的Tensor.md) 支持转置场景。
- 当前仅支持 transposeX1 为 true，transposeX2 为 false。
- groupSize 相关约束：传入的 groupSize 内部会按公式分解得到 groupSizeM、groupSizeN、groupSizeK，当其中有 1 个或多个为 0 时，会根据 x1/x2/x1Scale/x2Scale 输入 shape 重新推断。
- 动态量化（mx 量化）场景约束：
  - 输入和输出支持以下数据类型组合：x1、x2 为 FLOAT8_E5M2/FLOAT8_E4M3FN，x1Scale、x2Scale 为 FLOAT8_E8M0，yRef 为 FLOAT32。
  - x1 shape 为 (k, m)，x2 shape 为 (k, n)，x1Scale shape 为 (ceil(k/64), m, 2)，x2Scale shape 为 (ceil(k/64), n, 2)，yRef shape 为 (m, n)，[gsM, gsN, gsK] 为 [1, 1, 32]，groupSize 为 32。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口 | [test_aclnn_quant_batch_matmul_inplace_add_mxfp8](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp) | 通过 [aclnnQuantBatchMatmulInplaceAdd](docs/aclnnQuantBatchMatmulInplaceAdd.md) 方式调用 QuantBatchMatmulInplaceAdd 算子。 |
