# QuantBatchMatmulV4


##  产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件|√|

## 参数说明

- 算子功能：完成量化的矩阵乘计算。
- 计算公式：

  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64，yOffset为FLOAT32，out为FLOAT16/BFLOAT16：

      $$
      out = ((x1 @ (x2*x2Scale)) + yoffset) * x1Scale
      $$

    - 无x1Scale无bias：

      $$
      out = x1@x2 * x2Scale + x2Offset
      $$

    - bias INT32：

      $$
      out = (x1@x2 + bias) * x2Scale + x2Offset
      $$

    - bias BFLOAT16/FLOAT32（此场景无offset）：

      $$
      out = x1@x2 * x2Scale + bias
      $$

    - x1Scale无bias：

      $$
      out = x1@x2 * x2Scale * x1Scale
      $$

    - x1Scale， bias INT32（此场景无offset）：

      $$
      out = (x1@x2 + bias) * x2Scale * x1Scale
      $$

    - x1Scale， bias BFLOAT16/FLOAT16/FLOAT32（此场景无offset）：

      $$
      out = x1@x2 * scale * x1Scale + bias
      $$
    
    - x1，x2为INT8，x1Scale, x2Scale为FLOAT32，bias为FLOAT32，out为FLOAT16/BFLOAT16  (pertoken-pergroup量化):

      $$
      out = (x1 @ x2) * x1Scale * x2Scale + bias
      $$

    - x1，x2为INT4，x1Scale, x2Scale为FLOAT32，x2Offset为FLOAT16, out为FLOAT16/BFLOAT16 (pertoken-pergroup非对称量化):

      $$
      out = x1Scale * x2Scale @ (x1 @ x2 - x1 @ x2Offset)
      $$

## 算子规格

<table class="tg" style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 81px">
<col style="width: 121px">
<col style="width: 430px">
<col style="width: 390px">
<col style="width: 144px">
</colgroup>
<thead>
  <tr>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">参数名</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">输入/输出/属性</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">描述</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据类型</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">数据格式</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算中的左矩阵。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E5M2, FLOAT8_E4M3FN, INT4, INT8</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算中的右矩阵。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT4_E1M2, FLOAT4_E2M1, INT4, INT8</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算后累加的偏置，对应公式中的bias。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x1_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘计算时，量化参数的缩放因子，对应公式的x1Scale。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16, FLOAT32, FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x2_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘计算时，量化参数的缩放因子，对应公式的x2Scale。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32, UINT64, FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算后，量化参数的缩放因子，对应公式的yScale。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">UINT64</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘计算时，量化参数的偏置因子，对应公式的x1Offset。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘计算时，量化参数的偏置因子，对应公式的x2Offset。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">输入</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">矩阵乘运算后，量化参数的偏置因子，对应公式的yOffset。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">输出</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">矩阵乘运算的计算结果。</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
</tbody></table>

- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件、Atlas A3 训练系列产品/Atlas A3 推理系列产品：

  | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT32                     | FLOAT32     | UINT64          | nullptr        | nullptr     | nullptr         | FLOAT32    | FLOAT16/BFLOAT16                       |
  | INT8                      | INT8                      | nullptr        | UINT64/INT64    | nullptr        | nullptr     | nullptr/INT32   | nullptr       | FLOAT16                                |
  | INT8                      | INT8                      | nullptr        | UINT64/INT64    | nullptr/FLOAT32| nullptr     | nullptr/INT32   | nullptr       | INT8                                   |
  | INT8                      | INT8                      | nullptr/FLOAT32| BFLOAT16        | nullptr        | nullptr     | nullptr/INT32/BFLOAT16/FLOAT32   | nullptr       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | nullptr        | nullptr     | nullptr/INT32/FLOAT16/FLOAT32    | nullptr       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | nullptr        | UINT64/INT64    | nullptr        | nullptr     | nullptr/INT32   | nullptr       | INT32                                  |
  | INT8                      | INT8                      | nullptr        | FLOAT32/BFLOAT16| nullptr        | nullptr     | nullptr/INT32   | nullptr       | FLOAT16                                |
  | INT8                      | INT8                      | FLOAT32        | FLOAT32| nullptr        | nullptr     | FLOAT32   | nullptr       | BFLOAT16                                |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| nullptr        | nullptr     | nullptr/INT32/BFLOAT16/FLOAT32   | nullptr       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | nullptr        | nullptr     | nullptr/INT32/FLOAT16/FLOAT32    | nullptr       | FLOAT16               |
  | INT4                | INT4                | FLOAT32     | FLOAT32         | FLOAT16        | nullptr     | nullptr    | nullptr       | BFLOAT16               |

## 约束说明

- 不支持空tensor。
- 支持连续tensor，[非连续tensor](../../docs/zh/context/非连续的Tensor.md)只支持转置场景。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quant_matmul_v5](examples/test_aclnn_quant_matmul_v5_at2_at3.cpp) | 通过<br>[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5.md)<br>等方式调用QuantBatchMatmulV4算子。 |