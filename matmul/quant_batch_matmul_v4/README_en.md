# QuantBatchMatmulV4

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
|Ascend 950PR&950DT products|√|
|Atlas A3 products|√|
|Atlas A2 products|√|
|Atlas 200I/500 A2 inference products|x|
|Atlas inference products|x|
|Atlas training products|x|
|Kirin X90 processor series products|√|
|Kirin 9030 processor series products|√|

## Function

- Operator function: Performs quantized matrix multiplication.
- Formula:

  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:
    - x1 is INT8, x2 is INT32, x1Scale is FLOAT32, x2Scale is UINT64, yOffset is FLOAT32, and out is FLOAT16/BFLOAT16:

      $$
      out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
      $$

    - Without x1Scale and without bias:

      $$
      out = x1@x2 * x2Scale + x2Offset
      $$

    - bias INT32:

      $$
      out = (x1@x2 + bias) * x2Scale + x2Offset
      $$

    - bias BFLOAT16/FLOAT32 (no offset in this scenario):

      $$
      out = x1@x2 * x2Scale + bias
      $$

    - With x1Scale and without bias:

      $$
      out = x1@x2 * x2Scale * x1Scale
      $$

    - With x1Scale and bias INT32 (no offset in this scenario):

      $$
      out = (x1@x2 + bias) * x2Scale * x1Scale
      $$

    - With x1Scale and bias BFLOAT16/FLOAT16/FLOAT32 (no offset in this scenario):

      $$
      out = x1@x2 * x2Scale * x1Scale + bias
      $$

    - x1 and x2 are INT8, x1Scale and x2Scale are FLOAT32, bias is FLOAT32, and out is FLOAT16/BFLOAT16 (pertoken-pergroup quantization):

      $$
      out = (x1 @ x2) * x1Scale * x2Scale + bias
      $$

    - x1 and x2 are INT4, x1Scale and x2Scale are FLOAT32, x2Offset is FLOAT16, and out is FLOAT16/BFLOAT16 (pertoken-pergroup asymmetric quantization):

      $$
      out = x1Scale * x2Scale @ (x1 @ x2 - x1 @ x2Offset)
      $$

  - <term>Ascend 950PR&950DT products</term>:

    - x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, without x1Scale, x2Scale is INT64/UINT64, without x2Offset, the dtype of the optional parameter bias is FLOAT32, and out is FLOAT16/BFLOAT16/FLOAT32:

      $$
      out = (x1@x2 + bias) * x2Scale
      $$

    - In the MX [quantization mode](../../docs/en/context/quant_mode_introduction.md), x1 and x2 are FLOAT4_E2M1/FLOAT8_E4M3FN/FLOAT8_E5M2, x1Scale is FLOAT8_E8M0, x2Scale is FLOAT8_E8M0, without x2Offset, and the dtype of the optional parameter bias is FLOAT32:

      $$
      out = (x1* x1Scale)@(x2* x2Scale) + bias
      $$

    - x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is FLOAT32, x2Scale is FLOAT32, without x2Offset, and the dtype of the optional parameter bias is FLOAT32:

      $$
      out = (x1@x2 + bias) * x2Scale * x1Scale
      $$

    - In the G-B && B-B [quantization mode](../../docs/en/context/quant_mode_introduction.md), x1 and x2 are FLOAT8_E4M3FN/FLOAT8_E5M2/HIFLOAT8, x1Scale is FLOAT32, x2Scale is FLOAT32, without x2Offset and without bias. When x1 is (a0, a1) and x2 is (b0, b1), x1Scale is (ceil(a0 / 128), ceil(a1 / 128)) or (a0, ceil(a1 / 128)), and x2Scale is (ceil(b0 / 128), ceil(b1 / 128)):

      $$
      out_{pq} = \sum_{0}^{\left \lfloor \frac{k}{blockSize} \right \rfloor} (x1_{pr}@x2_{rq}*(x1Scale_{pr}*x2Scale_{rq}))
      $$

    - x1 is FLOAT8_E4M3FN, x2 is FLOAT4_E2M1, x1Scale is FLOAT8_E8M0, x2Scale is FLOAT8_E8M0, without x1Offset and without x2Offset, the dtype of the optional parameter bias is BFLOAT16/FLOAT16, and out is BFLOAT16/FLOAT16:

      $$
      out = (x1 * x1Scale)@(x2 * x2Scale) + bias
      $$

    - x1 is FLOAT8_E4M3FN, x2 is FLOAT4_E2M1, without x1Scale, x2Scale is BFLOAT16/FLOAT16, without x1Offset and without x2Offset, without bias, yScale is UINT64, and out is BFLOAT16/FLOAT16:

      $$
      out = (x1@(x2 * x2Scale)) * yScale
      $$

## Parameters

<table class="tg" style="undefined;table-layout: fixed; width: 1166px"><colgroup>
<col style="width: 81px">
<col style="width: 121px">
<col style="width: 430px">
<col style="width: 390px">
<col style="width: 144px">
</colgroup>
<thead>
  <tr>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-xbcz"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Left matrix in the matrix multiplication.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">FLOAT8_E5M2, FLOAT8_E4M3FN, INT4, INT8</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Right matrix in the matrix multiplication.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">FLOAT4_E2M1, INT4, INT8</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND, FRACTAL_NZ</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">bias</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Bias accumulated after the matrix multiplication, corresponding to bias in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x1_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Scaling factor of the quantization parameters during matrix multiplication, corresponding to x1Scale in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16, FLOAT32, FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x2_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Scaling factor of the quantization parameters during matrix multiplication, corresponding to x2Scale in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32, UINT64, FLOAT8_E8M0</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y_scale</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Scaling factor of the quantization parameters after matrix multiplication, corresponding to yScale in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">UINT64</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">x1_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Offset factor of the quantization parameters during matrix multiplication, corresponding to x1Offset in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">x2_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Offset factor of the quantization parameters during matrix multiplication, corresponding to x2Offset in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">y_offset</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Input</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">Offset factor of the quantization parameters after matrix multiplication, corresponding to yOffset in the formula.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">BFLOAT16, FLOAT16, FLOAT32</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--theme-table-header-bg)">ND</span></td>
  </tr>
  <tr>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">y</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Output</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">Computation result of the matrix multiplication.</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">BFLOAT16, FLOAT16</span></td>
    <td class="tg-zgfj"><span style="color:var(--theme-aide-text);background-color:var(--devui-base-bg, #ffffff)">ND</span></td>
  </tr>
</tbody></table>

## Constraints

- Empty tensors are not supported.
- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support only the transpose scenario.
- The inputs and outputs support the following data type combinations:

  - <term>Atlas A2 products</term> and <term>Atlas A3 products</term>:

    | x1                        | x2                        | x1_scale     | x2_scale         | x2_offset    | y_scale   | bias         | y_offset    | y                                    |
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

  - <term>Ascend 950PR&950DT products</term>:

    | x1                        | x2                        | x1_scale     | x2_scale     | x2_offset | y_scale | bias    | y                                    |
    | ------------------------- | ------------------------- | ----------- | ----------- | -------- | -------| ------- | -------------------------------------- |
    | INT8                      | INT8                      | nullptr        | UINT64/INT64      | nullptr     | nullptr     | nullptr/INT32   | FLOAT16/BFLOAT16                       |
    | INT8                      | INT8                      | nullptr        | UINT64/INT64      | nullptr/FLOAT32  | nullptr     | nullptr/INT32   | INT8                              |
    | INT8                      | INT8                      | nullptr/FLOAT32| FLOAT32/BFLOAT16  | nullptr     | nullptr     | nullptr/INT32/FLOAT32/BFLOAT16   | BFLOAT16              |
    | INT8                      | INT8                      | FLOAT32     | FLOAT32           | nullptr     | nullptr     | nullptr/INT32/FLOAT32/FLOAT16  | FLOAT16                 |
    | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | nullptr        | UINT64/INT64      | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32            |
    | HIFLOAT8                  | HIFLOAT8                  | nullptr        | UINT64/INT64      | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32            |
    | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT32     | FLOAT32           | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
    | HIFLOAT8                  | HIFLOAT8                  | FLOAT32     | FLOAT32           | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
    | FLOAT4_E2M1               | FLOAT4_E2M1               | FLOAT8_E8M0 | FLOAT8_E8M0       | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
    | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0 | FLOAT8_E8M0       | nullptr     | nullptr     | nullptr/FLOAT32 | FLOAT16/BFLOAT16/FLOAT32               |
    | FLOAT8_E4M3FN             | FLOAT4_E2M1               | FLOAT8_E8M0 | FLOAT8_E8M0       | nullptr     | nullptr     | nullptr/BFLOAT16/FLOAT16| BFLOAT16/FLOAT16                               |
    | FLOAT8_E4M3FN             | FLOAT4_E2M1               | nullptr        | BFLOAT16/FLOAT16 | nullptr     | INT64/UINT64    | nullptr  | BFLOAT16/FLOAT16               |

## Invocation Description

- <term>Atlas A2 products and Atlas A3 products</term>:

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn API  | [test_aclnn_quant_matmul_v5](examples/test_aclnn_quant_matmul_v5_at2_at3.cpp) | Invoke the QuantBatchMatmulV4 operator through<br>[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5_en.md)<br>and other APIs. |

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn API  | [test_aclnn_quant_matmul_v5](examples/arch35/test_aclnn_quant_matmul_v5_f8f4_nd.cpp) | Invoke the QuantBatchMatmulV4 operator through<br>[aclnnQuantMatmulV5](docs/aclnnQuantMatmulV5_en.md)<br>and other APIs. |
