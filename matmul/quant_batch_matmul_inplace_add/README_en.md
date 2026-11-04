# QuantBatchMatmulInplaceAdd

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| ---- | :----:|
| Ascend 950PR&950DT products | √ |
| Atlas A3 products | × |
| Atlas A2 products | × |
| Atlas 200I/500 A2 inference products | × |
| Atlas inference products | × |
| Atlas training products | × |

## Function

- Operator function: in the micro-batch training scenario, gradient accumulation of micro-batches is required, and a large number of fusion scenarios where QuantBatchMatmul is followed by InplaceAdd exist. The QuantBatchMatmulInplaceAdd operator fuses the preceding operators to improve network performance. It implements quantized matrix multiplication and addition, and its basic function is a combination of matrix multiplication and addition.

- Formula:

  - **MX quantization:**

    $$
    y[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice)) * (scale1[m, j] * scale2[j, n])) + y[m,n]
    $$

    where $gsK$ denotes the quantization block size of the K axis, that is, 32; $x1Slice$ denotes the vector of length $gsK$ in row m of $x1$; $x2Slice$ denotes the vector of length $gsK$ in column n of $x2$. Slicing on the K axis starts from $j*gsK$, the value range of j is [0, kLoops), kLoops=ceil($K_i$ / $gsK$), and the case where the length of the last slice is less than $gsK$ is supported.

  - **HIFLOAT8 T-T quantization:**

    $$
    y[m,n] = (\sum_{k=0}^{K_i-1} (x1[m,k] * x2[k,n])) * (scale1 * scale2) + y[m,n]
    $$

    where $scale1$ and $scale2$ correspond to $x1Scale$ and $x2Scale$ respectively, both being scalars (shape $(1)$). When `transposeX1`/`transposeX2` is set, the $x1$ and $x2$ in the formula participate in the computation with the transposed view.

## Parameters

<table class="tg"><thead>
  <tr>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Parameter name</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Input/Output/Attribute</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Description</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data type</span></th>
    <th class="tg-hlb2"><span style="font-weight:700;color:var(--theme-text);background-color:var(--theme-table-header-bg)">Data format</span></th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-22a9">x1</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Left matrix in the matrix multiplication, corresponding to input x1 in the formula.</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Right matrix in the matrix multiplication, corresponding to input x2 in the formula.</td>
    <td class="tg-22a9">FLOAT8_E4M3FN, FLOAT8_E5M2, HIFLOAT8</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x1Scale</td>
    <td class="tg-22a9">Optional input</td>
    <td class="tg-22a9">Scaling factor introduced by the quantization of x1 in the quantization parameters, corresponding to scale1 in the formula.</td>
    <td class="tg-22a9">FLOAT8_E8M0 (MX), FLOAT32 (HIFLOAT8 TT)</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">x2Scale</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Scaling factor introduced by the quantization of x2 in the quantization parameters, corresponding to scale2 in the formula.</td>
    <td class="tg-22a9">FLOAT8_E8M0 (MX), FLOAT32 (HIFLOAT8 TT)</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">yRef</td>
    <td class="tg-22a9">Input and output</td>
    <td class="tg-22a9">Result after the matrix multiplication and the addition accumulation, corresponding to input and output y in the formula.</td>
    <td class="tg-22a9">FLOAT32</td>
    <td class="tg-22a9">ND</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX1</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Indicates whether the input shape of x1 is transposed.</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">transposeX2</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Indicates whether the input shape of x2 is transposed.</td>
    <td class="tg-22a9">bool</td>
    <td class="tg-22a9">-</td>
  </tr>
  <tr>
    <td class="tg-22a9">groupSize</td>
    <td class="tg-22a9">Input</td>
    <td class="tg-22a9">Integer parameter used for the quantization group sizes along the m, n, and k directions of the input.</td>
    <td class="tg-22a9">INT64</td>
    <td class="tg-22a9">-</td>
  </tr>
</tbody></table>

- <term>Ascend 950PR&950DT products</term>:
  - x1 and x2 support the FLOAT8_E4M3FN, FLOAT8_E5M2, and HIFLOAT8 data types.
  - x1Scale and x2Scale support the FLOAT8_E8M0 (MX) and FLOAT32 (HIFLOAT8 TT) data types.
  - yRef supports only the FLOAT32 data type.
  - Currently only transposeX1=true and transposeX2=false are supported.
  - groupSize is composed by concatenating groupSizeM, groupSizeN, and groupSizeK: groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32.

## Constraints

- Contiguous tensors are supported, and [non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) support the transposition scenario.
- groupSize-related constraints: the passed-in groupSize is internally decomposed according to the formula to obtain groupSizeM, groupSizeN, and groupSizeK. When one or more of them is 0, the values are inferred again according to the input shapes of x1/x2/x1Scale/x2Scale.
- Constraints of the dynamic quantization (MX quantization) scenario:
  - The inputs and outputs support the following data type combinations: x1 and x2 are FLOAT8_E5M2/FLOAT8_E4M3FN, x1Scale and x2Scale are FLOAT8_E8M0, and yRef is FLOAT32.
  - The shape of x1 is (k, m), the shape of x2 is (k, n), the shape of x1Scale is (ceil(k/64), m, 2), the shape of x2Scale is (ceil(k/64), n, 2), the shape of yRef is (m, n), [gsM, gsN, gsK] is [1, 1, 32], and groupSize is 32.
- Constraints of the HIFLOAT8 T-T scenario:
  - The inputs and outputs support the following data type combinations: x1 and x2 are HIFLOAT8, x1Scale and x2Scale are FLOAT32, and yRef is FLOAT32.
  - The shape of x1 is (k, m), the shape of x2 is (k, n), and the shape of yRef is (m, n).
  - The shape of x1Scale is (1), the shape of x2Scale is (1), and groupSize must be 0.
  - The bias and offset parameters are not supported.

## Invocation Description

| Invocation method | Sample code | Description |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_quant_batch_matmul_inplace_add_mxfp8](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_mxfp8.cpp) | Shows the invocation method of the mxFP8 quantization scenario, where `x1/x2` are FLOAT8, `x1Scale/x2Scale` are FLOAT8_E8M0, and `groupSize=32`. |
| aclnn interface | [test_aclnn_quant_batch_matmul_inplace_add_TT](examples/arch35/test_aclnn_quant_batch_matmul_inplace_add_TT.cpp) | Shows the invocation method of the HIFLOAT8 TT scenario under the currently supported combination `transposeX1=true` and `transposeX2=false`. |
