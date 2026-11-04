# Dilation2D

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: the 2D implementation of the morphological dilation operation. The operator performs the sliding window operation independently on each channel of the input image: at each window position, the maximum of the sum of the input value and the filter value is taken as the output.

- Formula:

$$
y[b, oh, ow, c] = \max_{fh, fw} (x[b, oh \times stride_h + fh \times rate_h - pad\_top, ow \times stride_w + fw \times rate_w - pad\_left, c] + filter[fh, fw, c])
$$

Here $pad\_top$ and $pad\_left$ are determined by the padding mode: in SAME/CALCULATED mode they are computed by the formula below, and in VALID mode they are 0. Positions beyond the input boundaries are treated as negative infinity and do not participate in the max comparison.

## Output Size Computation

Let the effective filter size be $filter\_h\_eff = fH + (fH-1) \times (rate\_h-1)$, and $filter\_w\_eff$ is computed in the same way.

- SAME: $oH = \lceil H / stride\_h \rceil$ and $oW = \lceil W / stride\_w \rceil$; the pad is computed as $pad_h = \max((oH-1) \times stride\_h + filter\_h\_eff - H, 0)$ and then split evenly between top and bottom ($pad\_top = pad\_h // 2$).
- VALID: $oH = \max(0, \lfloor (H - filter\_h\_eff + stride\_h) / stride\_h \rfloor)$, and $oW$ is computed in the same way; there is no padding.
- CALCULATED: the pads attribute explicitly specifies $[H\_top, H\_bottom, W\_left, W\_right]$. Let $hNum = H + H\_top + H\_bottom - filter\_h\_eff$:
  - ceil_mode=false (floor convention): $oH = \max(0, \lfloor hNum / stride\_h \rfloor + 1)$; when $hNum < 0$, the output is empty ($oH = 0$).
  - ceil_mode=true (ceil convention): $oH = \max(0, \lceil hNum / stride\_h \rceil + 1)$.
  - $oW$ is computed in the same way.

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>4D input tensor in NHWC or NCHW format. In NHWC, the filter shape is (fH,fW,C); in NCHW, it is (C,fH,fW).</td>
      <td>FLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>filter</td>
      <td>Input</td>
      <td>3D filter tensor, whose C dimension is consistent with the C of x. In NHWC, the shape is (fH,fW,C); in NCHW, it is (C,fH,fW).</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>4D output tensor, with oH/oW derived from the input shape and attributes.</td>
      <td>Same as x</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Optional attribute</td>
      <td>Four positive integers, the sliding window stride. In NHWC, the value is [sN, sH, sW, sC]; in NCHW, it is [sN, sC, sH, sW]. Value range: [1, 2147483647].</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rates</td>
      <td>Optional attribute</td>
      <td>Four positive integers, the dilation rates. In NHWC, the value is [rN, rH, rW, rC]; in NCHW, it is [rN, rC, rH, rW]. Value range: [1, 2147483647].</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Optional attribute</td>
      <td>Padding mode: "SAME", "VALID", or "CALCULATED". The default value is "SAME".</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Optional attribute</td>
      <td>Four non-negative integers, the explicit padding values [H_top, H_bottom, W_left, W_right]. The default value is {0,0,0,0}, which takes effect only in CALCULATED mode. Value range: [0, 2147483647].</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Optional attribute</td>
      <td>Whether the output size computation uses ceil or floor. The default value is false. It takes effect only in CALCULATED mode.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Optional attribute</td>
      <td>Data format: "NHWC" or "NCHW". The default value is "NHWC".</td>
      <td>String</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- The input x must be a 4D tensor, and the filter must be a 3D tensor.
- The dtypes of x and filter must be consistent; only FLOAT16 is supported.
- The channel count (C dimension) of the filter must be consistent with the channel count of x. In NHWC, the filter shape is (fH,fW,C); in NCHW, it is (C,fH,fW).
- The input dimensions and the values of strides/rates/pads must not exceed 2147483647 (the upper limit of int32); inputs that exceed this limit are rejected.
- strides and rates must be four positive integers. In NHWC, strides[0]=strides[3]=1 and rates[0]=rates[3]=1; in NCHW, strides[0]=strides[1]=1 and rates[0]=rates[1]=1.
- In CALCULATED mode, the pads must be smaller than the effective filter size, and the pads format is [H_top, H_bottom, W_left, W_right].
- Positions beyond the input boundaries are treated as negative infinity and do not participate in the max comparison.
- The three padding modes SAME, VALID, and CALCULATED are supported.
- Dynamic shapes are supported: when the input contains -1 (unknown dimension), the output is derived from the computable dimensions, and a dimension that cannot be computed outputs -1.
- Only Ascend 950 is supported.

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| GE graph mode | [test_geir_dilation2_d](./examples/arch35/test_geir_dilation2_d.cpp) | Calls the Dilation2D operator by building a graph through [operator IR](./op_graph/dilation2_d_proto.h). |
