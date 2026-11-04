# AvgPoolUpdate

English | [简体中文](./README.md)


## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |     √    |
| <term>Atlas A2 products</term>    |     √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |     √    |
| <term>Atlas inference products</term>                               |     √    |
| <term>Atlas training products</term>                               |     √    |

## Function

- Operator function: computes the update value of average pooling. The operator divides the sum pooling result by the number of valid elements actually covered by the pooling window to obtain the average value. x1 is the sum pooling output, and x2 is the original input feature map (used only to obtain the input spatial size and not involved in numeric computation).

- Formula:

$$
y = x1 \oslash \text{mean\_matrix}
$$

Here mean_matrix is the pooling divisor matrix:

$$
\text{mean\_matrix}[h, w] = \text{mean\_h} \times \text{mean\_w}
$$

$$
\text{mean\_h} = \max(\min(\min(h \cdot s_h - p_t + k_h,\; (H_{out}-1-h) \cdot s_h - p_b + k_h),\; \min(k_h,\; H_{in})),\; 1)
$$

$$
\text{mean\_w} = \max(\min(\min(w \cdot s_w - p_l + k_w,\; (W_{out}-1-w) \cdot s_w - p_r + k_w),\; \min(k_w,\; W_{in})),\; 1)
$$

In the formulas, $s_h$ and $s_w$ are the H/W components of strides; $p_t$, $p_b$, $p_l$, and $p_r$ are the top/bottom/left/right components of pads; $k_h$ and $k_w$ are the H/W components of ksize; $H_{in}$ and $W_{in}$ are the spatial input size of x2; and $H_{out}$ and $W_{out}$ are the spatial output size of x1. The mean_matrix computation uses int64 intermediate variables, which are cast to the dtype of x1 before the division.

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
      <td>x1</td>
      <td>Input</td>
      <td>Sum pooling output result, x1 in the formula. The shape is (N, C, H_out, W_out) (NCHW) or (N, H_out, W_out, C) (NHWC).</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Original input feature map, x2 in the formula. It is used only to obtain the input spatial size H_in/W_in and is not involved in numeric computation. The shape is (N, C, H_in, W_in) (NCHW) or (N, H_in, W_in, C) (NHWC).</td>
      <td>INT4, INT8, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Average pooling update result, y in the formula. The shape is the same as x1.</td>
      <td>FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Pooling window size, a list of length 4, with the dimension order consistent with data_format.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Pooling stride, a list of length 4, with the dimension order consistent with data_format.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding_mode</td>
      <td>Attribute</td>
      <td>Padding mode. Value range: CALCULATED, VALID, SAME. The default value is CALCULATED.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Padding values, a list of length 4 (top, bottom, left, right). It takes effect only when padding_mode is CALCULATED. The default value is {0, 0, 0, 0}.</td>
      <td>ListInt</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Data format. Value range: NCHW, NHWC. The default value is NHWC.</td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceil_mode</td>
      <td>Attribute</td>
      <td>Whether to use the ceil mode to compute the output size. true means ceil, and false means floor. The default value is false.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
    <tr>
      <td>exclusive</td>
      <td>Attribute</td>
      <td>Whether to exclude the padding area from the window count. true means excluding the padding (using the number of actually covered elements), and false means using the constant window size. The default value is true.</td>
      <td>Bool</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- **exclusive constraint**: exclusive must be true. When exclusive is false, the operator is meaningless (the pooling factor is a constant and needs no update), and the operator exits with an error.
- **padding_mode and ceil_mode constraints**: when padding_mode is VALID and ceil_mode is false, the operator is meaningless (the pooling factor is a constant), and the operator exits with an error.
- **Input dimensions constraint**: x1 and x2 must be 4D tensors, supporting the two data formats NCHW and NHWC. Other dimension counts are not supported.
- **ksize constraint**: ksize is a list of length 4, and each component must be a positive integer. The positions of the H/W components (k_h, k_w) are determined by data_format.
- **strides constraint**: strides is a list of length 4, and each component must be a positive integer. The positions of the H/W components (s_h, s_w) are determined by data_format.
- **pads constraint**: pads is a list of length 4 (top, bottom, left, right), and each component must be a non-negative integer. It takes effect only when padding_mode is CALCULATED; when padding_mode is VALID, pads is ignored (set to 0); when padding_mode is SAME, pads is computed automatically by the framework.
- **data_format constraint**: only NCHW and NHWC are supported.
- **padding_mode constraint**: only the three values CALCULATED, VALID, and SAME are supported.
- **dtype constraint**: the dtypes of x1 and y must be the same (FLOAT16 or FLOAT). The dtype of x2 is independent and supports INT4, INT8, FLOAT16, and FLOAT (x2 is used only to obtain the input spatial size and is not involved in numeric computation).

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Invocation sample</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>Graph-mode invocation</td>
    <td><a href="examples/arch35/test_geir_avg_pool_update.cpp">test_geir_avg_pool_update</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
