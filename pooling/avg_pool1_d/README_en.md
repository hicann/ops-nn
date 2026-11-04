# AvgPool1d

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :--- | :---: |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | × |
| <term>Atlas A2 products</term> | × |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Perform 1D average pooling on the last dimension of the input tensor. The operator supports asymmetric padding, output length correction with ceil_mode, and denominator selection with count_include_pad.

- Formula:

Assume that the shape of input x is (..., L), pads=[pad_left, pad_right], and the padded length L_padded = L + pad_left + pad_right. The output length L_out is calculated by the following formula:

$$
L_{out} = \begin{cases} \left\lfloor \frac{L_{padded} - ksize}{strides} \right\rfloor + 1, & ceil\_mode = False \\ \left\lceil \frac{L_{padded} - ksize}{strides} \right\rceil + 1, & ceil\_mode = True \end{cases}
$$

When pad_left > 0 and the start position of the last window falls in the padding area (that is, (L_out - 1) * strides >= L + pad_left), L_out decreases by 1. If the L_out calculated by the formula is <= 0, L_out is processed as 0 and an empty tensor is output (no error is reported).

The value of each output window is:

$$
y[..., w] = \frac{\sum_{i=0}^{ksize-1} x_{padded}[..., w \times strides + i]}{divisor}
$$

When count_include_pad=True, divisor is the actual number of elements of the window in the padded space (which may be less than ksize in ceil_mode). When count_include_pad=False, divisor is the size of the intersection between the window and the original data area (minimum value: 1).

## Parameters

<table><thead>
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
    <td>Input parameter for 1D average pooling. The shape is (..., L), where L is the length of the pooling dimension.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>1D average pooling result. The shape is (..., L_out), and the dtype is the same as that of x.</td>
    <td>float16, float32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>ksize</td>
    <td>Attribute</td>
    <td>Pooling window size.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>strides</td>
    <td>Attribute</td>
    <td>Sliding stride of the pooling window.</td>
    <td>int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pads</td>
    <td>Attribute</td>
    <td>Left and right padding sizes. The format is [pad_left, pad_right]. Asymmetric padding is supported.</td>
    <td>ListInt</td>
    <td>-</td>
  </tr>
  <tr>
    <td>ceil_mode</td>
    <td>Attribute</td>
    <td>When this parameter is True, the output length is calculated in ceil mode. When this parameter is False, the output length is calculated in floor mode. Default value: False.</td>
    <td>bool</td>
    <td>-</td>
  </tr>
  <tr>
    <td>count_include_pad</td>
    <td>Attribute</td>
    <td>When this parameter is True, the denominator includes padding elements. When this parameter is False, the denominator includes only valid elements. Default value: False.</td>
    <td>bool</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The rank of input x must be greater than or equal to 3. 3D to 8D tensors are supported.
- ksize must be greater than or equal to 1.
- strides must be greater than or equal to 1 (stride=0 causes an error).
- Each element of pads must be greater than or equal to 0, and the length of pads is 2.
- Input x supports only the float16 and float32 types.
- Only the ND data format is supported.
- Output y has the same dtype as input x.
- Empty tensor inputs are supported (an empty tensor is output when L=0).
- If the L_out calculated by the formula is <= 0 (for example, ksize is greater than L_padded), L_out is processed as 0 and an empty tensor is output, and the operator does not report an error. PyTorch reports an error (Output size is too small) for the same configuration. This is an intentional deviation.
- When pad_left > 0 and the start position of the last window falls in the padding area (that is, (L_out - 1) * strides >= L + pad_left), L_out decreases by 1. This correction is not performed when pad_left is 0 (an intentional difference from the ceil_mode behavior of PyTorch: keeping the last window avoids the NaN problem of PyTorch when pad_l=0).
- When pad_left is 0 and the last window is completely outside the original data area, the window is kept in the output and computed as a zero-padded window (no valid elements, and the result is 0).
- When count_include_pad=False, divisor is the size of the intersection between the window and the original data area, with a minimum value of 1. A division-by-zero error cannot occur.
- Input x and output y must be contiguous tensors. The operator declares AutoContiguous, and non-contiguous inputs are automatically converted to contiguous tensors by the framework.
- Deterministic implementation: sequential accumulation within a single thread is used, and atomic operations are not used. The same input produces the same result across multiple executions.
- NaN inputs propagate according to IEEE 754 rules, and Inf inputs propagate according to IEEE 754 rules.

## Invocation Description

<table><thead>
  <tr>
    <th>Invocation method</th>
    <th>Sample code</th>
    <th>Description</th>
  </tr></thead>
<tbody>
  <tr>
    <td>GE graph mode</td>
    <td><a href="./examples/test_geir_avg_pool1_d.cpp">test_geir_avg_pool1_d</a></td>
    <td>Invoke the AvgPool1D operator through graph construction using the operator IR defined in <a href="./op_graph/avg_pool1_d_proto.h">avg_pool1_d_proto.h</a>. For details, see <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a>.</td>
  </tr>
</tbody>
</table>
