# MaxPool3DGradGrad

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

- Operator function: MaxPool3DGradGrad computes the gradient of the upstream gradient with respect to one backward propagation of 3D max pooling. The operator scans the valid input positions in each pooling window in the D→H→W order. When a match exists, `grads` is read from the first position where `orig_x` equals the target value of `orig_y`; otherwise, the FP16 zero is output.

### Formula

$$
p^*(q)=\operatorname{first}\{p\in W_{valid}(q)\mid orig\_x(p)=orig\_y(q)\}
$$

$$
y(q)=
\begin{cases}
grads(p^*(q)), & p^*(q)\text{ exists} \\
0_{FP16}, & p^*(q)\text{ does not exist}
\end{cases}
$$

where $q=(n,od,oh,ow,c)$, and $W_{valid}$ contains only the valid input positions that fall in the range of `orig_x`. SAME padding positions do not participate in the equality comparison.

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
      <td>orig_x</td>
      <td>Input</td>
      <td>Input of forward MaxPool3D, with the shape `(N,D,H,W,C)`.</td>
      <td>FLOAT16</td>
      <td>ND (logical NDHWC)</td>
    </tr>
    <tr>
      <td>orig_y</td>
      <td>Input</td>
      <td>Forward output derived from orig_x and the pooling attributes, with the shape `(N,Do,Ho,Wo,C)`.</td>
      <td>FLOAT16</td>
      <td>ND (logical NDHWC)</td>
    </tr>
    <tr>
      <td>grads</td>
      <td>Input</td>
      <td>Second-order upstream input, with the shape exactly the same as orig_x.</td>
      <td>FLOAT16</td>
      <td>ND (logical NDHWC)</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Gradient result of each pooling window, with the shape exactly the same as orig_y.</td>
      <td>FLOAT16</td>
      <td>ND (logical NDHWC)</td>
    </tr>
    <tr>
      <td>ksize</td>
      <td>Attribute</td>
      <td>Required attribute, indicating the pooling window sizes in the D/H/W directions.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>strides</td>
      <td>Attribute</td>
      <td>Required attribute, indicating the pooling strides in the D/H/W directions.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>pads</td>
      <td>Attribute</td>
      <td>Required attribute, a non-negative padding configuration of length 6. All 0s indicate VALID, and any non-0 value indicates SAME.</td>
      <td>LIST_INT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>data_format</td>
      <td>Attribute</td>
      <td>Optional attribute, the logical data format, with the default value NDHWC.</td>
      <td>STRING</td>
      <td>-</td>
    </tr>
  </tbody></table>

## Constraints

- `orig_x`, `orig_y`, `grads`, and `y` support only FLOAT16, the ND physical format, and the logical NDHWC format.
- All inputs and outputs are non-empty rank=5 tensors, and each dimension must be greater than 0. Scalars, 1D tensors, empty tensors, and 8D tensors are not supported.
- The shape of `orig_x` must be exactly the same as that of `grads`; the shape of `orig_y` must be exactly the same as that of `y`.
- The N and C dimensions of `orig_x` and `orig_y` must be the same, and the D, H, and W dimensions of `orig_y` must be consistent with the pooling output derived from `ksize`, `strides`, and `pads`.
- The lengths of `ksize` and `strides` support only 1, 3, or 5, and the spatial values must be greater than 0. When the length is 5, the first and last elements must be 1, and pooling along the N and C axes is prohibited.
- The length of `pads` must be 6 and the elements must be non-negative. All 0s are processed as VALID, and any non-0 value is processed as SAME.
- `data_format` supports only NDHWC.
- When multiple `orig_x` values equal the target, the first position in the D→H→W scan order is always selected. When there is no match or the target is NaN, the FP16 zero is output. The K1 path also compares `orig_x` with `orig_y` first, and outputs zero when they do not match.
- SAME boundaries scan only valid input positions. Padding does not construct visible sentinels and does not participate in the `orig_x == orig_y` matching.

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
    <td><a href="./examples/test_geir_max_pool_3d_grad_grad.cpp">test_geir_max_pool_3d_grad_grad</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
