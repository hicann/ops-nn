# PSROIPoolingV2

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

PSROIPoolingV2 performs position-sensitive RoI average pooling on the input feature map. For each RoI, the operator maps each output channel and pooling position to a unique input channel, completing region aggregation while preserving the spatial layout of the RoI. It is suitable for position-sensitive object detection networks such as R-FCN.

The logical shape of input `x` is `[N,C,H,W]`, and the shape of input `rois` is `[N,5,R]`. Let `O=output_dim` and `G=group_size`. It is required that `C=O*G*G`, and the shape of output `y` is `[N*R,O,G,G]`. The input channel corresponding to each output position is:

$$
c_{\mathrm{in}} = o \times G^2 + p_h \times G + p_w
$$

The operator first applies `rint` to the RoI coordinates, then adds 1 to the bottom-right endpoint and multiplies by `spatial_scale`, clips to the feature map range, and computes the average of the corresponding grouped region.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :----- | :------------ | :--- | :------- | :------- |
| `x` | Input | Input feature map, with the shape `[N,C,H,W]`. | `float16`, `float32` | ND |
| `rois` | Input | RoI data, with the shape `[N,5,R]`. The five planes are `batch_id,x1,y1,x2,y2` in order. | Same as `x` | ND |
| `y` | Output | Pooling result, with the shape `[N*R,output_dim,group_size,group_size]`. | Same as `x` | ND |
| `spatial_scale` | Attribute | Scaling ratio from the RoI coordinates of the original image to the feature map coordinates. Mandatory. | Float | - |
| `output_dim` | Attribute | Number of output channels. Mandatory. | Int | - |
| `group_size` | Attribute | Position-sensitive group size. The output space is `group_size*group_size`. Mandatory. | Int | - |

## Constraints

- `x` must be a 4D ND tensor, `rois` must be a 3D ND tensor, and `rois.shape[1]` must be 5.
- `x.shape[0]` must equal `rois.shape[0]`, and the number of input channels must satisfy `C=output_dim*group_size²`.
- `x`, `rois`, and `y` support only the two combinations of all `float16` or all `float32`; a mixed dtype is not supported.
- `spatial_scale` must be a finite positive number, `output_dim` must be greater than 0, and the value range of `group_size` is `[1,127]`.
- The coordinates in `rois` must be finite and non-negative, and `round(batch_id)` must be in `[0,N-1]`.
- Empty outputs with `N=0` or `R=0` are supported; the known `C`, `H`, and `W` dimensions must be greater than 0.
- The coordinate processing order is fixed as "rounding → add 1 to the bottom-right endpoint → scaling", and the order of rounding and scaling must not be exchanged.
- The forward computation is a deterministic implementation; IEEE NaN/Inf values in `x` propagate naturally according to floating-point semantics.

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
    <td><a href="./examples/test_geir_psroi_pooling_v2.cpp">test_geir_psroi_pooling_v2</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
