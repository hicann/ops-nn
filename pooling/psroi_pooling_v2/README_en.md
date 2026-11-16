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

The operator first applies round-to-nearest-even (ties to even, equivalent to `rint` in the default rounding mode) to the RoI coordinates, then adds 1 to the bottom-right endpoint and multiplies by `spatial_scale`, clips to the feature map range, and computes the average of the corresponding grouped region. Coordinate and boundary calculations use float32 intermediate values.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :----- | :------------ | :--- | :------- | :------- |
| `x` | Input | Input feature map, with the shape `[N,C,H,W]`. | `float16`, `float32` | NCHW (graph input), internally converted to contiguous ND |
| `rois` | Input | RoI data, with the shape `[N,5,R]`. The five planes are `batch_id,x1,y1,x2,y2` in order. | Same as `x` | ND |
| `y` | Output | Pooling result, with the shape `[N*R,output_dim,group_size,group_size]`. | Same as `x` | NCHW (graph output), internally contiguous ND |
| `spatial_scale` | Attribute | Scaling ratio from the RoI coordinates of the original image to the feature map coordinates. Mandatory. | Float | - |
| `output_dim` | Attribute | Number of output channels. Mandatory. | Int | - |
| `group_size` | Attribute | Position-sensitive group size. The output space is `group_size*group_size`. Mandatory. | Int | - |

## Constraints

- In the graph interface, `x` and `y` use the NCHW logical format, and the framework converts them to the contiguous ND format used by the A5 kernel through `AutoContiguous`; `rois` uses ND. `x` must be a 4D tensor, `rois` must be a 3D tensor, and `rois.shape[1]` must be 5.
- `x.shape[0]` must equal `rois.shape[0]`, and the number of input channels must satisfy `C=output_dim*group_size²`.
- `x`, `rois`, and `y` support only the two combinations of all `float16` or all `float32`; a mixed dtype is not supported.
- `spatial_scale` must be a finite positive number, `output_dim` must be greater than 0, and the value range of `group_size` is `[1,127]`.
- Valid `rois` coordinates must be finite and non-negative. `batch_id` and the four coordinates use round-to-nearest-even, and the rounded `batch_id` must be in `[0,N-1]`. If any RoI does not satisfy these value constraints, all outputs corresponding to only that RoI are set to 0, other RoIs continue to be processed, and no out-of-bounds access occurs.
- Empty outputs with `N=0` or `R=0` are supported; the known `C`, `H`, and `W` dimensions must be greater than 0.
- Fixed-rank dynamic shapes are supported. During graph construction, `x` and `rois` may use unknown dimensions `-1`; before execution, they must be concretized as 4D and 3D tensors, respectively.
- For fixed-rank dynamic shapes, known dimensions are immediately checked against static constraints. After unknown dimensions are concretized, they must satisfy `C=output_dim*group_size²`, `H/W>0`, `rois.shape[1]=5`, and `x.shape[0]=rois.shape[0]`.
- The output is always 4D. Only when both `rois.shape[0]` and `rois.shape[2]` are known is the leading output dimension concretized as their product; otherwise, it remains `-1`. The other three dimensions are always `[output_dim,group_size,group_size]`.
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
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification. See the <a href="./op_graph/psroi_pooling_v2_proto.h">PSROIPoolingV2 IR prototype</a> for the interface definition.</td>
  </tr>
</tbody>
</table>
