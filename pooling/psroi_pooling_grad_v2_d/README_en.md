# PSROIPoolingGradV2D

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |     √    |
| <term>Atlas A2 products</term>    |     √    |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

Ascend950 uses the new arch35 implementation added in this directory; Atlas A2/A3 training and inference products use the existing TBE implementation of CANN. The two implementations differ in data formats and constraints, so the Ascend950 operator package cannot be directly used for A2/A3.

| Implementation | x/y data types and storage formats | rois data type and storage format | ROI count constraint |
| :-- | :-- | :-- | :-- |
| Ascend950 (arch35 in this directory) | FLOAT32, ND or NCHW; the formats of x/y must be the same | FLOAT32, ND | R is a non-negative integer, and R=0 is supported |
| Atlas A2/A3 (existing TBE of CANN) | FLOAT32, NC1HWC0, with the original format NCHW | FLOAT32, ND | R is a positive integer and a multiple of 16 |

## Function

PSROIPoolingGradV2D scatters the upstream gradient of PS ROI Pooling back to the input feature map space according to the position-sensitive channel mapping and the bin area. The logical shape of the input `x` is `[B*R, O, G, G]`, the shape of `rois` is `[B, 5, R]`, and the logical shape of the output `y` is `[B, O*G*G, H, W]`, where `O=output_dim`, `G=group_size`, and `[H,W]=input_size`.

In the Ascend950 implementation, overlapping contributions to the same output element are accumulated in a fixed batch/ROI order. The ROI bin boundaries use separate FP32 multiplication and addition, and FMA contraction is prohibited, to ensure the discrete coverage regions and result determinism. This determinism statement does not extend to the existing TBE implementation.

## Operator Prototype

For the public graph-mode prototype, see [psroi_pooling_grad_v2_d_proto.h](./op_graph/psroi_pooling_grad_v2_d_proto.h), and the registration name is `PSROIPoolingGradV2D`. The input order is `x` and `rois`, and the output is `y`; `spatial_scale`, `output_dim`, `group_size`, and `input_size` are all required attributes.

For compatibility with existing APIs, the public `REG_OP` keeps the FLOAT16/FLOAT32 declarations. This does not mean that the implementation of a specific product supports FLOAT16. Both implementations support only FLOAT32, and the data types of `x`, `rois`, and `y` must be the same.

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
| :-- | :--: | :-- | :--: | :--: |
| x | Input | Upstream gradient, with the logical shape `[B*R, O, G, G]`. | FLOAT32 | ND or NCHW for 950; NC1HWC0 for A2/A3 |
| rois | Input | The shape is `[B, 5, R]`. The five planes store `batch_id, x1, y1, x2, y2` in order, that is, the r-th ROI is `rois[b, :, r]`. | FLOAT32 | ND |
| spatial_scale | Attribute | Scaling ratio from ROI coordinates to feature map coordinates. It must be finite and greater than 0. | FLOAT | - |
| output_dim | Attribute | Number of pooling output channels O. It must be greater than 0. | INT | - |
| group_size | Attribute | Position-sensitive group size G. The value range is `[1, 127]`. | INT | - |
| input_size | Attribute | Output spatial size `[H, W]`. It must have exactly two elements, and both of them must be greater than 0. | LIST_INT | - |
| y | Output | Gradient of the input feature map, with the logical shape `[B, O*G*G, H, W]`. | FLOAT32 | ND or NCHW for 950 and it must be consistent with x; NC1HWC0 for A2/A3 |

## Constraints

- The four ROI coordinates must be finite and greater than or equal to 0; `batch_id` must be an integer in `[0, B)`. Callers must ensure these tensor value constraints, and shape derivation does not validate tensor contents.
- Ascend950: `x` and `y` must both be 4D ND or both be 4D NCHW, and they cannot be mixed. The two formats use the same contiguous 4D layout. `rois` must be 3D ND and `rois.shape[1] == 5`; `x.shape == [B*R, output_dim, group_size, group_size]`.
- Graph-mode construction for Ascend950 uses `x/y=NCHW, rois=ND`, and the ND computation entry is still supported.
- Ascend950: when `R == 0`, an all-zero output with the correct shape is returned; multiple runs with the same valid input produce bit-exact results.
- Atlas A2/A3: the logical format of `x` and `y` is NCHW, and the physical storage format is NC1HWC0. The physical shapes are `[B*R, ceil(O/16), G, G, 16]` and `[B, ceil(O*G*G/16), H, W, 16]` respectively. `rois` is an ND tensor of `[B, 5, R]`, and R must be a positive multiple of 16.
- The A2/A3 support comes from the existing implementation of CANN, and the CANN operator package of the corresponding product needs to be installed. The Ascend950 build and samples in this directory do not provide A2/A3 kernels and do not declare bit-exact consistency between the two implementations.

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
    <td><a href="./examples/test_geir_psroi_pooling_grad_v2_d.cpp">test_geir_psroi_pooling_grad_v2_d</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">Operator Invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
