# MultiScaleDeformableAttentionGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>  |     √    |
|  <term>Atlas A2 products</term>     |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>    |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function:

  The forward MultiScaleDeformableAttention operator traverses different sampling points on feature maps of different sizes through parameters such as sampling locations, attention weights, mapped value features, the start index positions of multi-scale features, and the spatial sizes of multi-scale feature maps (so that sampling positions are converted from normalized values to absolute positions). The backward operator computes the gradients of the inputs based on the contribution of the forward inputs to the output and the initial gradient.

- Layout convention (kernel perspective):

  The kernel of this operator is implemented based on the **transposed layout**. The query dimension (num_queries) of sampling_locations and attention_weights is moved to the last dimension, that is, the inputs received by the kernel are:

  - `sampling_locations = (bs, num_heads, num_levels, num_points, 2, num_queries)`, where the second-to-last dimension is the coordinate component of the sampling points (always 2, where the 0th and 1st elements correspond to the x and y directions respectively), and the last dimension is the query dimension;
  - `attention_weights = (bs, num_heads, num_levels, num_points, num_queries)`.

  The public aclnn API accepts the public layouts `(bs, num_queries, num_heads, num_levels, num_points, 2)` / `(bs, num_queries, num_heads, num_levels, num_points)`. The aclnn adaptation layer performs the preceding transpose and then dispatches the kernel. In graph mode (GE), callers need to connect this operator node directly in the transposed layout.

- Formula:

    Let $b \in [0, bs)$, $q \in [0, num\_queries)$, $h \in [0, num\_heads)$, $\ell \in [0, num\_levels)$, $p \in [0, num\_points)$, and $c \in [0, channels)$ be the indexes of batch, query, head, feature map, sampling point, and channel respectively. The height and width of the $\ell$-th feature map are $H_\ell$ and $W_\ell$, that is, $\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$; the flattened index of pixel $(y, x)$ of the $\ell$-th feature map on the $num\_keys$ dimension of $\mathrm{value}$ is $k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x$. The sampling point $\mathrm{location}[b, q, h, \ell, p] = (u, v)$ (where the 0th and 1st elements of the last dimension correspond to the $x$ and $y$ directions respectively) is mapped to the pixel coordinates $x = u \cdot W_\ell - 0.5$ and $y = v \cdot H_\ell - 0.5$; let $x_0 = \lfloor x \rfloor$, $x_1 = x_0 + 1$, $y_0 = \lfloor y \rfloor$, $y_1 = y_0 + 1$, $\alpha_x = x - x_0$, $\alpha_y = y - y_0$, the bilinear interpolation weights be $w_{00} = (1-\alpha_y)(1-\alpha_x)$, $w_{10} = (1-\alpha_y)\alpha_x$, $w_{01} = \alpha_y(1-\alpha_x)$, $w_{11} = \alpha_y\alpha_x$, and the feature vectors of the four neighboring points be $V_{y_i, x_j} := \mathrm{value}[b,\; k_\ell(y_i, x_j),\; h,\; :]$ ($i, j \in \{0, 1\}$; out-of-bounds neighboring points are treated as 0).

    - The backward operator computes the gradients of the three inputs $\mathrm{value}$, $\mathrm{location}$, and $\mathrm{attnWeight}$ based on the gradient $\mathrm{gradOutput}$. Let the slice of head $h$ in $\mathrm{gradOutput}$ be $G_{b,q,h,c} := \mathrm{gradOutput}[b,\; q,\; h \times channels + c]$ (the layout of gradOutput is $(bs, num\_queries, num\_heads \times channels)$, and the kernel does not transpose this input).
      1. Expand by the attention weights:

         $$
         \tilde{G}_{b,q,h,\ell,p,c} = \mathrm{attnWeight}[b, q, h, \ell, p] \cdot G_{b,q,h,c}
         $$

      2. Compute the gradient $\mathrm{gradValue}$ of $\mathrm{value}$ (the shape is the same as value): for each sampling point $(b, q, h, \ell, p)$, accumulate its expanded gradient to the four neighboring point positions with the bilinear weights (accumulate over all $q$ and $p$; out-of-bounds neighboring points are not accumulated):

         $$
         \begin{aligned}
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_0),\; h,\; c] &+= w_{00} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_1),\; h,\; c] &+= w_{10} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_0),\; h,\; c] &+= w_{01} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_1),\; h,\; c] &+= w_{11} \cdot \tilde{G}_{b,q,h,\ell,p,c}
         \end{aligned}
         $$

      3. Compute the gradient $\mathrm{gradAttnWeight}$ of $\mathrm{attnWeight}$, which is the inner product of the output gradient and the sampled feature vector:

         $$
         \mathrm{gradAttnWeight}[b, q, h, \ell, p] =
         \sum_{c} G_{b,q,h,c} \cdot
         \left( \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c] \right)
         $$

      4. Compute the gradient $\mathrm{gradLocation}$ of $\mathrm{location}$. The gradient of the sampling point pixel coordinates $(x, y)$ is obtained from the first-order partial derivatives of the bilinear interpolation:

         $$
         \nabla x_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_0, x_1}[c] - V_{y_0, x_0}[c])(1-\alpha_y)
             + (V_{y_1, x_1}[c] - V_{y_1, x_0}[c])\alpha_y \big]
         $$

         $$
         \nabla y_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_1, x_0}[c] - V_{y_0, x_0}[c])(1-\alpha_x)
             + (V_{y_1, x_1}[c] - V_{y_0, x_1}[c])\alpha_x \big]
         $$

      5. From $x = u \cdot W_\ell - 0.5$ and $y = v \cdot H_\ell - 0.5$, scale the gradients of the regressed normalized coordinates according to the chain rule:

         $$
         \mathrm{gradLocation}[b, q, h, \ell, p] = (\nabla u, \nabla v), \qquad
         \nabla u = W_\ell \cdot \nabla x_{b,q,h,\ell,p}, \qquad
         \nabla v = H_\ell \cdot \nabla y_{b,q,h,\ell,p}
         $$

    - The preceding formulas are written in the logical layout (the query dimension is the 1st dimension); the physical layout of the four tensors `sampling_locations`, `attention_weights`, `grad_sampling_locations`, and `grad_attention_weights` sent and received by the kernel is the transposed layout (see "Layout convention"). The semantics are the same, differing only by one `transpose(0, 2, 3, 4, 5, 1)` in the dimension order (the attnWeight side is `transpose(0, 2, 3, 4, 1)`).

## Parameters

<table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 181px">
  <col style="width: 144px">
  <col style="width: 273px">
  <col style="width: 256px">
  <col style="width: 116px">
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
      <td>value</td>
      <td>Input</td>
      <td>Feature values of the feature maps, with the shape (bs, num_keys, num_heads, channels). Corresponds to `value` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_spatial_shapes</td>
      <td>Input</td>
      <td>Stores the height and width of the feature map of each scale, with the shape (num_levels, 2). Corresponds to `spatialShape` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_level_start_index</td>
      <td>Input</td>
      <td>Start index of each feature map on the num_keys dimension of value, with the shape (num_levels). Corresponds to `levelStartIndex` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampling_locations</td>
      <td>Input</td>
      <td>Sampling point position tensor, with the shape (bs, num_heads, num_levels, num_points, 2, num_queries). The second-to-last dimension is the coordinate component (always 2). Corresponds to `location` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attention_weights</td>
      <td>Input</td>
      <td>Sampling point weight tensor, with the shape (bs, num_heads, num_levels, num_points, num_queries). Corresponds to `attnWeight` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_output</td>
      <td>Input</td>
      <td>Upstream gradient of the forward output output (the initial gradient of the backward operator), with the shape (bs, num_queries, num_heads × channels). Corresponds to `gradOutput` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_value</td>
      <td>Output</td>
      <td>Gradient of the input value, with the shape the same as value. Corresponds to `gradValue` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_sampling_locations</td>
      <td>Output</td>
      <td>Gradient of the input sampling_locations, with the shape the same as sampling_locations, that is, (bs, num_heads, num_levels, num_points, 2, num_queries). Corresponds to `gradLocation` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>grad_attention_weights</td>
      <td>Output</td>
      <td>Gradient of the input attention_weights, with the shape the same as attention_weights, that is, (bs, num_heads, num_levels, num_points, num_queries). Corresponds to `gradAttnWeight` in the formula.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The number of channels satisfies channels % 8 = 0 and channels <= 256
- The number of queries satisfies num_queries < 500000
- The number of feature maps satisfies num_levels <= 16
- The number of heads satisfies num_heads <= 16
- The number of sampling points satisfies num_points <= 16

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_multi_scale_deformable_attention_grad](./examples/test_aclnn_multi_scale_deformable_attention_grad.cpp) | Calls the aclnnMultiScaleDeformableAttentionGrad operator through the [aclnnMultiScaleDeformableAttentionGrad](./docs/aclnnMultiScaleDeformableAttentionGrad_en.md) API.    |
