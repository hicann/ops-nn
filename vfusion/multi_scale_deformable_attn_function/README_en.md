# MultiScaleDeformableAttnFunction

English | [简体中文](./README.md)

## Applicable Products

|Product      | Support |
|:----------------------------|:-----------:|
|<term>Ascend 950PR&950DT products</term>|      √     |
|<term>Atlas A3 products</term>|      √     |
|<term>Atlas A2 products</term>|      √     |
|<term>Atlas 200I/500 A2 inference products</term>|      ×     |
|<term>Atlas inference products</term>|      √     |
|<term>Atlas training products</term>|      ×     |

## Function

- Operator function:

  It traverses different sampling points of feature maps of different sizes based on parameters such as the sampling location, attention weights, mapped value features, start index positions of multi-scale features, and spatial sizes of multi-scale feature maps (to convert sampling locations from normalized values into absolute positions).

- Formula:

    Let $b \in [0, bs)$, $q \in [0, num\_queries)$, $h \in [0, num\_heads)$, $\ell \in [0, num\_levels)$, $p \in [0, num\_points)$, and $c \in [0, channels)$ be the indexes of batch, query, head, feature map, sampling point, and channel, respectively. The height and width of the feature map of layer $\ell$ are $H_\ell$ and $W_\ell$, that is, $\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$; the $num\_keys$ dimension of $\mathrm{value}$ concatenates the feature map pixels of all layers flattened by layer ($num\_keys = \sum_{\ell=0}^{num\_levels-1} H_\ell W_\ell$), and the start index of layer $\ell$ is $\mathrm{levelStartIndex}[\ell]$.

    1. Map the normalized coordinates of the sampling point $\mathrm{location}[b, q, h, \ell, p] = (u, v) \in [0,1]^2$ ($u$ and $v$ are the 0th and 1st elements of the last dimension, corresponding to the $x$ and $y$ directions) to the pixel coordinate system of the feature map of layer $\ell$:

       $$
       x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
       $$

    2. Determine the four integer grid points between which the sampling point falls:

       $$
       x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
       y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
       $$

    3. Compute the offsets of the sampling point relative to the top-left grid point, which are used for interpolation weights:

       $$
       \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
       $$

    4. Compute the bilinear interpolation weights, and the sum of the four neighboring points is 1:

       $$
       \begin{aligned}
       w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
       w_{10} &= (1-\alpha_y)\alpha_x, \\
       w_{01} &= \alpha_y(1-\alpha_x), \\
       w_{11} &= \alpha_y\alpha_x
       \end{aligned}
       $$

    5. After flattening, the index of pixel $(y, x)$ on the feature map of layer $\ell$ in the $num\_keys$ dimension of $\mathrm{value}$ is:

       $$
       k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x
       $$

    6. Perform bilinear sampling on $\mathrm{value}$ to obtain the feature vector (length $channels$) corresponding to the sampling point:

       $$
       \begin{aligned}
       \operatorname{bilinear}(\mathrm{value};\,b,h,\ell,x,y) ={}&
       w_{00} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_0),\; h,\; :] \\
       &+ w_{10} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_1),\; h,\; :] \\
       &+ w_{01} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_0),\; h,\; :] \\
       &+ w_{11} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_1),\; h,\; :]
       \end{aligned}
       $$

    7. The bilinear sampling results of all layers and all sampling points are weighted and summed with $\mathrm{attnWeight}$ to obtain the final output $\mathrm{output}$ (shape $(bs, num\_queries, num\_heads \times channels)$; the last dimension is arranged by $h \times channels + c$):

       $$
       \mathrm{output}[b,\; q,\; h \times channels + c] =
       \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
       \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
       \operatorname{bilinear}\!\left(\mathrm{value};\, b, h, \ell,\;
       x_{b,q,h,\ell,p},\, y_{b,q,h,\ell,p}\right)[c]
       $$

       Here, $x_{b,q,h,\ell,p}$ and $y_{b,q,h,\ell,p}$ are the pixel coordinates of the sampling point $\mathrm{location}[b, q, h, \ell, p]$ obtained through the preceding coordinate mapping.

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
      <td>Feature values of feature maps, with shape (bs, num_keys, num_heads, channels). It corresponds to `value` in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_spatial_shapes</td>
      <td>Input</td>
      <td>Stores the height and width of the feature map of each scale, with shape (num_levels, 2). It corresponds to `spatialShape` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>value_level_start_index</td>
      <td>Input</td>
      <td>Start index of each feature map in the num_keys dimension of value, with shape (num_levels). It corresponds to `levelStartIndex` in the formula.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>sampling_locations</td>
      <td>Input</td>
      <td>Sampling point position tensor, with shape (bs, num_queries, num_heads, num_levels, num_points, 2). It corresponds to `location` in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>attention_weights</td>
      <td>Input</td>
      <td>Sampling point weight tensor, with shape (bs, num_queries, num_heads, num_levels, num_points). It corresponds to `attnWeight` in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>output</td>
      <td>Output</td>
      <td>Computation output of the operator, with shape (bs, num_queries, num_heads × channels). It corresponds to `output` in the formula.</td>
      <td>FLOAT, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
  </tbody></table>

- <term>Atlas inference products</term>: BFLOAT16 is not supported

## Constraints

- <term>Atlas inference products</term>:
  - Number of channels: channels%32 = 0, and channels <= 256
  - Number of queries: 32 <= num_queries < 500000
  - Number of feature maps: num_levels <= 16
  - Number of heads: num_heads = [2, 4, 8]
  - Number of sampling points: num_points = [4, 8]
- <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, <term>Ascend 950PR&950DT products</term>:
  - Number of channels: channels%8 = 0, and channels <= 256
  - Number of queries: 32 <= num_queries < 500000
  - Number of feature maps: num_levels <= 16
  - Number of heads: num_heads <= 16
  - Number of sampling points: num_points <= 16

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_multi_scale_deformable_attn_function](./examples/test_aclnn_multi_scale_deformable_attn_function.cpp) | Calls the aclnnMultiScaleDeformableAttnFunction operator through the [aclnnMultiScaleDeformableAttnFunction](./docs/aclnnMultiScaleDeformableAttnFunction_en.md) interface.    |
