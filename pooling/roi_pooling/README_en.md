# ROIPooling

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

- Operator function: Performs max pooling on the input feature map by ROI (Region of Interest) regions and outputs pooling results of a fixed size, used for tasks such as object detection.

- Formula:

For each ROI n (rois[n] = [batch_idx, x1, y1, x2, y2]), channel c, and pooling position (ph,pw):

**1. ROI coordinate mapping** (uses separate h/w scaling factors, roundf rounds to int, and the end coordinates have no +1 offset):

$$
\text{roi\_start}_w = \text{round}(x1 \cdot \text{spatial\_scale\_w}), \quad \text{roi\_start}_h = \text{round}(y1 \cdot \text{spatial\_scale\_h})
$$

$$
\text{roi\_end}_w = \text{round}(x2 \cdot \text{spatial\_scale\_w}), \quad \text{roi\_end}_h = \text{round}(y2 \cdot \text{spatial\_scale\_h})
$$

> All the preceding coordinates are **integers** (int type); the +1 offset is on the ROI size (see below), not on the coordinates.

**2. ROI size** (int arithmetic, +1 on width/height, malformed ROIs are forced to be non-empty):

$$
\text{roi\_w} = \max(\text{roi\_end}_w - \text{roi\_start}_w + 1, 1), \quad \text{roi\_h} = \max(\text{roi\_end}_h - \text{roi\_start}_h + 1, 1)
$$

**3. Bin splitting and boundary clipping** (bin_size is float, based on the int roi_w/roi_h):

$$
\text{bin\_size}_w = \frac{\text{roi\_w}}{\text{pooled\_w}}, \quad \text{bin\_size}_h = \frac{\text{roi\_h}}{\text{pooled\_h}}
$$

$$
\text{bin}_{x1} = \text{clamp}(\lfloor pw \cdot \text{bin\_size}_w \rfloor + \text{roi\_start}_w, 0, W)
$$

$$
\text{bin}_{x2} = \text{clamp}(\lceil (pw+1) \cdot \text{bin\_size}_w \rceil + \text{roi\_start}_w, 0, W)
$$

(The same applies to the y direction, clamped to [0, H])

> floor/ceil is based on the relative offset (excluding roi_start), and then the integer roi_start is added.

**4. Max pooling** (empty bins output 0):

$$
y[n, c, ph, pw] = \begin{cases} 0 & \text{if } \text{bin}_{x2} \le \text{bin}_{x1} \text{ or } \text{bin}_{y2} \le \text{bin}_{y1} \\ \max_{h, w \in [\text{bin}_{y1}, \text{bin}_{y2}) \times [\text{bin}_{x1}, \text{bin}_{x2})} x[\text{batch\_idx}, c, h, w] & \text{otherwise} \end{cases}
$$

The +1 offset is on the ROI size (`roi_width = roi_end - roi_start + 1`), which is the common standard of ROI Pooling (originating from Fast R-CNN)

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
    <td>Feature map, with shape [N,C,H,W]; it is x in the formulas.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>rois</td>
    <td>Input</td>
    <td>ROI boxes, with shape [K,5]; each row is [batch_idx,x1,y1,x2,y2].</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>roi_actual_num</td>
    <td>Input</td>
    <td>Actual number of ROIs of each batch; optional input that does not participate in the computation in the current version.</td>
    <td>INT32</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>y</td>
    <td>Output</td>
    <td>Pooling result, with shape [K,C,pooled_h,pooled_w]; it is y in the formulas.</td>
    <td>FLOAT, FLOAT16</td>
    <td>ND</td>
  </tr>
  <tr>
    <td>pooled_h</td>
    <td>Attribute</td>
    <td>Height of the pooling output.</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>pooled_w</td>
    <td>Attribute</td>
    <td>Width of the pooling output.</td>
    <td>Int</td>
    <td>-</td>
  </tr>
  <tr>
    <td>spatial_scale_h</td>
    <td>Attribute</td>
    <td>Scaling factor in the height direction.</td>
    <td>Float</td>
    <td>-</td>
  </tr>
  <tr>
    <td>spatial_scale_w</td>
    <td>Attribute</td>
    <td>Scaling factor in the width direction.</td>
    <td>Float</td>
    <td>-</td>
  </tr>
</tbody></table>

## Constraints

- The input x supports only 4 dimensions [N,C,H,W].
- The input rois supports only 2 dimensions [K,5]; each row is [batch_idx,x1,y1,x2,y2].
- roi_actual_num is an optional input and does not participate in the computation in the current version.
- pooled_h and pooled_w must be greater than 0.
- spatial_scale_h and spatial_scale_w must be greater than 0.
- The dtypes of x and rois must be consistent.
- The shape of the output y is [K,C,pooled_h,pooled_w].
- ROI coordinates are rounded to int with roundf, and the +1 offset is on the ROI size (roi_width = roi_end - roi_start + 1).
- Malformed ROIs are forced to be non-empty, and roi_size takes max(size,1) (int arithmetic).
- Empty bins output 0.

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
    <td><a href="./examples/test_geir_roi_pooling.cpp">test_geir_roi_pooling</a></td>
    <td>See <a href="../../docs/zh/invocation/quick_op_invocation.md">operator invocation</a> to complete operator compilation and verification.</td>
  </tr>
</tbody>
</table>
