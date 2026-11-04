# SwigluGroupQuantGrad

English | [简体中文](./README.md)

## Applicable Products

|Product             |  Support  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR&950DT products</term>   |     √    |
|  <term>Atlas A3 products</term>   |     ×    |
|  <term>Atlas A2 products</term>     |     ×    |
|  <term>Atlas 200I/500 A2 inference products</term>    |     ×    |
|  <term>Atlas inference products</term>     |     ×    |
|  <term>Atlas training products</term>    |     ×    |

## Function

- Operator function: The SwigluGroupQuantGrad operator implements the backward gradient computation of the grouped quantization of the SwiGLU activation function, and is used to compute the input gradient `grad_x` and the weight gradient `grad_weight`.
- Operator support scope: The operator supports the MoE scenario (pass in groupIndex) and the non-MoE scenario (pass empty groupIndex), supports the optional Clamp backpropagation mask, and supports the optional Weight gradient computation.
- Computation flow:
  - Step 0: GroupIndex processing (optional) → compute trunc
  - Step 1: Input splitting (split x into x0 and x1)
  - Step 2: Clamp processing (optional)
  - Step 3: SwiGLU backpropagation computation
  - Step 4: Weight gradient computation (optional)
  - Step 5: Gradient concatenation output

- GroupIndex processing formula for the MoE scenario:
  $$
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]
  $$
  where $G$ is the number of MoE expert groups, and all subsequent steps process only the first $\text{trunc}$ rows of data.

- Input splitting formula:
  $$
    \mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)
  $$

  $$
    \mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)
  $$

- Clamp processing formula (when clamp_limit > 0):
  $$
    \mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)
  $$

  $$
    \mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)
  $$
  where $c$ is `clamp_limit`.

- SiLU gradient formula:
  $$
    \frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)
  $$
  where $\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}$

- Input gradient computation formula:
  $$
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]
  $$

  $$
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])
  $$
  where $\mathbf{grad}_{y_0} = \mathbf{grad}_{y} \cdot \mathbf{weight}$ if weight is provided, and $\mathbf{grad}_{y_0} = \mathbf{grad}_{y}$ if weight is not provided.

- Weight gradient computation formula (optional):
   $$
     \mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{y}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]
   $$
   where $\mathbf{y}_{\text{origin}}$ is the original activation value output of the SwiGLU forward propagation, summed along the last dimension (H dimension).

   $$
     \mathbf{grad}_{\text{weight}}[t] = \mathbf{grad}_{\text{weight}}[t] \cdot \mathbb{I}(t < \text{trunc})
   $$

- Clamp backpropagation mask formula (when clamp_limit > 0):
  $$
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbb{I}(\mathbf{x}_0[t, h] < c)
  $$

  $$
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)
  $$
  where $\mathbb{I}$ is the indicator function.

- Gradient concatenation and GroupIndex processing formula:
  $$
    \mathbf{grad}_x[t, h] = \begin{cases}
    \mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\
    \mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)
    \end{cases}
  $$

  $$
    \mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})
  $$

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
      <td>gradY</td>
      <td>Input</td>
      <td>Gradient output tensor, that is, the gradient from the downstream layer.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Input tensor of the forward propagation.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>weightOptional</td>
      <td>Input</td>
      <td>MoE weight tensor.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>yOriginOptional</td>
      <td>Input</td>
      <td>Original activation value output of the SwiGLU forward propagation.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>groupIndexOptional</td>
      <td>Input</td>
      <td>GroupIndex tensor for dynamic core allocation.</td>
      <td>INT64</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>clampLimit</td>
      <td>Attribute</td>
      <td>Clamp threshold.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut</td>
      <td>Output</td>
      <td>Input gradient tensor.</td>
      <td>BFLOAT16, FLOAT16, FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional</td>
      <td>Output</td>
      <td>Weight gradient tensor.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic computation: deterministic implementation by default.

- Input shape constraints:
  - The last dimension of x must be an even number ($2H$)
  - The last dimension of gradY is $H$, corresponding to half of the last dimension of x
  - The shapes of the first n-1 dimensions of gradY and x must be the same

- Optional parameter constraints:
  - When weight is provided, yOrigin must be provided at the same time to compute gradWeight
  - The number of elements of weight must equal the product of the number of elements of x or gradY excluding the last dimension
  - The shape of yOrigin must be the same as that of gradY

- Data type constraints:
  - The data types of gradY, x, yOrigin, and gradXOut must be the same (FLOAT, FLOAT16, or BFLOAT16)
  - weight and gradWeightOutOptional must be of the FLOAT type
  - groupIndex must be of the INT64 type

- Clamp constraints:
  - The value range of clampLimit is -1.0 or >0.0
  - clampLimit=-1.0 means that the Clamp backpropagation mask is not enabled. When it is enabled, clampLimit must be >0.0

- Specification constraints:

  | Specification item | Specification | Specification description |
  |--------|------|----------|
  | B | 1~31 | - |
  | S | 0~128K | - |
  | H | 512, 768, 1024, 1536, 1792, 2048, 2560, 4096 | - |

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                             |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn invocation | [test_aclnn_swiglu_group_quant_grad](./examples/arch35/test_aclnn_swiglu_group_quant_grad.cpp) | Invoke the SwigluGroupQuantGrad operator through the [aclnnSwigluGroupQuantGrad](./docs/aclnnSwigluGroupQuantGrad_en.md) API.    |
| Graph-mode invocation | -   | Invoke the SwigluGroupQuantGrad operator by building a graph through the [operator IR](./op_graph/swiglu_group_quant_grad_proto.h). |
