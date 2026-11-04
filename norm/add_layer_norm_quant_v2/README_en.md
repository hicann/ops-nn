# AddLayerNormQuantV2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                             |    √     |
| <term>Atlas A3 products</term>     |    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    √     |
| <term>Atlas training products</term>                              |    ×     |

## Function

- Description: The LayerNorm operator is a normalization operation commonly used in large models. The AddLayerNormQuantV2 operator fuses the Add operator before LayerNorm and the LayerNorm normalization output feeding 1 or 2 downstream quantization operators, reducing data move-in and move-out operations. The quantization operators downstream of LayerNorm can be the Quantize, AscendQuantV2, or DynamicQuant operator, and the specific quantization operator type is determined by the attr parameters div_mode and quant_mode. When there are 2 downstream quantization operators, the operator types, the input/output dtype combinations, and the optional input combinations of the 2 quantization operators must be exactly the same.
- Formula:

  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$

  - When quant_mode is "static", the outputs out_scales1 and out_scales2 have no practical meaning. Depending on the input of div_mode, the fused quantization operator is Quantize or AscendQuantV2:

    - When div_mode is true, the fused quantization operator is Quantize, and the formula is as follows:

      $$
      y1Out = round(y / scales1 + zero_points1)
      $$

      $$
      y2Out = round(y / scales2 + zero_points2), \quad \text{if and only if scales2 exists}
      $$

    - When div_mode is false, the fused quantization operator is AscendQuantV2, and the formula is as follows:

      $$
      y1Out = round(y * scales1 + zero_points1)
      $$

      $$
      y2Out = round(y * scales2 + zero_points2), \quad \text{if and only if scales2 exists}
      $$

  - When quant_mode is "dynamic", the inputs zero_points1 and zero_points2 have no practical meaning. The fused quantization operator is DynamicQuant, and div_mode is invalid in this case:

    - If neither scales1 nor scales2 has an input, the outputs y2Out and scale2Out have no practical meaning and can be ignored. The formula is as follows:

      $$
      out_scales1 = row\_max(abs(y))/127
      $$

      $$
      y1Out = round(y / out_scales1)
      $$

    - If only scales1 has an input, the outputs y2Out and scale2Out have no practical meaning and can be ignored. The formula is as follows:

      $$
      tmp1 = y * scales1
      $$

      $$
      out_scales1 = row\_max(abs(tmp1))/127
      $$

      $$
      y1Out = round(tmp1 / out_scales1)
      $$

    - If both scales1 and scales2 exist, the outputs y2Out and scale2Out are valid. The formula is as follows:

      $$
      tmp1 = y * scales1, \quad tmp2 = y * scales2
      $$

      $$
      out_scales1 = row\_max(abs(tmp1))/127, \quad out_scales2 = row\_max(abs(tmp2))/127
      $$

      $$
      y1Out = round(tmp1 / out_scales1),\quad y2Out = round(tmp2 / out_scales2)
      $$

      where row\_max represents computing the maximum value of each row.

## Parameters

<table style="undefined;table-layout: fixed; width: 1005px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 352px">
  <col style="width: 213px">
  <col style="width: 100px">
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
      <td>Input of the addition computation in AddLayerNorm, corresponding to x1 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>Input</td>
      <td>Input of the addition computation in AddLayerNorm, corresponding to x2 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>Input</td>
      <td>gamma parameter in layer normalization, corresponding to gamma in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>beta</td>
      <td>Input</td>
      <td>beta parameter in layer normalization, corresponding to beta in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>bias</td>
      <td>Optional input</td>
      <td>Input of the addition computation in AddLayerNormQuantV2, corresponding to bias in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales1</td>
      <td>Optional input</td>
      <td>scale/smooth input of the first fused quantization operator, corresponding to scales1 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>scales2</td>
      <td>Optional input</td>
      <td>scale/smooth input of the second fused quantization operator, corresponding to scales2 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points1</td>
      <td>Optional input</td>
      <td>zeroPoints input of the first fused quantization operator, corresponding to zero_points1 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>zero_points2</td>
      <td>Optional input</td>
      <td>zeroPoints input of the second fused quantization operator, corresponding to zero_points2 in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>quant_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Determines whether the fused operator fuses a static or dynamic quantization operator, corresponding to quant_mode in the formula. The value can be "static" or "dynamic".</li><li>Default value: "dynamic".</li></ul></td>
      <td>String</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon</td>
      <td>Optional attribute</td>
      <td><ul><li>Value added to the denominator to ensure numerical stability and prevent division-by-zero errors, corresponding to `epsilon` in the formula.</li><li>Default value: 1e-5.</li><li>It should be greater than 0. When a value less than or equal to 0 is passed in, the operator still executes (inf/NaN may be produced when the variance is 0), and only a warning is printed.</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additional_output</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether to enable the output of x=x1+x2+bias.</li><li>Default value: false.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>div_mode</td>
      <td>Optional attribute</td>
      <td><ul><li>Specifies whether the static quantization processes scale by multiplication or division, corresponding to div_mode in the formula.</li><li>Default value: true.</li></ul></td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to y1Out in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y2</td>
      <td>Output</td>
      <td>Quantization output tensor, corresponding to y2Out in the formula.</td>
      <td>INT8</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>x</td>
      <td>Output</td>
      <td>Sum of x1 and x2, corresponding to `x` in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>layernormRes</td>
      <td>Output</td>
      <td>Result output of layernorm, corresponding to y in the formula.</td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales1</td>
      <td>Output</td>
      <td>Dynamic quantization scaling result computed through scales1, corresponding to out_scales1 in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>out_scales2</td>
      <td>Output</td>
      <td>Dynamic quantization scaling result computed through scales2, corresponding to out_scales2 in the formula.</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- Product constraints:

  - <term>Ascend 950PR&950DT products</term>:
    - `quant_mode` supports "static" or "dynamic"; when `quant_mode="dynamic"`, `gamma` supports 2 to 8 dimensions.
    - Both perChannel and perTensor quantization modes are supported: in perChannel mode, the shapes of `scales1`, `scales2`, `zero_points1`, and `zero_points2` are the same as `gamma`; in perTensor mode, the shapes of the preceding inputs are `[1]`, and `gamma` is `[1, the last dimension of x1]`.
    - The data types of `scales1`, `scales2`, `zero_points1`, and `zero_points2` can be the same as `x1`, or FLOAT32 (when `x1` is FLOAT16/BFLOAT16).

  - <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
    - The current scenario supports only "static", "false", and "false" for `quant_mode`, `additional_output`, and `div_mode` respectively.
    - The current scenario supports only empty `scales2` and `zero_points2`.
    - The current scenario supports only the perTensor quantization mode, that is, the shapes of `scales1` and `zero_points1` are `[1]`, and the shapes of `gamma` and `beta` are `[1, the last dimension of x1]`.
    - The current scenario supports only the data types of `scales1` and `zero_points1` being the same as `x1`.

  - <term>Atlas inference products</term>:
    - The current scenario supports only "static", "false", and "false" for `quant_mode`, `additional_output`, and `div_mode` respectively.
    - The current scenario supports only empty `scales2` and `zero_points2`.
    - The current scenario supports only the perTensor quantization mode, that is, the shapes of `scales1` and `zero_points1` are `[1]`, and the shapes of `gamma` and `beta` are `[1, the last dimension of x1]`.
    - The current scenario supports only the data types of `scales1` and `zero_points1` being the same as `x1`.
    - The BFLOAT16 type is not supported in the current scenario.
    - The norm axis length of inputs `x1` and `x2` must be greater than or equal to 32 bytes.

- Shape constraints:

  - The shape of `beta` must be the same as `gamma`.
  - When `bias` exists, its shape must be the same as `x1` or `gamma`.
  - The number of dimensions of `gamma` cannot exceed the number of dimensions of `x1`. In perTensor mode, the shape of `gamma` is `[1, the last dimension of x1]`; in other cases, the shape of `gamma` must be equal to the shape composed of the last `gammaDimNum` dimensions of `x1`.
  - When `quant_mode="dynamic"`, the number of dimensions of `x1` must be no less than 2, and `out_scales1` and `out_scales2` are the prefix of `x1` after removing the trailing dimensions occupied by `gamma`.

- Functional dimensions:

  * The supported optional input combinations of the optional inputs (scales1, scales2, zero_points1, zero_points2) are as follows:

    | scales1 | scales2 | zero_points1 | zero_points2 | quant_mode | Valid |
    | --------------- | --------------- | -------------------- | -------------------- | ---------- | :------: |
    | T               | T               | T                    | T                    | "static"   |    T     |
    | T               | T               | T                    | F                    | "static"   |    F     |
    | T               | T               | F                    | T                    | "static"   |    F     |
    | T               | T               | F                    | F                    | "static"   |    T     |
    | T               | F               | T                    | T                    | "static"   |    F     |
    | T               | F               | T                    | F                    | "static"   |    T     |
    | T               | F               | F                    | T                    | "static"   |    F     |
    | T               | F               | F                    | F                    | "static"   |    T     |
    | F               | X               | X                    | X                    | "static"   |    F     |
    | T               | T               | F                    | F                    | "dynamic"  |    T     |
    | T               | F               | F                    | F                    | "dynamic"  |    T     |
    | F               | T               | F                    | F                    | "dynamic"  |    F     |
    | F               | F               | F                    | F                    | "dynamic"  |    T     |
    | X               | X               | T                    | X                    | "dynamic"  |    F     |
    | X               | X               | X                    | T                    | "dynamic"  |    F     |

    Where: `T` indicates that the optional input exists or the combination is valid; `F` indicates that the optional input does not exist or the combination is invalid; `X` indicates any case.

  * Data type support description:

    - When `quant_mode` is "static":

      | x1 | x2 | gamma | beta | bias | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x | layernorm_res | out_scales1 | out_scales2 |
      | -- | -- | ----- | ---- | ---- | ------- | ------- | ------------ | ------------ | -- | -- | - | ------------- | ----------- | ----------- |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16 | FLOAT16 | FLOAT16 | FLOAT16 | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |
      | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | FLOAT32  | FLOAT32  | FLOAT32 | FLOAT32 |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 | FLOAT32 | FLOAT32 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |

    - When `quant_mode` is "dynamic":

      | x1 | x2 | gamma | beta | bias | scales1 | scales2 | zero_points1 | zero_points2 | y1 | y2 | x | layernorm_res | out_scales1 | out_scales2 |
      | -- | -- | ----- | ---- | ---- | ------- | ------- | ------------ | ------------ | -- | -- | - | ------------- | ----------- | ----------- |
      | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | FLOAT16  | INT8 | INT8 | FLOAT16  | FLOAT16  | FLOAT32 | FLOAT32 |
      | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | BFLOAT16 | INT8 | INT8 | BFLOAT16 | BFLOAT16 | FLOAT32 | FLOAT32 |

## Invocation Description

| Invocation method | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API  | [test_aclnn_add_layer_norm_quant_v2](examples/test_aclnn_add_layer_norm_quant_v2.cpp) | Invoke the AddLayerNormQuantV2 operator through the [aclnnAddLayerNormQuantV2](docs/aclnnAddLayerNormQuantV2_en.md) API. |
