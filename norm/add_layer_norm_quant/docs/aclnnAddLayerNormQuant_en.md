# aclnnAddLayerNormQuant

[📄 View Source Code](https://gitcode.com/cann/ops-nn/tree/master/norm/add_layer_norm_quant)

## Supported Products

| Product                                                        | Supported|
| :----------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                            |    √     |
| <term>Atlas A3 training products/Atlas A3 inference products</term>    |    ×     |
| <term>Atlas A2 training products/Atlas A2 inference products</term>|    ×     |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×     |
| <term>Atlas inference products</term>                            |    ×     |
| <term>Atlas training products</term>                             |    ×     |

## Function

- The interface function: The LayerNorm operator is a common normalization operation used in large models. The AddLayerNormQuant operator is used to fuse the Add operator before LayerNorm and the LayerNorm normalization output into one or two downstream quantization operators, reducing the data transfer operations. The downstream quantization operators of LayerNorm can be Quantize, AscendQuantV2, or DynamicQuant. The specific quantization operator type is determined by the attr input parameters divMode and quantMode. When there are two downstream quantization operators, the operator types, input and output dtype combinations, and optional input combinations of the two operators must be the same.
- Formulas:
  
  $$
  x = x1 + x2 + biasOptional
  $$
  
  $$
  y = {{x-E(x)}\over\sqrt {Var(x)+epsilon}} * gamma + beta
  $$
  
  - When quantMode is set to "static", the outputs outScales1Out and outScales2Out are meaningless. Depending on the input of divMode, the fused quantization operator may be Quantize or AscendQuantV2.
    - When divMode is set to true, the fused quantization operator is Quantize. The calculation formula is as follows:
  
        $$
        y1Out = round(y / scales1Optional + zeroPoints1Optional)
        $$
  
        $$
        y2Out = round(y / scales2Optional + zeroPoints2Optional), \quad \text{if scales2Optional is present}
        $$
  
    - When divMode is set to false, the fused quantization operator is AscendQuantV2. The calculation formula is as follows:
  
        $$
        y1Out = round(y * scales1Optional + zeroPoints1Optional)
        $$
  
        $$
        y2Out = round(y * scales2Optional + zeroPoints2Optional), \quad \text{if scales2Optional is present}
        $$
  
  - When quantMode is set to "dynamic", the inputs zeroPoints1Optional and zeroPoints2Optional are meaningless. The fused quantization operator is DynamicQuant. In this case, divMode is invalid.
    - If neither scales1Optional nor scales2Optional is input, the outputs y2Out and scale2Out are meaningless and can be ignored. The calculation formula is as follows:
  
        $$
        outScales1Out = row\_max(abs(y))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out)
        $$
  
    - If only scales1Optional is input, the outputs y2Out and scale2Out are meaningless and can be ignored. The calculation formula is as follows:
  
        $$
        tmp1 = y * scales1Optional
        $$
  
        $$
        outScales1Out = row\_max(abs(tmp1))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out)
        $$
  
    - If both scales1Optional and scales2Optional are present, the outputs y2Out and scale2Out are valid. The calculation formula is as follows:
  
        $$
        tmp1 = y * scales1Optional, \quad tmp2 = y * scales2Optional
        $$
  
        $$
        outScales1Out = row\_max(abs(tmp1))/127, \quad outScales2Out = row\_max(abs(tmp2))/127
        $$
  
        $$
        y1Out = round(y / outScales1Out),\quad y2Out = round(y / outScales2Out)
        $$
  
        row\_max indicates the maximum value of each row.

## Prototype

Each operator is divided into two APIs (../../../docs/en/context/two_phase_api.md). You must call the `aclnnAddLayerNormQuantGetWorkspaceSize` API to obtain the input parameters and the workspace size required by the computation process, and then call the `aclnnAddLayerNormQuant` API to perform computation.

```Cpp
aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  const aclTensor *biasOptional,
  const aclTensor *scales1Optional,
  const aclTensor *scales2Optional,
  const aclTensor *zeroPoints1Optional,
  const aclTensor *zeroPoints2Optional,
  const char      *quantMode,
  double           epsilon,
  bool             additionalOutput,
  bool             divMode,
  aclTensor       *y1Out,
  aclTensor       *y2Out,
  aclTensor       *xOut,
  aclTensor       *outScales1Out,
  aclTensor       *outScales2Out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormQuantGetWorkspaceSize

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Usage</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (Shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x1 (aclTensor*) </td>
      <td>Input</td>
      <td>Input of the addition operation in AddLayerNorm. The operator performs the x1 + x2 + biasOptional operation and normalizes the result by layer. It corresponds to `x1` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the shape supports 1 to 8 dimensions, and the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions, and the data type can be FLOAT16 or BFLOAT16.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2 (aclTensor*) </td>
      <td>Input</td>
      <td>Input of the addition operation in AddLayerNorm. The operator performs the x1 + x2 + biasOptional operation and normalizes the result by layer. It corresponds to `x2` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the shape supports 1 to 8 dimensions, and the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions, and the data type can be FLOAT16 or BFLOAT16. </li><li>The shape is the same as that of `x1`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma (aclTensor*) </td>
      <td>Input</td>
      <td>gamma parameter for layer normalization. It corresponds to `gamma` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the shape supports 1 to 8 dimensions, and the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions, and the data type can be FLOAT16 or BFLOAT16. </li><li>The data dimension must be the same as the last several dimensions of `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta (aclTensor*) </td>
      <td>Input</td>
      <td>Beta in the LayerNorm formula, indicating the beta parameter in layer normalization. It corresponds to `beta` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions, and the data type can be FLOAT16 or BFLOAT16. </li><li>The shape can be consistent with that of `gamma`/`beta` or `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input parameter. You can pass an aclTensor that meets the following constraints or use nullptr to indicate that the optional input does not exist. Indicates the input for the addition computation in AddLayerNorm. The operator performs the computation of x1 + x2 + biasOptional and performs layer normalization on the computation result. It corresponds to `biasOptional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the shape supports 1 to 8 dimensions, and the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the shape supports 2 to 8 dimensions, and the data type can be FLOAT16 or BFLOAT16. </li><li>The shape can be consistent with that of `gamma`/`beta` or `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales1Optional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input parameter, indicating the scale/smooth input in the first quantized computation sublayer to be fused. It corresponds to `scales1Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the data type can be FLOAT16 or BFLOAT16. </li><li>The shape is the same as that of `gamma`. </li><li>For details about the value constraints when this parameter is passed, see <a href="#constraints">Constraints.</a></li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scales2Optional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input parameter, indicating the scale/smooth input in the second quantized computation sublayer to be fused. It corresponds to `scales2Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the data type can be FLOAT16 or BFLOAT16. </li><li>The shape is the same as that of `gamma`. </li><li>For details about the value constraints when an optional input parameter is passed, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints1Optional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input parameter, indicating the zeroPoints input of the first quantized computation sublayer to be fused. This parameter is valid only when quantMode is set to "static". It corresponds to `zeroPoints1Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the data type can be FLOAT16 or BFLOAT16. </li><li>The shape is the same as that of `gamma`. </li><li>For details about the value constraints when an optional input parameter is passed, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zeroPoints2Optional (aclTensor*) </td>
      <td>Input</td>
      <td>Optional input, indicating the zero points input of the second quantized computation subgraph to be fused. This parameter is valid only when quantMode is set to "static". It corresponds to `zeroPoints2Optional` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li>When quantMode is set to "static", the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li>When quantMode is set to "dynamic", the data type can be FLOAT16 or BFLOAT16. </li><li>The shape is the same as that of `gamma`. </li><li>For details about the value constraints when this parameter is passed, see <a href="#constraints">Constraints</a>.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>quantMode (char*) </td>
      <td>Input</td>
      <td>Quantization mode, which is used to determine whether the fusion operator fuses static or dynamic quantization operators. It corresponds to `quantMode` in the formula description. The value can be "static" or "dynamic".</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>epsilon (double) </td>
      <td>Input</td>
      <td>Epsilon in LayerNorm, which is added to the denominator to ensure numerical stability. corresponding to `epsilon` in the formula.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>additionalOutput (bool) </td>
      <td>Input</td>
      <td>Indicates whether to enable the output of x = x1 + x2 + biasOptional.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode (bool) </td>
      <td>Input</td>
      <td>Valid only when quantMode = "static". It indicates whether the scale method used for static quantization is multiplication or division. If true is passed, the scale is divided during operator quantization. It corresponds to `divMode` in the formula description.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*) </td>
      <td>Output</td>
      <td> indicates the result of quantizing the output y of LayerNorm by the first quantization operator. It corresponds to `y1Out` in the formula.</td>
      <td>shape must be the same as that of input x1/x2.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*) </td>
      <td>Output</td>
      <td> indicates the result of quantizing the output y of LayerNorm by the second quantization operator. It corresponds to `y2Out` in the formula.</td>
      <td>shape must be the same as that of input x1/x2.</td>
      <td>INT8</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut (aclTensor*) </td>
      <td>Output</td>
      <td>Output x of the Add result. corresponding to `x` in the formula.</td>
      <td><ul><li>Empty tensors are supported. </li><li> When quantMode is set to static, the data type can be FLOAT32, FLOAT16, or BFLOAT16. </li><li> When quantMode is dynamic, the data type can be FLOAT16 or BFLOAT16. </li><li>The shape must be the same as that of input `x1`/`x2`.</li></ul></td>
      <td>FLOAT32, FLOAT16, BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales1Out (aclTensor*) </td>
      <td>Output</td>
      <td>Indicates the output of the outScale result of the first dynamic quantization calculation. This parameter is valid only when quantMode="dynamic". It corresponds to `outScales1Out` in the formula.</td>
      <td>The shape is the shape of the input `x1` with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outScales2Out (aclTensor*) </td>
      <td>Output</td>
      <td>Indicates the output of the outScale result of the second dynamic quantization calculation. This parameter is valid only when quantMode="dynamic". It corresponds to `outScales2Out` in the formula.</td>
      <td>The shape is the shape of the input `x1` with the last dimension removed.</td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>0-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*) </td>
      <td>Output</td>
      <td>Size of the workspace to be allocated on the device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**) </td>
      <td>Output</td>
      <td>Operator executor, containing the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
  
- **Returns:**

  `aclnnStatus`: status code. For details, see [aclnn Return Code](../../../docs/en/context/aclnn_return_code.md).
  
  The first-phase API implements input parameter validation. The following error codes may be returned.

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return code</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The required input, output, or attribute is passed as a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>The hardware platform is not supported.</td>
    </tr>
    <tr>
      <td>The value of quantMode is not "static" or "dynamic".</td>
    </tr>
    <tr>
      <td>The input data type combination is invalid. For details about the valid data type combinations, see the following restrictions and descriptions.</td>
    </tr>
    <tr>
      <td>The shapes of x1, gamma, and outScales1Out must meet the following conditions:
      <ol>
      <li>The last several dimensions of gamma are inconsistent with those of x1.</li>
      <li>When the quantization mode is dynamic (that is, the value of quantMode is "dynamic"), the number of dimensions of x1 is less than 2, or the number of dimensions of gamma is not 1.</li>
      <li>When the quantization mode is dynamic (that is, the value of quantMode is "dynamic"), the shape of outScales1Out is not the shape of x1 with the last dimension removed.</li></ol></td>
    </tr>
    <tr>
      <td>The shapes of all input tensors must meet the following conditions:
      <ol>
      <li>1. The shapes of x1, x2, xOut, and y1 are different. If the optional input scales2Optional is provided, the shapes of x1, x2, xOut, y1, and y2 are different.</li>
      <li>The shapes of gamma and beta are different. If the optional inputs scales1Optional, scales2Optional, zeroPoints1Optional, and zeroPoints2Optional are provided, their shapes are different from that of gamma.</li>
      <li>When biasOptional is present, its shape is different from that of gamma and x1.</li>
      <li>When the quantization mode is dynamic (that is, the value of quantMode is "dynamic") and scales2Optional is present, the shape of outScales1Out is different from that of outScales2Out.</li>
      </ol>
      </td>
    </tr>
    <tr>
      <td>The optional inputs (scales1Optional, scales2Optional, zeroPoints1Optional, and zeroPoints2Optional) do not meet the specific combination relationship.</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormQuant

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace allocated on the device, which is obtained by the first API aclnnAddLayerNormQuantGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  `aclnnStatus`: status code. For details, see aclnn Return Cod](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Functional dimensions:
  
  * The following table lists the supported combinations of optional inputs (scales1Optional, scales2Optional, zeroPoints1Optional, and zeroPoints2Optional).
    
    | scales1Optional | scales2Optional | zeroPoints1Optional | zeroPoints2Optional | quantMode | Checks whether the domain name is valid.|
    | --------------- | --------------- | ------------------- | ------------------- | ----------------- | :------ |
    | T               | T               | T                   | T                   | "static"          | T       |
    | T               | T               | T                   | F                   | "static"          | F       |
    | T               | T               | F                   | T                   | "static"          | F       |
    | T               | T               | F                   | F                   | "static"          | T       |
    | T               | F               | T                   | T                   | "static"          | F       |
    | T               | F               | T                   | F                   | "static"          | T       |
    | T               | F               | F                   | T                   | "static"          | F       |
    | T               | F               | F                   | F                   | "static"          | T       |
    | F               | X               | X                   | X                   | "static"          | F       |
    | T               | T               | F                   | F                   | "dynamic"         | T       |
    | T               | F               | F                   | F                   | "dynamic"         | T       |
    | F               | T               | F                   | F                   | "dynamic"         | F       |
    | F               | F               | F                   | F                   | "dynamic"         | T       |
    | X               | X               | T                   | X                   | "dynamic"         | F       |
    | X               | X               | X                   | T                   | "dynamic"         | F       |

    The values are as follows:
    - `T` indicates that the optional input exists and `/` is valid.
    - `F` indicates that the optional input does not exist, and `/` is invalid.
    - `X` indicates that any case is acceptable.
- Data type support:
  - When `quantMode` is "static":
    
    | x1| x2| gamma| beta| bias| Scale1 data type| Scale2 data type| zeroPoints1 data type| zeroPoints2 data type| y1 data type| y2 data type| x| outScale1 data type| outScale2 data type|
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |
    | FLOAT32    | FLOAT32   | FLOAT32       | FLOAT32     | FLOAT32      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT32   | FLOAT32           | FLOAT32          |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | FLOAT32        | FLOAT32        | FLOAT32            | FLOAT32             | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

  - When `quantMode` is "dynamic":
    
    | x1| x2| gamma| beta| bias| Scale1 data type| Scale2 data type| zeroPoints1 data type| zeroPoints2 data type| y1 data type| y2 data type| x| outScale1 data type| outScale2 data type|
    | ---------- | --------- | ------------- | ----------- | ------------ | -------------- | -------------- | ------------------ | ------------------- | --------- | ---------- | --------- | ----------------- | :--------------- |
    | FLOAT16    | FLOAT16   | FLOAT16       | FLOAT16     | FLOAT16      | FLOAT16        | FLOAT16        | FLOAT16            | FLOAT16             | INT8      | INT8       | FLOAT16   | FLOAT32           | FLOAT32          |
    | BFLOAT16   | BFLOAT16  | BFLOAT16      | BFLOAT16    | BFLOAT16     | BFLOAT16       | BFLOAT16       | BFLOAT16           | BFLOAT16            | INT8      | INT8       | BFLOAT16  | FLOAT32           | FLOAT32          |

- Deterministic computation:
  - The aclnnAddLayerNormQuant is implemented in a deterministic manner by default.

## Examples

The following example is for reference only. For details, see [Compile and Run Sample](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_quant.h"

#define CHECK_RET(cond, return_expr)\
do {                                \
  if (!(cond)) {                    \
    return_expr;                    \
  }                                 \
} while (0)

#define LOG_PRINT(message, ...)   \
    do {                          \
  printf(message, ##__VA_ARGS__); \
} while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // (Fixed writing) Initialize resources.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevicefailed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate memory on the device.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy the data on the host to the memory on the device. 
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Compute the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device and stream. For details, see the ACL API manual.
  // Set the device ID in use.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // Handle the check as required.
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output based on the API. In this example, the test cases with and without bias input are executed respectively.
  float eps = 1e-6;
  bool additionalOut = true;
  bool divMode = true;
  const char* quantMode = "dynamic";

  std::vector<int64_t> xShape = {8, 64};
  std::vector<int64_t> gammaShape = {64};

  std::vector<int64_t> reduceShape = {8,};

  void *x1DeviceAddr = nullptr;
  void *x2DeviceAddr = nullptr;
  void *betaDeviceAddr = nullptr;
  void *gammaDeviceAddr = nullptr;
  void *biasDeviceAddr = nullptr;
  void *s1DeviceAddr = nullptr;
  void *s2DeviceAddr = nullptr;
  void *z1DeviceAddr = nullptr;
  void *z2DeviceAddr = nullptr;

  // Output device address without bias
  void *y1DeviceAddr = nullptr;
  void *y2DeviceAddr = nullptr;
  void *xDeviceAddr = nullptr;
  void *outScales1DeviceAddr = nullptr;
  void *outScales2DeviceAddr = nullptr;

  aclTensor *x1 = nullptr;
  aclTensor *x2 = nullptr;
  aclTensor *beta = nullptr;
  aclTensor *gamma = nullptr;
  aclTensor *bias = nullptr;
  aclTensor *s1 = nullptr;
  aclTensor *s2 = nullptr;
  aclTensor *z1 = nullptr;
  aclTensor *z2 = nullptr;

  // Used for aclTensor that does not include bias
  aclTensor *y1 = nullptr;
  aclTensor *y2 = nullptr;
  aclTensor *x = nullptr;
  aclTensor *outScales1 = nullptr;
  aclTensor *outScales2 = nullptr;

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t reduceShapeSize = GetShapeSize(reduceShape);

  std::vector<float> x1HostData(xShapeSize, 0x3C00);
  std::vector<float> x2HostData(xShapeSize, 0x3C00);
  std::vector<float> gammaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> betaHostData(gammaShapeSize, 0x3C00);
  std::vector<float> biasHostData(gammaShapeSize, 0x3C00);

  std::vector<float> s1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> s2HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z1HostData(gammaShapeSize, 0x3C00);
  std::vector<float> z2HostData(gammaShapeSize, 0x3C00);

  // Used for HostData that does not include bias
  std::vector<int8_t> y1HostData(xShapeSize, 0);
  std::vector<int8_t> y2HostData(xShapeSize, 0);
  std::vector<float> xHostData(xShapeSize, 0);
  std::vector<float> outScales1HostData(reduceShapeSize, 0);
  std::vector<float> outScales2HostData(reduceShapeSize, 0);

  // Create a self aclTensor.
  ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(betaHostData,  gammaShape, & betaDeviceAddr, aclDataType::ACL_FLOAT16, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(biasHostData, gammaShape, &biasDeviceAddr, aclDataType::ACL_FLOAT16, &bias);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s1HostData, gammaShape, &s1DeviceAddr, aclDataType::ACL_FLOAT16, &s1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(s2HostData, gammaShape, &s2DeviceAddr, aclDataType::ACL_FLOAT16, &s2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z1HostData, gammaShape, &z1DeviceAddr, aclDataType::ACL_FLOAT16, &z1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(z2HostData, gammaShape, &z2DeviceAddr, aclDataType::ACL_FLOAT16, &z2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Create an aclTensor that does not include bias.
  ret = CreateAclTensor(y1HostData, xShape, &y1DeviceAddr, aclDataType::ACL_INT8, &y1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(y2HostData, xShape, &y2DeviceAddr, aclDataType::ACL_INT8, &y2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales1HostData, reduceShape, &outScales1DeviceAddr, aclDataType::ACL_FLOAT, &outScales1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(outScales2HostData, reduceShape, &outScales2DeviceAddr, aclDataType::ACL_FLOAT, &outScales2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // Example of calling the aclnnAddLayerNormQuant API, including the cases with and without bias
  // 3. Call the CANN operator library API. Modify the API name to the actual one.

  // 3.1 Example that does not include the optional bias input
  // Call the first part of the aclnnAddLayerNormQuant API.
  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;
  ret = aclnnAddLayerNormQuantGetWorkspaceSize(x1, x2, gamma, beta, bias, s1, s2, nullptr, nullptr, quantMode, eps, additionalOut, divMode, y1, y2, x, outScales1, outScales2, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // Allocate device memory based on workspaceSize computed by the first-phase API.
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // Call the second part of the aclnnAddLayerNormQuant API.
  ret = aclnnAddLayerNormQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormQuant failed. ERROR: %d\n", ret); return ret);

  // 4. (Boilerplate) Wait until the task execution is complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. Obtain the output value and copy the result from the device memory to the host. Modify the code based on the API definition.

  auto y1Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY1(y1Size, 0);
  ret = aclrtMemcpy(resultDataY1.data(), resultDataY1.size() * sizeof(resultDataY1[0]), y1DeviceAddr, y1Size * sizeof(resultDataY1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y1 output");
  for (int64_t i = 0; i < y1Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY1[i]);
  }

  auto y2Size = GetShapeSize(xShape);
  std::vector<int8_t> resultDataY2(y2Size, 0);
  ret = aclrtMemcpy(resultDataY2.data(), resultDataY2.size() * sizeof(resultDataY2[0]), y2DeviceAddr, y2Size * sizeof(resultDataY2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant y2 output");
  for (int64_t i = 0; i < y2Size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultDataY2[i]);
  }

  auto xSize = GetShapeSize(xShape);
  std::vector<float> resultDataX(xSize, 0);
  ret = aclrtMemcpy(resultDataX.data(), resultDataX.size() * sizeof(resultDataX[0]), xDeviceAddr, xSize * sizeof(resultDataX[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant x output");
  for (int64_t i = 0; i < xSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataX[i]);
  }

  auto outScale1Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale1(outScale1Size, 0);
  ret = aclrtMemcpy(resultDataOutScale1.data(), resultDataOutScale1.size() * sizeof(resultDataOutScale1[0]), outScales1DeviceAddr, outScale1Size * sizeof(resultDataOutScale1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale1 output");
  for (int64_t i = 0; i < outScale1Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale1[i]);
  }

  auto outScale2Size = GetShapeSize(reduceShape);
  std::vector<float> resultDataOutScale2(outScale2Size, 0);
  ret = aclrtMemcpy(resultDataOutScale2.data(), resultDataOutScale2.size() * sizeof(resultDataOutScale2[0]), outScales2DeviceAddr, outScale2Size * sizeof(resultDataOutScale2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from Deviceto host failed. ERROR: %d\n", ret); return ret);
  LOG_PRINT("==== AddLayerNormQuant outScale2 output");
  for (int64_t i = 0; i < outScale2Size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultDataOutScale2[i]);
  }


  // 6. Release aclTensor and aclScalar. Modify the code based on the API definition.
  aclDestroyTensor(x1);
  aclDestroyTensor(x2);
  aclDestroyTensor(beta);
  aclDestroyTensor(gamma);
  aclDestroyTensor(bias);
  aclDestroyTensor(s1);
  aclDestroyTensor(s2);
  aclDestroyTensor(z1);
  aclDestroyTensor(z2);

  aclDestroyTensor(y1);
  aclDestroyTensor(y2);
  aclDestroyTensor(x);
  aclDestroyTensor(outScales1);
  aclDestroyTensor(outScales2);

  // 7. Release device resources. Modify the code based on the API definition.
  aclrtFree(x1DeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(biasDeviceAddr);
  aclrtFree(s1DeviceAddr);
  aclrtFree(s2DeviceAddr);
  aclrtFree(z1DeviceAddr);
  aclrtFree(z2DeviceAddr);

  aclrtFree(y1DeviceAddr);
  aclrtFree(y2DeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(outScales1DeviceAddr);
  aclrtFree(outScales2DeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
