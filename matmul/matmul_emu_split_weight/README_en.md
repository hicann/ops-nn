# MatmulEmuSplitWeight

English | [简体中文](./README.md)

## Applicable Products

| Product                                                     | Support |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                   |    √     |
| <term>Atlas A3 products</term>|    √     |
| <term>Atlas A2 products</term> |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                  |    ×     |
| <term>Atlas inference products</term>                          |    ×     |
| <term>Atlas training products</term>                          |    ×     |

## Function

- Operator function: uses fused dual-path BF16 GEMM to emulate a matrix multiplication with FP32 accuracy. Offline, the FP32 weight is split into a high-bit BF16 weight and a low-bit residual BF16 weight; at the inference stage, two BF16 GEMM operations are executed and linearly combined, the activation values remain BF16 throughout, and both matrix multiplications run on the Cube.

- Formula:

  $$
  Y = X \times W_{high} + scale \times (X \times W_{low})
  $$

  Here $X$ is the BF16 activation matrix, $W_{high}$ is the high-bit weight obtained by truncating the FP32 weight to BF16, $W_{low}$ is the low-bit residual weight obtained by dividing the FP32 weight residual by scale and truncating to BF16, and $scale$ is the scale factor (typical value 1/256=0.00390625).

- Offline weight splitting formula:

  $$
  W_{high} = \text{BF16}(W_{fp32})
  $$

  $$
  W_{low} = \text{BF16}\left(\frac{W_{fp32} - W_{high}.float()}{scale}\right)
  $$

## Parameters

  <table style="undefined;table-layout: fixed; width: 962px"><colgroup>
  <col style="width: 170px">
  <col style="width: 170px">
  <col style="width: 310px">
  <col style="width: 212px">
  <col style="width: 100px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Activation matrix, corresponding to X in the formula. When transpose_x=false, the shape is [M, K]; when transpose_x=true, the shape is [K, M].</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_high</td>
      <td>Input</td>
      <td>High-bit weight, corresponding to W_high in the formula. When transpose_w=false, the shape is [K, N]; when transpose_w=true, the shape is [N, K].</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_low</td>
      <td>Input</td>
      <td>Low-bit residual weight, corresponding to W_low in the formula. Its shape must be exactly consistent with w_high.</td>
      <td>BFLOAT16</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Computed output matrix, corresponding to Y in the formula. Only FP32 output is supported (yDtype=0). The shape is [M, N].</td>
      <td>FLOAT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>w_low_scale</td>
      <td>Attribute</td>
      <td>Scale factor, corresponding to scale in the formula. Only 1/256 (0.00390625) is supported.</td>
      <td>FLOAT</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_x</td>
      <td>Attribute</td>
      <td>Specifies whether to transpose the input x.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transpose_w</td>
      <td>Attribute</td>
      <td>Specifies whether to transpose the weights w_high/w_low.</td>
      <td>BOOL</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y_dtype</td>
      <td>Attribute</td>
      <td>Output data type identifier. Only 0 (FP32) is supported, and other values report an error.</td>
      <td>INT</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

## Constraints

- x, w_high, w_low, and y are all 2-dimensional.
- The K dimension of x must be consistent with the K dimension of w_high/w_low.
- The value ranges of M, K, and N are (0, INT32_MAX].
- transpose_x and transpose_w default to false.

## Invocation Description

- <term>Atlas A2 products and Atlas A3 products</term>:

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn interface  | [test_aclnn_matmul_emu_split_weight](examples/test_aclnn_matmul_emu_split_weight.cpp) | Invokes the MatmulEmuSplitWeight operator through<br>the [aclnnMatmulEmuSplitWeight](docs/aclnnMatmulEmuSplitWeight_en.md)<br>interface. |
  | torch interface  | [matmul_emu_split_weight](torch_extension/matmul_emu_split_weight.py) | Invokes the MatmulEmuSplitWeight operator through<br>the [torchapi](docs/torchapi_matmul_emu_split_weight_en.md)<br>interface. |

- <term>Ascend 950PR&950DT products</term>:

  | Invocation method   | Sample code           | Description                                         |
  | ---------------- | --------------------------- | --------------------------------------------------- |
  | aclnn interface  | [test_aclnn_matmul_emu_split_weight](examples/arch35/test_aclnn_matmul_emu_split_weight.cpp) | Invokes the MatmulEmuSplitWeight operator through<br>the [aclnnMatmulEmuSplitWeight](docs/aclnnMatmulEmuSplitWeight_en.md)<br>interface. |
  | torch interface  | [matmul_emu_split_weight](torch_extension/matmul_emu_split_weight.py) | Invokes the MatmulEmuSplitWeight operator through<br>the [torchapi](docs/torchapi_matmul_emu_split_weight_en.md)<br>interface. |
