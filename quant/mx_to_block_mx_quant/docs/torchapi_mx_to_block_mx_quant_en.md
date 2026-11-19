# mx_to_block_mx_quant

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T21:36:48.691Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Not Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Not Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not Supported
<!-- end id6 -->

## Function

- Description: Converts the FLOAT4 tensor quantized by calling npu_dynamic_mx_quant together with FLOAT8_E8M0 scale factors into the FLOAT8 blockwise quantization format, and outputs quantization scales along the -1 and -2 axes.

- Formula:

  $$
  \begin{aligned}
  scale_{fp8} &= max(mxscale_{fp4\_block}) / 2^{MAX\_OFFSET} \\
  offset &= mxscale_{fp4\_block} / scale_{fp8} \\
  x_{fp8} &= x_{fp4} \times offset
  \end{aligned}
  $$

  - where $mxscale_{fp4\_block}$ is the FP8_E8M0 scale factor provided by the input mxscale; $MAX\_OFFSET$ is the exponent offset between the input and output data types; $x_{fp4}$ is the quantized FLOAT4 tensor; $x_{fp8}$ is the converted FLOAT8 tensor.

  - MAX_OFFSET mapping table:

    | Input    | Output     | MAX_OFFSET |
    | -------- | ---------- | ---------- |
    | FP4_E2M1 | FP8_E5M2   | 13         |
    | FP4_E1M2 | FP8_E5M2   | 15         |
    | FP4_E2M1 | FP8_E4M3FN | 6          |
    | FP4_E1M2 | FP8_E4M3FN | 8          |

## Function Prototype

```python
cann_ops_nn.mx_to_block_mx_quant(x, mxscale, *, dst_type=292, x_type=296)
    -> (Tensor, Tensor, Tensor)
```

## Mandatory

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Parameter</th>
    <th>Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>x</td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>Quantization tensor input to the operator, x_fp4 in the formula.</td>
        <td>FLOAT4_E2M1, FLOAT4_E1M2</td>
        <td>2-3</td>
    </tr>
    <tr>
        <td>mxscale </td>
        <td>Tensor</td>
        <td>Mandatory</td>
        <td>Quantization scale computed by calling DynamicMxQuant, mxscale_fp4_block in the formula. The shape must satisfy the formula in Constraints.</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
    <tr>
        <td>dst_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Specifies the data type of output y. Currently, only the input range {291, 292} is supported, corresponding to the data type of output y {291: FLOAT8_E5M2, 292: FLOAT8_E4M3FN}. Defaults to 292 (FLOAT8_E4M3FN). If not specified, the default value is used. Alternatively, the corresponding torch_npu extended dtype object can be passed: torch_npu.float8_e5m2 is equivalent to 291, and torch_npu.float8_e4m3fn is equivalent to 292.</td>
        <td>INT</td>
        <td>-</td>
    </tr>
    <tr>
        <td>x_type</td>
        <td>int</td>
        <td>Optional</td>
        <td>Indicates the actual data type of input x. Currently, only the input range {296, 297} is supported, corresponding to the data type of input x {296: FLOAT4_E2M1, 297: FLOAT4_E1M2}. Defaults to 296 (FLOAT4_E2M1). If not specified, the default value is used. Alternatively, the corresponding torch_npu extended dtype object can be passed: torch_npu.float4_e2m1fn_x2 is equivalent to 296, and torch_npu.float4_e1m2fn_x2 is equivalent to 297.</td>
        <td>INT</td>
        <td>-</td>
    </tr>
</tbody>
</table>

## Outputs

<table style="undefined;table-layout: fixed; width:1625px"><colgroup>
<col style="width: 147px">
<col style="width: 132px">
<col style="width: 132px">
<col style="width: 480px">
<col style="width: 330px">
<col style="width: 280px">
</colgroup>
<thead>
<tr>
    <th>Output</th>
    <th>Output Type</th>
    <th>Optional/Required</th>
    <th>Description</th>
    <th>Data Type</th>
    <th>Dimension (Shape)</th>
</tr>
</thead>
<tbody>
    <tr>
        <td>y</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Quantization output, which is x_fp8 in the Formula. The shape is the same as that of x, and the data type is determined by dst_type.</td>
        <td>FLOAT8_E5M2, FLOAT8_E4M3FN</td>
        <td>2-3</td>
    </tr>
    <tr>
        <td>scale1</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Represents the quantization scale corresponding to each group along the -1 axis. The shape must satisfy the formula in Constraints.</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
    <tr>
        <td>scale2</td>
        <td>Tensor</td>
        <td>Required</td>
        <td>Represents the quantization scale corresponding to each group on the -2 axis. The output needs to be interleaved for every two rows of data. The shape must satisfy the formula in the Constraints section.</td>
        <td>FLOAT8_E8M0</td>
        <td>3-4</td>
    </tr>
</tbody>
</table>

## Constraints

- `x` supports only 2D or 3D input, with the -2 axis being a multiple of 64 and the -1 axis being a multiple of 2.
- The shape constraints of `x`, `mxscale`, `scale1`, and `scale2` are as follows:
  - rank(mxscale) = rank(x) + 1.
  - mxscale.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - mxscale.shape[-1] = 2.
  - Other dimensions are the same as input `x`.
- Shape constraints of output `scale1` are as follows:
  - rank(scale1) = rank(x) + 1.
  - scale1.shape[-2] = (Ceil(x.shape[-1] / 32) + 2 - 1) / 2.
  - scale1.shape[-1] = 2.
  - Other dimensions are the same as input `x`.
- Shape constraints of output `scale2` are as follows:
  - rank(scale2) = rank(x) + 1.
  - scale2.shape[-3] = ((Ceil(x.shape[-2] / 32) + 2 - 1) / 2) * 2 / 2.
  - scale2.shape[-2] = x.shape[-1].
  - scale2.shape[-1] = 2.
  - Other dimensions are consistent with the input `x`.
- This API supports use in inference scenarios.
- This API supports calls in single-operator mode and TorchAir graph mode.

## Deterministic Computation

Defaults to a deterministic computation.

## Example

- Call in single-operator mode (eager)

    The example first constructs a random bf16 input, quantizes it along the -1 axis in groups of 32 via npu_dynamic_mx_quant to obtain the uint8-packed FLOAT4_E2M1 tensor x and the FLOAT8_E8M0 scale mxscale; then calls mx_to_block_mx_quant to convert them into the FLOAT8_E5M2 (dst_type=torch_npu.float8_e5m2, equivalent to the enum value 291) block quantization format, outputting the quantization result y and the quantization scales scale1 and scale2 along the -1 and -2 axes. Expected output shapes: x is [64, 32], mxscale is [64, 1, 2], y is [64, 64], scale1 is [64, 1, 2], scale2 is [1, 64, 2]; mxscale/scale1/scale2 are printed as uint8 views for easy verification.

    ```python
    import torch
    import torch_npu
    import cann_ops_nn

    # Construct a 64x64 bf16 random input and move it to NPU
    x_tmp = torch.randn((64, 64), dtype=torch.bfloat16).npu()
    # Quantize along the -1 axis in groups of block_size=32 to obtain
    # the FLOAT4_E2M1 tensor x and the FLOAT8_E8M0 scale mxscale
    x, mxscale = torch_npu.npu_dynamic_mx_quant(
        x_tmp,
        axis=-1,
        round_mode="rint",
        dst_type=torch_npu.float4_e2m1fn_x2,
        block_size=32,
        scale_alg=0,
    )
    print("npu_dynamic_mx_quant:============")
    print(x)

    # Convert to the FLOAT8 block quantization format, outputting y, scale1, and scale2
    # dst_type=torch_npu.float8_e5m2 is equivalent to the enum value 291,
    # corresponding to output FLOAT8_E5M2
    # x_type=torch_npu.float4_e2m1fn_x2 is equivalent to the enum value 296,
    # corresponding to input FLOAT4_E2M1, consistent with the quantization
    # output in the previous step
    y, scale1, scale2 = cann_ops_nn.mx_to_block_mx_quant(
        x, mxscale, dst_type=torch_npu.float8_e5m2, x_type=torch_npu.float4_e2m1fn_x2
    )
    print("npu_mx_to_block_mx_quant:============")
    # Print the shape and data of each tensor. mxscale/scale1/scale2 are FLOAT8_E8M0
    # and are printed as uint8 views for easy verification
    print("x:============", x.shape, x.cpu())
    print("mxscale:============", mxscale.shape, mxscale.cpu().view(torch.uint8))
    print("y:============", y.shape, y.cpu())
    print("scale1:============", scale1.shape, scale1.cpu().view(torch.uint8))
    print("scale2:============", scale2.shape, scale2.cpu().view(torch.uint8))
    ```

- Call in graph mode (torchair)

    The example wraps the operator in NetModel and compiles it in torchair graph mode via torch.compile with the NPU backend; the input x is a random uint8-packed FLOAT4_E2M1 tensor and mxscale is the FLOAT8_E8M0 scale, with conversion parameters dst_type=torch_npu.float8_e4m3fn (equivalent to the enum value 292, corresponding to FLOAT8_E4M3FN) and x_type=torch_npu.float4_e2m1fn_x2 (equivalent to the enum value 296, corresponding to FLOAT4_E2M1), outputting y, scale1, and scale2 and printing their shapes and uint8 views for easy verification. Expected output shapes: x is [64, 32], mxscale is [64, 1, 2], y is [64, 64], scale1 is [64, 1, 2], scale2 is [1, 64, 2].

    ```python
    import torch
    import torch_npu
    import torchair
    import cann_ops_nn


    class NetModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(
            self,
            x,
            mxscale,
            dst_type=torch_npu.float8_e4m3fn,
            x_type=torch_npu.float4_e2m1fn_x2,
        ):
            # dst_type=torch_npu.float8_e4m3fn is equivalent to the enum
            # value 292, corresponding to FLOAT8_E4M3FN
            # x_type=torch_npu.float4_e2m1fn_x2 is equivalent to the enum
            # value 296, corresponding to FLOAT4_E2M1
            return cann_ops_nn.mx_to_block_mx_quant(
                x, mxscale, dst_type=dst_type, x_type=x_type
            )


    def mx_to_block_mx_quant_test(x_dtype, dst_type, x_type):
        # Construct the uint8-packed FLOAT4 random input x and the FLOAT8_E8M0
        # scale mxscale, then move them to NPU
        x = torch.randn((64, 32), dtype=torch.float32).to(dtype=x_dtype)
        x = x.npu()
        mxscale = (
            torch.randn((64, 1, 2), dtype=torch.float32)
            .to(dtype=torch.float8_e8m0fnu)
            .npu()
        )
        model = NetModel()
        # Obtain the torchair NPU backend for graph-mode compilation
        config = torchair.CompilerConfig()
        npu_backend = torchair.get_npu_backend(compiler_config=config)
        model = torch.compile(model, backend=npu_backend, dynamic=True, fullgraph=True)
        # Execute mx_to_block_mx_quant in graph mode, outputting y, scale1, and scale2
        y, scale1, scale2 = model(x, mxscale, dst_type=dst_type, x_type=x_type)
        # Print the shape and data of each tensor. mxscale/scale1/scale2 are
        # printed as uint8 views for easy verification
        print("x:============", x.shape, x.cpu())
        print("mxscale:============", mxscale.shape, mxscale.cpu().view(torch.uint8))
        print("y:============", y.shape, y.cpu())
        print("scale1:============", scale1.shape, scale1.cpu().view(torch.uint8))
        print("scale2:============", scale2.shape, scale2.cpu().view(torch.uint8))


    if __name__ == "__main__":
        # x is packed as uint8; torch_npu.float8_e4m3fn is equivalent to the
        # enum value 292; torch_npu.float4_e2m1fn_x2 is equivalent to the
        # enum value 296
        mx_to_block_mx_quant_test(
            torch.uint8, torch_npu.float8_e4m3fn, torch_npu.float4_e2m1fn_x2
        )
    ```
