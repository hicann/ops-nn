# Quantization Introduction

Quantization is widely used in deep learning models, especially during inference. Through quantization, models can run more efficiently on hardware, reducing the consumption of computing resources and accelerating the inference process, while also lowering the storage requirements of the model.

CANN operator quantization refers to the calculation process of converting the input Tensor of matrix (cube) operators such as Matmul in neural networks from high-bit to low-bit, while generating corresponding quantization parameters scale. After low-bit cube calculation is completed, the low-bit values can be converted back to high-bit values through the quantization parameter scale, thereby ensuring the correctness of the overall calculation result (the effect is approximately equivalent to direct high-bit calculation), and effectively improving calculation efficiency.

- Static quantization: Uses pre-determined quantization parameters for quantization. In inference scenarios, quantization of weight is generally done using static quantization, which provides better quantization operator performance.
- Dynamic quantization: Uses input data to calculate quantization parameters online for quantization. In inference scenarios, quantization of activation is generally done using dynamic quantization, which can better adapt to data changes and has higher precision; in training scenarios, dynamic quantization is also generally used to improve quantization precision. Note that dynamic quantization has slightly worse quantization operator performance because quantization parameters are generated online.

## Quantization Modes

Quantization mode (also known as quantization granularity) refers to using different quantization calculation levels for different input Tensors of operators. Common quantization calculation modes include:

>Note:
>
>- The m, n, and k variables represent the sizes of different axes in Tensor calculation.
>- Left matrix and right matrix refer to the two input Tensors used for matrix multiplication calculation in cube operators. Generally, the left matrix represents activation and the right matrix represents weight. Please understand and use them according to the actual situation.

- pertensor quantization (abbreviated as T quantization): The quantization object can be either the left matrix or the right matrix, and each Tensor shares the same quantization parameter.

  Assuming the left matrix shape is (m, k) and the right matrix shape is (k, n), where k is the reduce axis, the generated quantization parameter shape is (1, ).

  <!--![Schematic](../figures/pertensor量化.png)-->

- perchannel quantization (abbreviated as C quantization): The quantization object is the right matrix, and each channel uses independent quantization parameters.

  Assuming the right matrix shape is (k, n), where k is the reduce axis, the generated quantization parameter shape is (n, ).

  <!--![Schematic](../figures/perchannel量化.png)-->

- pertoken quantization (abbreviated as K quantization): The quantization object is the left matrix, and each token uses independent quantization parameters.

  Assuming the left matrix shape is (m, k), where k is the reduce axis, the generated quantization parameter shape is (m, ).

  <!--![Schematic](../figures/pertoken量化.png)-->

- pergroup quantization (abbreviated as G quantization): The quantization object can be either the left matrix or the right matrix. Data is grouped on the reduce axis, and each group uses independent quantization parameters.
  - Assuming the left matrix shape is (m, k), where k is the reduce axis, grouping on the k axis with group size gs, the generated quantization parameter shape is (m, k/gs).
  - Assuming the right matrix shape is (k, n), where k is the reduce axis, grouping on the k axis with group size gs, the generated quantization parameter shape is (k/gs, n).

  <!--![Schematic](../figures/pergroup量化.png)-->

- perblock quantization (abbreviated as B quantization): The quantization object can be either the left matrix or the right matrix. Data is blocked on all axes, and each block uses independent quantization parameters.

  - Assuming the left matrix shape is (m, k), where k is the reduce axis, grouping data by (bs, bs) blocks on the m and k axes respectively, where bs is block size, the generated quantization parameter shape is (m/bs, k/bs).
  - Assuming the right matrix shape is (k, n), where k is the reduce axis, grouping data by (bs, bs) blocks on the k and n axes respectively, where bs is block size, the generated quantization parameter shape is (k/bs, n/bs).

  <!--![Schematic](../figures/perblock量化.png)-->

## Common Combined Quantization

- Full quantization: Generally refers to the mode of quantizing both the left and right matrices, including:
  - pertensor-perchannel quantization mode (abbreviated as T-C quantization mode)
  - pertoken-perchannel quantization mode (abbreviated as K-C quantization mode)
  - pergroup-perblock quantization mode (abbreviated as G-B quantization mode)
  - pertensor-perchannel-pergroup quantization mode (abbreviated as T-CG quantization mode)
  - perblock-perblock quantization mode (abbreviated as B-B quantization mode)
- Pseudo quantization: Generally refers to the mode of quantizing the weight matrix, including perchannel quantization mode (abbreviated as C quantization mode).
- mx quantization: Essentially Microscaling quantization, which maintains model precision at very low bits (such as 1bit) by dynamically adjusting the scaling factor. Here it refers to pergroup-pergroup quantization mode (abbreviated as G-G quantization mode), which is a special case where the quantization parameter type is FLOAT8_E8M0 and the group size is 32.
