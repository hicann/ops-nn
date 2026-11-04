# AABiasaddConvFusion

## Fusion Pattern

This fusion rule fuses the Conv convolution operator that does not contain Bias and the BiasAdd operator into a Conv convolution operator that contains a Bias input, improving the computation performance.

### Fusion Pattern 1

![Diagram of fusing Conv2D and BiasAdd](../../../docs/zh/figures/AABiasaddConvFusion_1.png)

### Fusion Pattern 2

![Diagram of fusing Conv2D and Add](../../../docs/zh/figures/AABiasaddConvFusion_2.png)

### Fusion Pattern 3

![Diagram of fusing DepthwiseConv2D and BiasAdd](../../../docs/zh/figures/AABiasaddConvFusion_3.png)

### Fusion Pattern 4

![Diagram of fusing DepthwiseConv2D and Add](../../../docs/zh/figures/AABiasaddConvFusion_4.png)

### Fusion Pattern 5

![Diagram of fusing Conv3D and BiasAdd](../../../docs/zh/figures/AABiasaddConvFusion_5.png)

### Fusion Pattern 6

![Diagram of fusing Conv3D and Add](../../../docs/zh/figures/AABiasaddConvFusion_6.png)

### Fusion Pattern 7

![Diagram of fusing Conv3D, Reshape, and Add](../../../docs/zh/figures/AABiasaddConvFusion_7.png)

## Usage Constraints

- The number of output nodes of the convolution operator must be 1.
- The number of input data edges of the convolution operator must be greater than or equal to 1.
- When the convolution operator already has a Bias, fusion is not supported.
- The scenario where the bias shape is unknown (dynamic shape) is not supported. The bias shape supports the reshape scenario of 1D (the dimension value equals the number C of convolution output channels) or 4D (at least 4 dimension values are 1).

## Supported Products

<!-- npu="310p" id1 -->
Atlas inference products
<!-- end id1 -->

<!-- npu="310b" id2 -->
Atlas 200I/500 A2 inference products
<!-- end id2 -->

<!-- npu="910" id3 -->
Atlas training products
<!-- end id3 -->

<!-- npu="910b" id4 -->
Atlas A2 products
<!-- end id4 -->

<!-- npu="950" id5 -->
Ascend 950PR&950DT products
<!-- end id5 -->
