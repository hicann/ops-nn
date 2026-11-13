# AscendWeightQuant

English | [简体中文](./README.md)

If you want to contribute the AscendC implementation of this operator, refer to [contribution process](../../CONTRIBUTING_en.md).

## Description

Weight quantization operator in the quantization pipeline (opFile=Null, no kernel implementation, handled only by fusion passes at graph compilation time).

- Inputs: `x` (int8, HWCN, required), `offset` (int8, HWCN, required)
- Output: `y` (int8, HWCN, required)
- Attribute: `dst_type` (optional int, supports `DT_INT8`/`DT_INT4`, defaults to `DT_INT8`)
- Supported SoCs: ascend950 / ascend960dt / ascend960pr
- aclnn: aclnn_exclude (not exposed as an aclnn interface)
