# QuantMatmulDequant

本目录仅包含QuantMatmulDequant算子对应的aclnn接口；如您想要贡献该算子的AscendC实现，请参考[贡献流程](../../CONTRIBUTING.md)。

## 调用说明

| 调用方式   | 样例代码           | 说明                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn接口  | [test_aclnn_quant_matmul_dequant](examples/arch20/test_aclnn_quant_matmul_dequant.cpp) | 通过[aclnnQuantMatmulDequant](docs/aclnnQuantMatmulDequant.md)接口方式调用。 |