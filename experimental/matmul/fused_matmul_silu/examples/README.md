<!-- codespell:ignore Silu -->
<!-- cspell:ignore Silu -->

# FusedMatmulSilu Example

The eager example is built and executed by the ops-nn framework. It creates
BF16 tensors, calls `aclnnFusedMatmulSilu`, and synchronizes the stream.

After the custom package has been built and installed, run:

```bash
bash build.sh --run_example fused_matmul_silu eager cust --vendor_name=custom_nn --soc=ascend910b --experimental
```
