# MatmulAdd

## Overview

MatmulAdd computes a fused matrix multiplication with optional bias addition:

    y = matmul(a, b) + bias

The operator targets the common plain ND matmul path used by Step1X-Edit. It fuses bias accumulation into the matmul kernel to avoid one extra global-memory read/write pass after matmul, which helps improve bandwidth utilization for large-batch inference workloads.

## Supported Inputs

| Name | Type | Description | Supported dtype | Format |
| ---- | ---- | ----------- | --------------- | ------ |
| a | Input | Left matrix with shape [M, K]. | float16, bfloat16 | ND |
| b | Input | Right matrix with shape [K, N]. | float16, bfloat16 | ND |
| bias | Optional input | Bias vector with shape [N]. | Same as a | ND |
| y | Output | Output matrix with shape [M, N]. | Same as a | ND |

## Shape Rules

- a and b must be 2-D tensors.
- a.shape[1] must equal b.shape[0].
- If bias is provided, it must be 1-D and its length must equal N.
- y is inferred as [a.shape[0], b.shape[1]].

## API

| API type | Description |
| -------- | ----------- |
| ACLNN | Calls the operator through aclnnMatmulAdd. |
