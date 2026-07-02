# aclnnMatmulAdd

## Function

aclnnMatmulAdd performs fused matrix multiplication and optional bias
addition:

    y = matmul(a, b) + bias

The operator supports the plain ND matrix multiplication path.

## API

    aclnnStatus aclnnMatmulAddGetWorkspaceSize(
        const aclTensor* a,
        const aclTensor* b,
        const aclTensor* bias,
        aclTensor* yOut,
        uint64_t* workspaceSize,
        aclOpExecutor** executor);

    aclnnStatus aclnnMatmulAdd(
        void* workspace,
        uint64_t workspaceSize,
        aclOpExecutor* executor,
        aclrtStream stream);

## Parameters

- a: Required 2-D tensor with shape [M, K].
- b: Required 2-D tensor with shape [K, N].
- bias: Optional 1-D tensor with shape [N]. It may be nullptr.
- yOut: Output tensor with shape [M, N].
- workspaceSize: Required output for the workspace size.
- executor: Required output for the operator executor.
- workspace: Workspace address used during execution.
- stream: ACL runtime stream.

## Constraints

- a and b must be 2-D ND tensors.
- a.shape[1] must equal b.shape[0].
- When provided, bias must be 1-D and its length must equal N.
- a, b, bias and yOut must use the same data type.
- Supported data types are FLOAT16 and BFLOAT16.
