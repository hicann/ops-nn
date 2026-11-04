# ApplyCamePart2

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
|:---|:---:|
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Operator function: Part2 of the 4-part split of the CAME (Confidence-guided Adaptive Memory Efficient Stochastic Optimizer) optimizer. After the previous part (ApplyCamePart1) has computed the row/column/total sum statistics of grad (sum_grad_r, sum_grad_c, and sum_grad_rc), this operator performs an EMA update on the row/column second-moment estimates r and c of the previous step, computes the normalized update direction u through confidence normalization, and accumulates the sum of squares of u into sum_square_u for Part3 (clipping + first moment + residual statistics) to consume. r and c are updated in place.
- Formula (aligned with the `ProcessU`/`ComputeR`/`ComputeC` functions of the kernel implementation `op_kernel/apply_came_part2.cpp`, where $N=\text{grad.shape}[0]$ is the number of rows and $M=\text{grad.shape}[1]$ is the number of columns):

  $$
  r_{\text{out}} = \beta_2 \cdot r + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_r}}{M}
  $$

  $$
  c_{\text{out}} = \beta_2 \cdot c + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_c}}{N}
  $$

  $$
  \text{sum\_r\_val} = \text{sum\_r}\ (\text{if absent, } \sum r),\quad \text{denom} = \beta_2 \cdot \frac{\text{sum\_r\_val}}{N} + (1 - \beta_2) \cdot \frac{\text{sum\_grad\_rc}}{M \cdot N}
  $$

  $$
  u = \frac{\text{grad}}{\sqrt{r_{\text{out}} \cdot c_{\text{out}} \,/\, \text{denom}}} \quad (r_{\text{out}} \text{ is broadcast by row, and } c_{\text{out}} \text{ is broadcast by column})
  $$

  $$
  \text{sum\_square\_u} = \sum (u \odot u)
  $$

  Here $r_{\text{out}}$ and $c_{\text{out}}$ are the row/column second moments after the EMA update. The normalization denominator `denom` of $u$ is the confidence normalization term (`beta2*sum_r/N` reflects the magnitude of the second moment, and `(1-beta2)*sum_grad_rc/(M*N)` reflects the magnitude of the current gradient).

## Parameters

| Parameter name | Input/Output/Attribute | Description | Data type | Data format |
|:---|:---|:---|:---|:---|
| grad | Input | Gradient, a 2D tensor [n, m]. | FLOAT16, FLOAT, BF16 | ND |
| sum_grad_r | Input | Row-wise sum of grad. The shape is [n]. | FLOAT | ND |
| sum_grad_c | Input | Column-wise sum of grad. The shape is [m]. | FLOAT | ND |
| sum_grad_rc | Input | Sum of all elements of grad. The shape is [1]. | FLOAT | ND |
| r | Input | Row second-moment estimate. The shape is [n]. Updated in place (`inplace_with(r)`). The dtype is the same as that of grad. | FLOAT16, FLOAT, BF16 | ND |
| c | Input | Column second-moment estimate. The shape is [m]. Updated in place (`inplace_with(c)`). The dtype is the same as that of grad. | FLOAT16, FLOAT, BF16 | ND |
| beta2 | Input | Second-moment decay coefficient, a scalar with shape [1]. | FLOAT | ND |
| sum_r | Optional input | Sum of all elements of r. The shape is [1]. If this parameter is absent, the operator internally computes $\sum r$. | FLOAT | ND |
| global_shape | Optional input | Original [n, m]. The shape is [2]. If this parameter is absent, the shape of grad is used. | INT64 | ND |
| u | Output | Normalized update direction. The shape is grad.shape, and the data type is forcibly FLOAT32 (for precision). | FLOAT | ND |
| sum_square_u | Output | Sum of squares of u. The shape is sum_grad_rc.shape, and the data type is FLOAT32. | FLOAT | ND |

## Constraints

- grad must be a 2D tensor (shape `[N, M]`), and r/c/sum_grad_r/sum_grad_c/sum_grad_rc must be 1D tensors.
- grad.shape[0] == r.shape[0] == sum_grad_r.shape[0] == N, and grad.shape[1] == c.shape[0] == sum_grad_c.shape[0] == M.
- grad, r, and c must have the same data type. FLOAT16, FLOAT32, and BFLOAT16 are supported.
- u and sum_square_u are forcibly output as FLOAT32. sum_grad_r/sum_grad_c/sum_grad_rc/beta2/sum_r are FLOAT32, and global_shape is INT64.
- r and c are updated in place, and the outputs share memory addresses with the inputs.
- sum_r is an optional input. If it is absent, the operator internally computes $\sum r$. global_shape is an optional input. If it is absent, the shape of grad is used.
- Empty tensors are supported (an empty result is returned when N=0 or M=0, with sum_square_u=0), and no error is reported.
- Broadcasting with size-1 dimensions (N=1 or M=1) is supported.
- When the gradient is all zeros, the output is 0, and sum_square_u is always non-negative.

## Invocation Description

| Invocation method | Sample code | Description |
|:---|:---|:---|
| GE graph mode | [test_geir_apply_came_part2](examples/test_geir_apply_came_part2.cpp) | Invoke the ApplyCamePart2 operator through graph construction using the [operator IR](op_graph/apply_came_part2_proto.h). |
