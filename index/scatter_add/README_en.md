# ScatterAdd

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    √    |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

- Operator function: fills the values in the src tensor into the self tensor one by one according to the specified axis direction and the position relationship in the index tensor. If more than one src value is filled into the same position of self, these values are accumulated at that position.
  For a 3D tensor, self is updated according to the following rules:

  ```bash
  self[index[i][j][k]][j][k] += src[i][j][k] # If dim == 0
  self[i][index[i][j][k]][k] += src[i][j][k] # If dim == 1
  self[i][j][index[i][j][k]] += src[i][j][k] # If dim == 2
  ```

  The following requirements must be satisfied during computation:
  - The numbers of dimensions of self, index, and src must be the same.
  - For every dimension d, index.size(d) <= src.size(d)
  - For every dimension d, if d != dim, index.size(d) <= self.size(d)
  - The value range of dim is [-self.dim(), self.dim() - 1]
- Example:

  Input tensor $self = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [7&8&9] \end{bmatrix}$,
  index tensor $index = \begin{bmatrix} [0&2&1] \\ [0&0&1] \end{bmatrix}$, dim = 1,
  source tensor $src = \begin{bmatrix} [10&11&12] \\ [13&14&15] \end{bmatrix}$,
  output tensor $output = \begin{bmatrix} [11&14&14] \\ [31&20&6] \\ [7&8&9] \end{bmatrix}$

  dim = 1 indicates that scatter_add performs accumulation on the columns of the tensor according to $index$.

  $output[0][0] = self[0][0] + src[0][0]$ = 1 + 10,

  $output[0][1] = self[0][1] + src[0][2]$ = 2 + 12,

  $output[0][2] = self[0][2] + src[0][1]$ = 3 + 11,

  $output[1][0] = self[1][0] + src[1][0] + src[1][1]$ = 4 + 13 + 14,

  $output[1][1] = self[1][1] + src[1][2]$ = 5 + 15,

  $output[1][2] = self[1][2]$ = 6,

  $output[2][0] = self[2][0]$ = 7,

  $output[2][1] = self[2][1]$ = 8,

  $output[2][2] = self[2][2]$ = 9.

  Here, the numbers of dimensions of $self$, $index$, and $src$ are all 2. The size of each dimension of $index$, {2, 3}, is not greater than the size of the corresponding dimension of $src$, {2, 3}. On the dimension where dim != 1 (dim = 0), the dimension size of $index$, {2}, is not greater than the size of the corresponding dimension of $self$, {3}. The maximum value in $index$, {2}, is less than the size of $self$ along the dim = 1 dimension, {3}.

  ## Parameters

    - self (aclTensor*, computation input): input `self` in the formula, an aclTensor on the Device. It is the target tensor of scatter. The shape supports 0-8 dimensions, and the number of dimensions must be the same as those of index and src. The data type is the same as that of src. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the supported [data format](../../docs/en/context/data_format.md) is ND.
      - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, and COMPLEX128 are supported.
    - dim (int64_t, computation input): input `dim` in the formula, with the data type INT64.

    - index (aclTensor*, computation input): input `index` in the formula, an aclTensor on the Device. The data types INT32 and INT64 are supported. The number of dimensions of index must be the same as that of src. [Non-contiguous tensors](../../docs/en/context/non_contiguous_tensor.md) are supported, and the supported [data format](../../docs/en/context/data_format.md) is ND.
      - <term>Ascend 950PR&950DT products</term>: if duplicate index values exist on the dim axis, the result is non-deterministic. If deterministic computation is enabled, the determinism of the result is guaranteed.
    - src (aclTensor*, computation input): input `src` in the formula, an aclTensor on the Device. It is the source tensor. The number of dimensions of src must be the same as that of index. The data type is the same as that of self. The supported [data format](../../docs/en/context/data_format.md) is ND.
      - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, and COMPLEX128 are supported.
    - out (aclTensor*, computation output): `output` in the formula, an aclTensor on the Device. The shape must be the same as that of self. The data type is the same as that of self. The supported [data format](../../docs/en/context/data_format.md) is ND.
      - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>, and <term>Ascend 950PR&950DT products</term>: the data types BFLOAT16, FLOAT16, FLOAT32, DOUBLE, INT64, INT32, INT16, INT8, UINT8, BOOL, COMPLEX64, and COMPLEX128 are supported.

## Constraints

None

## Invocation Description

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn interface | [test_aclnn_scatter_add](examples/test_aclnn_scatter_add.cpp) | Invokes the ScatterAdd operator through the [aclnnScatterAdd](docs/aclnnScatterAdd_en.md) interface. |
