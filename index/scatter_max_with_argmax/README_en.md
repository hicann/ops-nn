# ScatterMaxWithArgmax

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                     |     √    |
| <term>Atlas A3 products</term>   |    √     |
| <term>Atlas A2 products</term>    |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                               |    ×     |
| <term>Atlas training products</term>                               |    ×     |

## Function

- Operator function: Performs a scatter max operation along dimension 0, and records which update (argmax index) the maximum value at each position comes from.

- Formula:

Given:

```bash
  x:       [N, D1, D2, ...] float32   # Used to determine the output shape, does not participate in the computation
  indices: [M] int32                   # Scatter indexes, 0 <= indices[i] < N
  updates: [M, D1, D2, ...] float32   # Scatter source data

The computation is divided into three stages:

  # Stage 1: Initialization
  y      = full(x.shape, negative_large_value)  # Initialized to a large negative float32 value, serving only as an internal sentinel
  argmax = full(x.shape, M)                     # Initialized to M, indicating "no update"

  # Stage 2: Scatter max
  for i in range(M):
      idx = indices[i]
      for each element position p in (D1, D2, ...):
          if updates[i][p] > y[idx][p]:        # Strictly greater than (GT comparison)
              y[idx][p]      = updates[i][p]
              argmax[idx][p] = i

  # Stage 3: FillZeroForUncovered
  for each output position q:
      if argmax[q] == M:
          y[q] = 0
```

Outputs:

```bash
  y:      [N, D1, D2, ...] float32   # Scatter max result. Positions not covered by any index are 0
  argmax: [N, D1, D2, ...] int32     # Index of the source of the maximum value (0~M-1). Positions not pointed to are M
```

## Parameters

<table style="undefined;table-layout: fixed; width: 980px"><colgroup>
  <col style="width: 100px">
  <col style="width: 150px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 120px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter name</th>
      <th>Input/Output/Attribute</th>
      <th>Description</th>
      <th>Data type</th>
      <th>Data format</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x</td>
      <td>Input</td>
      <td>Reference tensor for the output shape. It is used only to determine the output shape and does not participate in the computation. N is the size of dimension 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>indices</td>
      <td>Input</td>
      <td>Scatter index tensor, 1D with shape=[M]. Value range: [0, N). M is the number of indexes.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>updates</td>
      <td>Input</td>
      <td>Scatter source data tensor, with shape[0] = M and shape[1:] the same as x.shape[1:].</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>y</td>
      <td>Output</td>
      <td>Scatter max result, with the same shape as x. Positions not covered by any index finally output 0.</td>
      <td>FLOAT</td>
      <td>ND</td>
    </tr>
    <tr>
      <td>argmax</td>
      <td>Output</td>
      <td>Index of the source of the maximum value (0~M-1), with the same shape as x. Positions not pointed to are M.</td>
      <td>INT32</td>
      <td>ND</td>
    </tr>
  </tbody></table>

## Constraints

- The input x supports 1D to 8D.
- `indices` must be 1D, with shape = [M].
- `updates.shape[0]` must equal `indices.shape[0]` (M is consistent).
- `updates.shape[1:]` must equal `x.shape[1:]` (all dimensions except dimension 0 are consistent).
- `x.shape[0]` must be >= max(indices) (indexes do not go out of bounds).
- Only the float32/int32 data type combination is supported.
- Determinism guarantee: the same input produces the same output (the GT strictly-greater-than comparison is used, and the earliest index is retained when values are equal).
- For positions not covered by any index, y finally outputs 0 and argmax outputs M. The internal large negative initialization value is rewritten to 0 in the FillZeroForUncovered stage.

## Invocation Description

| Invocation method | Invocation sample                                                                   | Description                                                           |
|--------------|------------------------------------------------------------------------|--------------------------------------------------------------|
| Graph-mode invocation | [test_geir_scatter_max_with_argmax](examples/arch35/test_geir_scatter_max_with_argmax.cpp) | Calls the ScatterMaxWithArgmax operator through graph construction with the [operator IR](./op_graph/scatter_max_with_argmax_proto.h). |
