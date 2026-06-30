# aclnnQuantMatmulV5

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_batch_matmul_v4)

## Supported Products

| Product                                                        |  Supported  |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 training series products/Atlas A3 inference series products</term>    |    √    |
| <term>Atlas A2 training series products/Atlas A2 inference series products</term>|    √    |
| <term>Atlas 200I/500 A2 inference products</term>                     |    ×    |
| <term>Atlas inference series products</term>                            |    ×    |
| <term>Atlas training series products</term>                             |    ×    |

## Function

- Description: Performs matrix multiplication for quantization. It is compatible with the **aclnnQuantMatmulV3** and **aclnnQuantMatmulV4** APIs. It supports at least one-dimensional input and at most two-dimensional input. Similar APIs include **aclnnMm** (only two-dimensional tensors can be used as the input of matrix multiplication).
- Formula:
  - x1 is INT8, x2 is INT32, x1Scale is FLOAT32, x2Scale is UINT64, and yOffset is FLOAT32:

    $$
    out = ((x1 @ (x2*x2Scale)) + yOffset) * x1Scale
    $$

  - No x1Scale, no bias:

    $$
    out = x1@x2 * x2Scale + x2Offset
    $$

  - bias (INT32):
  
    $$
    out = (x1@x2 + bias) * x2Scale + x2Offset
    $$

  - bias (BFLOAT16/FLOAT32) (no offset in this scenario):

    $$
    out = x1@x2 * x2Scale + bias
    $$

  - With x1Scale, no bias:

    $$
    out = x1@x2 * x2Scale * x1Scale
    $$

  - With x1Scale, bias (INT32) (no offset in this scenario):

    $$
    out = (x1@x2 + bias) * x2Scale * x1Scale
    $$

  - With x1Scale, bias BFLOAT16/FLOAT16/FLOAT32 (no offset in this scenario):

    $$
    out = x1@x2 * x2Scale * x1Scale + bias
    $$
  
  - x1 and x2 are INT8, x1Scale and x2Scale are FLOAT32, bias is FLOAT32, and out is FLOAT16 or BFLOAT16 (pergroup-perblock quantization):

    $$
    out = (x1 @ x2) * x1Scale * x2Scale + bias
    $$

  - x1 and x2 are INT4, x1Scale and x2Scale are FLOAT32, x2Offset is FLOAT16, and out is FLOAT16 or BFLOAT16 (pertoken-pergroup asymmetric quantization):
  
    $$
    out = x1Scale * x2Scale @ (x1 @ x2 - x1 @ x2Offset)
    $$

## Prototype

Each operator has [two-phase API](../../../docs/en/context/two_phase_api.md) calls. First, **aclnnQuantMatmulV5GetWorkspaceSize** is called to obtain the workspace size required for computation and the executor that contains the operator computation process. Then, **aclnnQuantMatmulV5** is called to perform computation.

```c++
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *x1Scale,
  const aclTensor *x2Scale,
  const aclTensor *yScale,
  const aclTensor *x1Offset,
  const aclTensor *x2Offset,
  const aclTensor *yOffset,
  const aclTensor *bias,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```
```c++
aclnnStatus aclnnQuantMatmulV5(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnQuantMatmulV5GetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1554px"><colgroup>
  <col style="width: 198px">
  <col style="width: 121px">
  <col style="width: 220px">
  <col style="width: 397px">
  <col style="width: 220px">
  <col style="width: 115px">
  <col style="width: 138px">
  <col style="width: 145px">
  </colgroup>
    <thead>
      <tr>
        <th>Name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage Notes</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimension (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>x1</td>
        <td>Input</td>
        <td>Input x1 in the formula.</td>
        <td>
          <ul>
            <li><a href="../../../docs/en/context/non_contiguous_tensors.md">Non-contiguous tensors</a> are supported only when the last m and k axes are transposed. In other scenarios, non-contiguous tensors are not supported.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-6</td>
        <td>×</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>Input</td>
        <td>Input x2 in the formula.</td>
        <td>
          <ul>
            <li>In AI processor affinity data layout format, the shape can be four- to eight-dimensional.</li>
            <li>In ND format, non-contiguous tensors are supported when the last two axes are transposed. In other scenarios, <a href="../../../docs/en/context/non_contiguous_tensors.md">non-contiguous tensors</a> are not supported.</li>
          </ul>
        </td>
        <td>INT4, INT8, INT32</td>
        <td>ND</td>
        <td>2-8</td>
        <td>x</td>
      </tr>
      <tr>
        <td>x1Scale</td>
        <td>Input</td>
        <td>Input x1Scale in the formula.</td>
        <td>-</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1-6</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Scale</td>
        <td>Input</td>
        <td>Quantization parameter, corresponding to the input x2Scale in the formula.</td>
        <td>-</td>
        <td>UINT64, INT64, FLOAT32, BFLOAT16</td>
        <td>ND</td>
        <td>1-6</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yScale</td>
        <td>Input</td>
        <td>Dequantization scale parameter of the output y.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x1Offset</td>
        <td>Input</td>
        <td>Input x1Offset in the formula.</td>
        <td>Reserved parameter. It is not supported in the current version. Pass nullptr.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>x2Offset</td>
        <td>Input</td>
        <td>Input x2Offset in the formula.</td>
        <td>The shape is one-dimensional (t,), and t = 1 or n, where n is the same as n of x2.</td>
        <td>FLOAT16, FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>-</td>
      </tr>
      <tr>
        <td>yOffset</td>
        <td>Input</td>
        <td>Input yOffset in the formula.</td>
        <td>The shape can be one-dimensional (n). The value must be 8\*x2\*x2Scale and be accumulated in the first dimension.</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>-</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>Input</td>
        <td>Input bias in the formula.</td>
        <td>-</td>
        <td>INT32, FLOAT32, BFLOAT16, FLOAT16</td>
        <td>ND</td>
        <td>1-3</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX1</td>
        <td>Input</td>
        <td>Whether the input shape of x1 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2</td>
        <td>Input</td>
        <td>Whether the input shape of x2 is transposed.</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize</td>
        <td>Input</td>
        <td>Quantization group size in the m, n, and k directions.</td>
        <td> The value is composed of groupSizeM, groupSizeN, and groupSizeK in three directions. Each value occupies 16 bits, and the lower 48 bits of groupSize of the int64_t type are occupied (the higher 16 bits of groupSize are invalid). Formula 1 below the table is used to compute the value.</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>Output</td>
        <td>out in the formula.</td>
        <td>-</td>
        <td>FLOAT16, INT8, BFLOAT16, INT32, HIFLOAT8, FLOAT8_E4M3FN</td>
        <td>ND</td>
        <td>2</td>
        <td>✓</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Size of the workspace required to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Operator executor, containing the operator computation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>
 
  - Formula 1:

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

  - <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:

    The size of the last dimension of **x1** or **x2** cannot exceed 65535. The last dimension of **x1** refers to **m** when **transposeX1** is **true** or **k** when **transposeX1** is **false**. The last dimension of **x2** refers to **k** when **transposeX2** is **true** or **n** when **transposeX2** is **false**.

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The first-phase API implements input parameter verification. The following errors may be thrown:
  
  <table style="undefined;table-layout: fixed;width: 1202px"><colgroup>
  <col style="width: 262px">
  <col style="width: 121px">
  <col style="width: 819px">
  </colgroup>
  <thead>
      <tr>
        <th>Return</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>The passed x1, x2, x1Scale, x2Scale, yOffset, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="5">161002</td>
        <td>The shape of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out does not meet the verification condition.</td>
      </tr>
      <tr>
        <td>The data type or format of x1, x2, bias, x1Scale, x2Scale, x2Offset, or out is not supported.</td>
      </tr>
      <tr>
        <td>x1, x2, bias, x2Scale, x2Offset, or out is an empty tensor.</td>
      </tr>
      <tr>
        <td>The passed groupSize does not meet the verification conditions or the passed groupSize is 0, and the groupSize cannot be deduced based on the shape relationship between x1, x2, x1Scale, and x2Scale.</td>
      </tr>
    </tbody></table>


## aclnnQuantMatmulV5

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 153px">
    <col style="width: 121px">
    <col style="width: 880px">
    </colgroup>
    <thead>
      <tr>
      <th>Name</th>
      <th>Input/Output</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Address of the workspace to be allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Size of the workspace to be allocated on the device, which is obtained by first-phase API aclnnQuantMatmulV5GetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Operator executor, containing the operator computation process.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody></table>

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).

## Constraints

- Determinism description: **aclnnQuantMatmulV5** defaults to a deterministic implementation.

- The input and output support the following data type combinations:

  | x1                        | x2                        | x1Scale     | x2Scale         | x2Offset    | yScale   | bias         | yOffset    | out                                    |
  | ------------------------- | ------------------------- | ----------- | -----------     | ----------- | -------  | ------------ | -----------| -------------------------------------- |
  | INT8                      | INT32                     | FLOAT32     | UINT64          | null        | null     | null         | FLOAT32    | FLOAT16/BFLOAT16                       |
  | INT8                      | INT8                      | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                |
  | INT8                      | INT8                      | null        | UINT64/INT64    | null/FLOAT32| null     | null/INT32   | null       | INT8                                   |
  | INT8                      | INT8                      | null/FLOAT32| FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT8                      | INT8                      | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4/INT32                | INT4/INT32                | null        | UINT64/INT64    | null        | null     | null/INT32   | null       | FLOAT16                                  |
  | INT8                      | INT8                      | null        | FLOAT32/BFLOAT16| null        | null     | null/INT32   | null       | INT32                                |
  | INT8                      | INT8                      | FLOAT32        | FLOAT32| null        | null     | FLOAT32   | null       | BFLOAT16                                |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32/BFLOAT16| null        | null     | null/INT32/BFLOAT16/FLOAT32   | null       | BFLOAT16              |
  | INT4/INT32                | INT4/INT32                | FLOAT32     | FLOAT32         | null        | null     | null/INT32/FLOAT16/FLOAT32    | null       | FLOAT16               |
  | INT4                | INT4                | FLOAT32     | FLOAT32         | FLOAT16        | null     | null    | null       | BFLOAT16               |


- The input dtype combinations of x1, x2, x1Scale, and x2Scale and the supported platforms in different quantization modes are as follows:

  The dtype and shape values of x1, x2, x1Scale, x2Scale, yOffset, and groupSize affect each other in different quantization scenarios. The relationships are as follows:

    | x1 Data Type                | x2 Data Type                | x1Scale Data Type| x2Scale Data Type| x1 shape | x2 shape| x1Scale shape| x2Scale shape| yOffset shape| Values of [groupSizeM, groupSizeN, groupSizeK]|
    | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------------ | ------------ | ------------ |
    | INT8                    |INT32                   |FLOAT32              |UINT64             | (m, k)|(k, n // 8)|(m, 1)|((k // 256), n)| (n) | [0, 0, 256]|
    | INT8                    |INT8                   |FLOAT32              |FLOAT32             | (m, k)|(n, k)|(m, k // 128)|((k // 128), (n // 128))| null | [1, 128, 128]|
    | INT4                    |INT4                   |FLOAT32              |FLOAT32             | (m, k)|(n, k)|(m, 1)| (k // 256, n) | null | [0, 0, 256]|

- The input combinations that belong to APIs **aclnnQuantMatmulV3** and **aclnnQuantMatmulV4** are not listed in the table. These two APIs do not support the input **groupsize**, and the default value of **groupsize** is **0**.

  - The constraints for **x1** are as follows:
    - The data type can be INT8, INT32, or INT4.
    - When the data type is INT32 or INT4, the INT4 quantization scenario is used, and **transposeX1** is set to **false**.
    - When the data type is INT4, the shape is (batch, m, k), where **k** must be an even number.
    - When the data type is INT32, each INT32 data entry stores eight INT4 data entries, with shape (batch, m, k // 8), where **k** must be a multiple of 8.
    - When A8W8 perblock symmetric quantization is used, the data type can be INT8. Currently, **n** must be a multiple of 256, **k** must be a multiple of both 128 and 4 × 128, **transposeX1** is set to **false**, and the shape is (m, k).
    - When A4W4 pergroup asymmetric quantization is used, the data type can be INT4. Currently, **k** must be a multiple of 1024, **transposeX1** is set to **false**, and the shape is (m, k).
    - When **transposeX1** is set to **false**, the shape is (batch, m, k).
    - When **transposeX1** is set to **true**, the shape is (batch, k, m), where **batch** represents the first one to four dimensions, and dimension 0 indicates that **batch** does not exist.
    - In AI processor affinity data layout format, the shape can be four- to eight-dimensional.
    - When **transposeX2** is **true**, the shape is (batch, k1, n1, n0, k0), where **batch** is optional, k0 = 32, and n0 = 16. **k** in the shape of **x1** and **k1** in the shape of **x2** must meet the following relationship: ceil (k/32) = k1.
    - When **transposeX2** is **false**, the shape is (batch, n1, k1, k0, n0), where **batch** is optional, k0 = 16, and n0 = 32. **k** in the shape of **x1** and **k1** in the shape of **x2** must meet the following relationship: ceil(k/16) = k1.
    - **aclnnCalculateMatmulWeightSizeV2** and **aclnnTransMatmulWeight** can be used to convert the input from ND format to AI processor affinity data layout format.
  
  - The constraints for **x2** are as follows:
    - The data type can be INT8, INT32, or INT4.
    - When the data type is INT32 or INT4, the INT4 quantization scenario is used. Currently, only the two-dimensional ND format is supported. In ND format, the shape can be two- to six-dimensional.
      - When **transposeX2** is set to **false**, the shape is (batch, k, n).
      - When **transposeX2** is set to **true**, the shape is (batch, n, k).
      - **batch** is optional and **k** is the same as that in shape of **x1**.
    - When the data type is INT4:
      - If **transposeX2** is set to **true**, the shape is (n, k), where **k** must be an even number.
      - If **transposeX2** is set to **false**, the shape is (k, n), where **n** must be an even number.
    - When the data type is INT32, each INT32 data entry stores eight INT4 data entries.
      - If **transposeX2** is set to **true**, the shape is (n, k // 8), where **k** must be a multiple of 8.
      - If **transposeX2** is set to **false**, the shape is (k, n // 8), where **n** must be a multiple of 8.
    - The **aclnnConvertWeightToINT4Pack** API can be used to convert **x2** from INT32 (one int32 space stores one int4 data entry in bits 0–3) to INT32 (one int32 space stores eight int4 data entries) or INT4 (one int4 space stores one int4 data entry). For details, see [aclnnConvertWeightToINT4Pack](../../convert_weight_to_int4_pack/docs/aclnnConvertWeightToINT4Pack_en.md).
    - When A8W8 perblock symmetric quantization is used, the data type can be INT8, **transposeX2** is set to **true**, and the shape is (n, k). Currently, **n** must be a multiple of 256, and **k** must be a multiple of both 128 and 4 × 128.
    - When A4W4 pergroup asymmetric quantization is used, the data type can be INT4, **transposeX2** is set to **true**, and the shape is (n, k). Currently, **k** must be symmetric with 1024, and **n** must be a multiple of 256.

  - The constraints for **x1Scale** are as follows:
    - The shape is two-dimensional and is represented by (m, 1). The data type can be FLOAT32.
    - When A8W8 perblock symmetric quantization is used, the data type is FLOAT32 and the shape is (m, ceil(k/128)).
    - When A4W4 pergroup asymmetric quantization is used, the data type is FLOAT32 and the shape is (m, 1).
  - The constraints for **x2Scale** are as follows:
    - The shape is two-dimensional and is represented by (k / groupSize, n), where **n** is the same as that of **x2**.
    - The data type can be UINT64, INT64, FLOAT32, or BFLOAT16.
    - **TransQuantParamV2** can only be one-dimensional. Therefore, if the original input type does not conform with the data type combinations described in [Constraints](#constraints), you need to convert x2_scale view to one-dimensional (k / groupSize * n), call the **aclnn** API of the TransQuantParamV2 operator to convert **x2Scale** to the UINT64 data type, and then convert the output view to two-dimensional (k / groupSize, n).
    - When A8W8 perblock symmetric quantization is used, the data type is FLOAT32. If **transpose** of **x2** is **true**, the shape is (ceil(n/128), ceil(k/128)). If **transpose** of **x2** is **false**, the shape is (ceil(k/128), ceil(n/128)).
    - When A4W4 pergroup asymmetric quantization is used, the data type is FLOAT32. If **transpose** of **x2** is **false**, the shape is (ceil(k / 256), n).
  - **yScale** is not supported in the current version. Pass **nullptr**.
  - The constraints for **x2Offset** are as follows:
    - Optional quantization parameter, the data type can be FLOAT32.
    - If the output data type is INT8, the offset can exist. For other input types, nullptr needs to be passed.
    - When A4W4 pergroup asymmetric quantization is used, the input type can be FLOAT16 and the shape is (ceil(k, 256), n).
  - The constraints for **bias** are as follows:
    - When A8W8 perblock symmetric quantization is used, the data type can be FLOAT32 and the shape can be one-dimensional (n,).
    - A4W4 pergroup is not supported in the current version. Pass **nullptr**.
  - **transposeX1**: When **x1** and **x2** are INT32 or INT4, **transposeX1** can only be **false** and the shape is represented by (m, k).
  - The constraints for **transposeX2** are as follows:
    - In ND format, if this parameter is **false**, the shape is (batch, k, n); if this parameter is **true**, the shape is (batch, n, k), where **batch** is optional and **k** is the same as that in the shape of **x1**.
    - In AI processor affinity data layout format:
      - If this parameter is **true**, the shape is (batch, k1, n1, n0, k0), where **batch** is optional, k0 = 32, and n0 = 16. **k** in the shape of **x1** and **k1** in the shape of **x2** must meet the following relationship: ceil (k/32) = k1.
      - If this parameter is **false**, the shape is (batch, n1, k1, k0, n0), where **batch** is optional, k0 = 16, and n0 = 32. **k** in the shape of **x1** and **k1** in the shape of **x2** must meet the following relationship: ceil(k/16) = k1.
  - The constraints for **groupSize** are as follows:
    - In common cases and A4W4 pergroup asymmetric quantization mode, the value combination of [groupSizeM, groupSizeN, groupSizeK] can only be [0, 0, 256]. That is, **groupSizeK** can only be **256**.
    - In A8W8 perblock symmetric quantization mode, the value combination of [groupSizeM, groupSizeN, groupSizeK] can only be [1, 128, 128].
  - The constraints for **out** are as follows:
    - The shape is two-dimensional and is represented by (m, n). The data type can be FLOAT16, INT8, BFLOAT16, or INT32.
    - In A8W8 perblock symmetric quantization mode, the output supports BFLOAT16.
    - In A4W4 pergroup asymmetric quantization mode, the output supports BFLOAT16.


## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

**x1** and **x2** are FLOAT8_E4M3FN, **x1Scale** and **x2Scale** are FLOAT32, **x2Offset** is not available, and **bias** is FLOAT32.

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

      int64_t
      GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // (Fixed writing) Initialize resources.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {16, 16};
      std::vector<int64_t> x2Shape = {16, 16};
      std::vector<int64_t> biasShape = {16};
      std::vector<int64_t> x2OffsetShape = {16};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> outShape = {16, 16};
      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x2OffsetDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *biasDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *bias = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(256, 1);
      std::vector<int8_t> x2HostData(256, 1);
      std::vector<int32_t> biasHostData(16, 1);
      std::vector<float> x2ScaleHostData(1, 1);
      std::vector<float> x2OffsetHostData(16, 1);
      std::vector<float> x1ScaleHostData(1, 1);
      std::vector<uint16_t> outHostData(256, 1);  // Half-precision float16

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr,
                            aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale,
                                                                                            aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a bias aclTensor.
      ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;
      // Call the first-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, x2Offset, nullptr, bias,
                                               transposeX1, transposeX2, 0, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(
          size, 0);  // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                        size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

- <term>Atlas A2 training series products/Atlas A2 inference series products</term> and <term>Atlas A3 training series products/Atlas A3 inference series products</term>:
**x1** is INT8, **x2** is INT32, **x1Scale** is FLOAT32, and **x2Scale** is UINT64

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
      // (Fixed writing) Initialize resources.
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      // Call aclrtMalloc to allocate memory on the device.
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      // Call aclrtMemcpy to copy the data on the host to the memory on the device.
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      // Compute the strides of the contiguous tensor.
      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      // Call aclCreateTensor to create an aclTensor.
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. Construct the input and output based on the API definition.
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), ans);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // Half-precision float16

      // Create an x1 aclTensor.
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2 aclTensor.
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x1Scale aclTensor.
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an x2Scale aclTensor.
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create a yoffset aclTensor.
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // Create an out aclTensor.
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. Call the CANN operator library API, which needs to be replaced with the actual API.
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // Allocate device memory based on workspaceSize computed by the first-phase API.
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // Call the second-phase API of aclnnQuantMatmulV5.
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. (Fixed writing) Wait until the task execution is complete.
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. Obtain the output value and copy the result from the device memory to the host. Modify the configuration based on the API definition.
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0); // The fp16 data cannot be directly printed in the C language. The data needs to be read by using uint16 and converted into fp16 in binary mode.
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. (Fixed writing) Initialize the device and stream. For details, see the ACL API manual.
      // Set the device ID in use.
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```
