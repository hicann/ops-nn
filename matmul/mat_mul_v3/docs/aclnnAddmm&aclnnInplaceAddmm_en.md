# aclnnAddmm&aclnnInplaceAddmm

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T18:13:33.725Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not Supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Supported
<!-- end id6 -->

## Function

- Description: Computes α multiplied by the product of mat1 and mat2, then adds the product of β and self.
- Formula:

  $$
  out = β  self + α  (mat1 @ mat2)
  $$

- Example:
  * For the aclnnAddmm API, the shape of self is [1, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the matrix multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For the aclnnAddmm API, the shape of self is [m, 1], the shape of mat1 is [m, k], the shape of mat2 is [k, n], the shape of the matrix multiplication result of mat1 and mat2 is [m, n], and the shape of self can be broadcast to [m, n].
  * For the aclnnAddmm API, the shape of self is [m, n], the shape of mat1 is [m, k], the shape of mat2 is [k, n], and the shape of the matrix multiplication result of mat1 and mat2 is [m, n].
  * For the aclnnInplaceAddmm API, the computation result is directly stored in the memory of the input tensor selfRef. The shape of selfRef is [m, n], the shape of mat1 is [m, k], and the shape of mat2 is [k, n].

## Function Prototype

- aclnnAddmm and aclnnInplaceAddmm implement the same functionality. The differences are as follows. Select the appropriate operator based on your actual scenario.
  - aclnnAddmm: Requires creating a new output tensor object to store the computation result.
  - aclnnInplaceAddmm: Does not require creating a new output tensor object. The computation result is stored directly in the memory of the input tensor.
- Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). You must first call the "aclnnAddmmGetWorkspaceSize" or "aclnnInplaceAddmmGetWorkspaceSize" API to obtain the input parameters and calculate the required workspace size based on the computation flow. Then call the "aclnnAddmm" or "aclnnInplaceAddmm" API to perform the computation.

```cpp
aclnnStatus aclnnAddmmGetWorkspaceSize(
  const aclTensor *self,
  const aclTensor *mat1,
  const aclTensor *mat2,
  const aclScalar *beta,
  const aclScalar *alpha,
  aclTensor       *out,
  int8_t          cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnAddmm(
  void           *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream    stream)
```

```cpp
aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(
  const aclTensor *selfRef,
  const aclTensor *mat1,
  const aclTensor *mat2,
  const aclScalar *beta,
  const aclScalar *alpha,
  int8_t          cubeMathType,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnInplaceAddmm(
  void           *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream    stream)
```

## aclnnAddmmGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1563px"><colgroup>
  <col style="width: 153px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 437px">
  <col style="width: 203px">
  <col style="width: 120px">
  <col style="width: 149px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td rowspan="2">self</td>
      <td rowspan="2">Input</td>
      <td rowspan="2">Represents the bias matrix (self in the formula).</td>
      <td>The data type must satisfy the data type deduction rule with mat1@mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td rowspan="2">BFLOAT16, FLOAT16, FLOAT32</td>
      <td rowspan="2">ND</td>
      <td rowspan="2">1-2</td>
      <td rowspan="2">√</td>
    </tr>
    <tr><td>Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with mat1@mat2.</td></tr>
    <tr>
      <td rowspan="4">mat1</td>
      <td rowspan="4">Input</td>
      <td rowspan="4">Denotes the first matrix for matrix multiplication, corresponding to mat1 in the formula.</td>
      <td>The data type must satisfy the data type inference rules with mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td rowspan="4">BFLOAT16, FLOAT16, FLOAT32</td>
      <td rowspan="4">ND</td>
      <td rowspan="4">2</td>
      <td rowspan="4">√</td>
    </tr>
      <tr><td>Must satisfy the broadcast relationship with self and mat2, as described in <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</td></tr>
      <tr><td>When mat1 is not transposed, the dimensions denote: (m, k).</td></tr>
      <tr><td>When mat1 is transposed, the dimensions denote: (k, m).</td></tr>
    <tr>
      <td rowspan="5">mat2</td>
      <td rowspan="5">Input</td>
      <td rowspan="5">Denotes the second matrix for Matrix Multiplication, i.e., mat2 in the formula.</td>
      <td>The data type must satisfy the data type inference rule with mat1 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td rowspan="5">BFLOAT16, FLOAT16, FLOAT32</td>
      <td rowspan="5">ND</td>
      <td rowspan="5">2</td>
      <td rowspan="5">√</td>
    </tr>
      <tr><td>The Reduce dimension of mat2 must have the same Dimension Size as that of mat1.</td></tr>
      <tr><td>Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with self and mat1.</td></tr>
      <tr><td>When mat2 is not transposed, the dimensions are (k, n).</td></tr>
      <tr><td>In the case of a transposed mat2, the dimensions represent: (n, k).</td></tr>
    <tr>
      <td>beta(β)</td>
      <td>Input</td>
      <td>Represents β in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha(α)</td>
      <td>Input</td>
      <td>Represents α in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>Output</td>
      <td>Represents the output matrix of the Addmm computation, that is, out in the formula, which is the result of β·self+α·(mat1@mat2).</td>
      <td>The data type must be consistent with the data types of self and mat2 after type inference (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a deduction relationship, this parameter is applied to the data type after deduction by default. The supported enumeration values are as follows:<ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports precision reduction of the input data for computation.</li>
        <li>2: USE_FP16, supports precision reduction of the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports precision reduction of the input to data type HFLOAT32 for computation.</li>
        <li>4: USE_FP32_ADD, supports high-precision computation.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device side.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the computation flow of the operator.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id7 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 and 4 are not supported.
  <!-- end id7 -->
  <!-- npu="A3,910b" id8 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: when self, mat1, and mat2 are all FLOAT16 or all BFLOAT16, the matrix multiplication and bias addition process uses FLOAT32 intermediate results for computation, self supports broadcast relative to the matrix multiplication result, and the output data type is specified by out. When the input data type is FLOAT32 and the k axis is greater than 2048, grouped accumulation is used for computation.
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: when the input data type is FLOAT16/BFLOAT16, the addmm process uses higher-precision computation. In this case, broadcast between the input self and the matmul computation result matrix is currently Not Supported. When the input data type is FLOAT32 and any of the following conditions is met, the vector core is used for computation to improve the precision of the computation result (this method may degrade operator performance in some use cases). Condition 1: the m axis equals 1 or the n axis equals 1. Condition 2: alpha does not equal 1 or beta does not equal 1. Condition 3: the shape of self is \(m,n\).

  <!-- end id9 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 283px">
    <col style="width: 120px">
    <col style="width: 747px">
    </colgroup>
    <thead>
      <tr>
        <th>Return Value</th>
        <th>Error Code</th>
        <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>self, mat1, mat2, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>The data type of self, mat1, mat2, or out is not within the supported range, or the data format of self is FRACTAL_NZ.</td>
      </tr>
      <tr>
        <td>mat1 and mat2 do not meet the multiplication condition.</td>
      </tr>
      <tr>
        <td>The shape of out is inconsistent with mat1@mat2.</td>
      </tr>
    </tbody>
    </table>

## aclnnAddmm

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the device, obtained by the first-phase API aclnnAddmmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Executor of the op, containing the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceAddmmGetWorkspaceSize

- **Parameters**
  <table style="undefined;table-layout: fixed; width: 1563px"><colgroup>
  <col style="width: 153px">
  <col style="width: 123px">
  <col style="width: 232px">
  <col style="width: 437px">
  <col style="width: 203px">
  <col style="width: 120px">
  <col style="width: 149px">
  <col style="width: 146px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
      <th>Instruction</th>
      <th>Data Type</th>
      <th>Data Format</th>
      <th>Dimension (shape)</th>
      <th>Non-contiguous Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>selfRef</td>
      <td>Input</td>
      <td>Refers to the input self and out in the formula.</td>
      <td><ul><li>The data type must satisfy the data type deduction rules with mat2 (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li>
      <li>The shape of self must be the same as that of mat1@mat2.</li></ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat1</td>
      <td>Input</td>
      <td>Denotes the first matrix of the matrix multiplication, mat1 in the formula.</td>
      <td><ul><li>The data type must satisfy the data type inference rules with selfRef (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The Reduce dimension of mat1 must have the same dimension size as the Reduce dimension of mat2.</li><li>Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with mat2.</li> </ul></td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>Input</td>
      <td>Denotes the second matrix of matrix multiplication, i.e., mat2 in the formula.</td>
      <td><ul><li>The data type must satisfy the data type inference rules with selfRef (see <a href="../../../docs/en/context/deduction_relationship.md">deduction relationship</a> and <a href="#constraints">Constraints</a>).</li><li>The Reduce dimension of mat2 must have the same dimension size as the Reduce dimension of mat1.</li><li>Must satisfy the <a href="../../../docs/en/context/broadcast_relationship.md">broadcast relationship</a> with mat1.</li> </ul>
      </td>
      <td>BFLOAT16, FLOAT16, FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta(β)</td>
      <td>Input</td>
      <td>Represents β in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>alpha(α)</td>
      <td>Input</td>
      <td>Represents α in the formula.</td>
      <td>-</td>
      <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>Input</td>
      <td>Specifies the computation logic of the Cube unit.</td>
      <td>If the input data types have a mutual inference relationship, this parameter processes the data type after mutual inference by default. Supported enumeration values are as follows: <ul>
        <li>0: KEEP_DTYPE, keeps the input data type for computation.</li>
        <li>1: ALLOW_FP32_DOWN_PRECISION, supports precision reduction of the input data for computation.</li>
        <li>2: USE_FP16, supports precision reduction of the input to FLOAT16 for computation.</li>
        <li>3: USE_HF32, supports precision reduction of the input to the HFLOAT32 data type for computation.</li>
        <li>4: USE_FP32_ADD, supports computation in high-precision mode.</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Output</td>
      <td>Returns the workspace size to be requested on the Device.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Output</td>
      <td>Returns the op executor, which contains the operator computation process.</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  <!-- npu="910,310p" id10 -->
  - <term>Atlas training products</term> and <term>Atlas inference products</term>:
    - BFLOAT16 data type is not supported.
    - When the input data type is FLOAT32, cubeMathType=0 is not supported.
    - For cubeMathType=1, when the input data type is FLOAT32, it is converted to FLOAT16 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=3 and 4 are not supported.
  <!-- end id10 -->
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A2 products</term>, <term>Atlas A3 products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: When selfRef, mat1, and mat2 are all FLOAT16 or all BFLOAT16, the matrix multiplication and bias addition process uses the FLOAT32 intermediate result for calculation. When the input data type is FLOAT32 and the k axis is greater than 2048, grouped accumulation is used for the computation.
  <!-- end id11 -->
  <!-- npu="950" id12 -->
  - <term>Ascend 950PR&950DT products</term>:
    - cubeMathType=1: When the input data type is FLOAT32, it is converted to HFLOAT32 for computation. When the input is of other data types, no processing is performed.
    - cubeMathType=2: This option is not supported when the input data type is BFLOAT16.
    - cubeMathType=3: When the input data type is FLOAT32, it will be converted to HFLOAT32 for computation. This option is not supported when the input is of other data types.
    - cubeMathType=4: When the input data type is FLOAT32, the k axis is greater than 512, and any of the following conditions is met, vector cores are used for computation to improve the result precision (this method may degrade operator performance in some use cases). Condition 1: the m axis equals 1 and the n axis equals 1; Condition 2: both the m axis and the n axis are not 1, mat1 is transposed, and mat2 is not transposed.

  <!-- end id12 -->

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
      <col style="width: 283px">
      <col style="width: 120px">
      <col style="width: 747px">
      </colgroup>
      <thead>
        <tr>
          <th>Return Value</th>
          <th>Error Code</th>
          <th>Description</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>ACLNN_ERR_PARAM_NULLPTR</td>
          <td>161001</td>
          <td>The input selfRef, mat1, or mat2 is a null pointer.</td>
        </tr>
        <tr>
          <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
          <td rowspan="3">161002</td>
          <td>The data type of selfRef, mat1, or mat2 is not within the supported range, or the data format of selfRef is FRACTAL_NZ.</td>
        </tr>
        <tr>
          <td>mat1 and mat2 do not meet the multiplication conditions.</td>
        </tr>
        <tr>
          <td>The shapes of selfRef and mat1@mat2 are inconsistent.</td>
        </tr>
      </tbody>
      </table>

## aclnnInplaceAddmm

- **Parameters**

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
  <col style="width: 168px">
  <col style="width: 128px">
  <col style="width: 854px">
  </colgroup>
  <thead>
    <tr>
      <th>Parameter</th>
      <th>Input/Output</th>
      <th>Description</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Workspace memory address allocated on the Device.</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnInplaceAddmmGetWorkspaceSize.</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>Input</td>
      <td>op executor that contains the operator computation flow.</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Specifies the stream for task execution.</td>
    </tr>
  </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

- Deterministic computation:

  <!-- npu="910,310p" id13 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>: aclnnAddmm&aclnnInplaceAddmm defaults to a non-deterministic implementation. Deterministic computation can be enabled through aclrtCtxSetSysParamOpt.
  <!-- end id13 -->
  <!-- npu="950" id14 -->
  - <term>Ascend 950PR&950DT products</term>: aclnnAddmm&aclnnInplaceAddmm default to a deterministic implementation.

  <!-- end id14 -->

<!-- npu="950" id15 -->
- <term>Ascend 950PR&950DT products</term>:
  - When mat1 and mat2 are both of the FLOAT16 or BFLOAT16 data type, the computation precision of aclnnAddmm is as follows:
    - If self is a 1D tensor:
      - If out is of the FLOAT32 data type, MatMul uses FLOAT32 accumulation. beta×self, alpha×(mat1@mat2), and the intermediate result of adding the two parts all use FLOAT32. The final result is written to out in FLOAT32.
      - If self has a shape of [n] and alpha=1 and beta=1, self is directly used as the bias input of MatMul, and the addition is performed inside MatMul.
      - If out is of the FLOAT16 or BFLOAT16 data type, setting cubeMathType=4 does not make the intermediate computation results use FLOAT32.
    - If self is a 2D tensor:
      - If out is of the FLOAT32 data type, MatMul uses FLOAT32 accumulation. beta×self, alpha×(mat1@mat2), and the intermediate result of adding the two parts all use FLOAT32. The final result is written to out in FLOAT32.
      - If out is of the FLOAT16 or BFLOAT16 data type and cubeMathType=4, the preceding intermediate results use FLOAT32, and the final result is converted to the data type specified by out.
      - If out is of the FLOAT16 or BFLOAT16 data type and cubeMathType is not 4, the intermediate results are not converted to FLOAT32.
      - If self has a shape of [1, n] and alpha=1 and beta=1, self is directly used as the bias input of MatMul, and the addition is performed inside MatMul.

<!-- end id15 -->
- Computation consistency description

  <!-- npu="910,310p" id16 -->
  - <term>Atlas training products</term>, <term>Atlas inference products</term>:
    - When the strong consistency computation feature is enabled, the computation result is deterministic, and multiple executions produce the same output. In addition, the computation result is independent of the data location.
    - aclnnAddmm&aclnnInplaceAddmm default to a non-deterministic implementation. You can enable consistency by calling aclrtCtxSetSysParamOpt.
    - For example, during Matrix Multiplication, the accumulation order of different basic blocks may vary, which can cause slight differences in the computation results of the same data across different rows. However, when deterministic computation is enabled, the computation results will be identical as long as the inputs are the same, even across different rows.

  <!-- end id16 -->

## Example

The sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_addmm.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // Boilerplate for resource initialization.
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // Call aclrtMalloc to allocate device memory.
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  // Call aclrtMemcpy to copy host data to the device memory.
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // Calculate the strides of the contiguous tensor.
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // Call aclCreateTensor to create an aclTensor.
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. (Boilerplate) Initialize the device/stream. For details, see the acl API manual.
  // Set deviceId based on the actual device.
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. Construct the input and output. Customize the construction based on the API.
  std::vector<int64_t> selfShape = {2, 4};
  std::vector<int64_t> mat1Shape = {2, 3};
  std::vector<int64_t> mat2Shape = {3, 4};
  std::vector<int64_t> outShape = {2, 4};
  void* selfDeviceAddr = nullptr;
  void* mat1DeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat1 = nullptr;
  aclTensor* mat2 = nullptr;
  aclScalar* alpha = nullptr;
  aclScalar* beta = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<float> mat1HostData = {1, 1, 1, 2, 2, 2};
  std::vector<float> mat2HostData = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<float> outHostData(8, 0);
  int8_t cubeMathType = 1;
  float alphaValue = 1.2f;
  float betaValue = 1.0f;
  // Create the self aclTensor.
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mat1 aclTensor.
  ret = CreateAclTensor(mat1HostData, mat1Shape, &mat1DeviceAddr, aclDataType::ACL_FLOAT, &mat1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the mat2 aclTensor.
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // Create the alpha aclScalar.
  alpha = aclCreateScalar(&alphaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, return ret);
  // Create the beta aclScalar.
  beta = aclCreateScalar(&betaValue,aclDataType::ACL_FLOAT);
  CHECK_RET(beta != nullptr, return ret);
  // Create the out aclTensor.
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. Call the CANN operator library API. Replace it with the actual API name.
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // Call the first-phase API of aclnnAddmm.
  ret = aclnnAddmmGetWorkspaceSize(self, mat1, mat2, beta, alpha, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmmGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  }
  // Call the second-phase API of aclnnAddmm.
  ret = aclnnAddmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddmm failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);

  // 4. (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);

  // 5. Obtain the output value. Copy the result from the device memory to the host memory. Modify the code based on the specific API definition.
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // aclnnInplaceAddmm
  // step3 Call the CANN operator library API.
  LOG_PRINT("\ntest aclnnInplaceAddmm\n");
  // Call the first-phase API of aclnnInplaceAddmm.
  ret = aclnnInplaceAddmmGetWorkspaceSize(self, mat1, mat2, beta, alpha, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddmmGetWorkspaceSize failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  // Allocate device memory based on the workspaceSize computed by the first-phase API.
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  }
  // Call the second-phase API of aclnnInplaceAddmm.
  ret = aclnnInplaceAddmm(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceAddmm failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);

  // step4 (Boilerplate) Synchronize and wait for task execution to complete.
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);

  // step5 Obtain the output value. Copy the result from the device memory to the host memory.
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d.\n[ERROR msg]%s", ret, aclGetRecentErrMsg()); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. Release aclTensor and aclScalar. Modify the code based on the specific API definition.
  aclDestroyTensor(self);
  aclDestroyTensor(mat1);
  aclDestroyTensor(mat2);
  aclDestroyScalar(beta);
  aclDestroyScalar(alpha);
  aclDestroyTensor(out);

  // 7. Release device resources. Modify the code based on the specific API definition.
  aclrtFree(selfDeviceAddr);
  aclrtFree(mat1DeviceAddr);
  aclrtFree(mat2DeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
