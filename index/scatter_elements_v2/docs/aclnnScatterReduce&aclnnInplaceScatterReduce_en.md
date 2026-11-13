# aclnnScatterReduce&aclnnInplaceScatterReduce

<!-- md-trans-meta sourceCommit=unknown translatedAt=2026-09-22T16:50:26.527Z -->

## Applicable Products

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT products</term>: Not supported
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 products</term>: Supported
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 products</term>: Supported
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Description: Writes or reduces values in tensor `src` into tensor `self` along the specified axis and at the positions specified by `index`.
- Reduction semantics defined by the API:
  - `none`: replacement write
  - `add`: accumulation
  - `mul`: cumulative multiplication
  - `max`: maximum value
  - `min`: minimum value
  - `mean`: average value
  - Actual support varies by product and must satisfy both the Applicable Products section above and the Constraints section in this document.
- `include_self` semantics:
  - `include_self=true`: the original values of `self` participate in the reduction.
  - `include_self=false`: The original value of `self` does not participate in the reduction as the initial value; the first `src` value is used as the initial value on the first hit.
- API responsibility boundaries:
  - `aclnnScatterReduce/aclnnInplaceScatterReduce` uniformly handles reduction use cases that explicitly require `include_self`. The available combinations are subject to product capability constraints.
  - `aclnnScatterAdd` is a shortcut API with fixed `reduce=add` and `includeSelf=true`, and no longer handles use cases that explicitly pass `include_self`.
  - If only the traditional `scatter/scatter_value` semantics are required and `include_self` is not explicitly exposed, use `aclnnScatter/aclnnInplaceScatter/aclnnScatterValue/aclnnInplaceScatterValue`.

- Example:
  For a 3D tensor, `self` is updated as follows:

  ```text
  out[index[i][j][k]][j][k] = src[i][j][k]                                 # If dim == 0 && reduce == 0(none)
  out[i][index[i][j][k]][k] += src[i][j][k]                                # If dim == 1 && reduce == 1(add)
  out[i][j][index[i][j][k]] = max(out[i][j][index[i][j][k]], src[i][j][k]) # If dim == 2 && reduce == 3(max)
  ```

- The following requirements must be met during computation:
  - `self`, `index`, and `src` must have the same number of dimensions.
  - For each dimension `d`, `index.size(d) <= src.size(d)` must be satisfied.
  - For each dimension `d`, if `d != dim`, `index.size(d) <= self.size(d)` must be satisfied.
  - The value of `dim` must be in the range `[-number of dimensions of self, number of dimensions of self-1]`.
  - The number of dimensions of `self` should be less than or equal to 8.
  - The values of `index` along dimension `dim` must be in the range `[0, self.size(dim)-1]`.

## Function Prototype

aclnnScatterReduce and aclnnInplaceScatterReduce implement the same reduction scatter function. The usage differences are as follows. Select an appropriate operator based on your actual use case.

- `aclnnScatterReduce`: a new output tensor object must be created to store the computation result.
- `aclnnInplaceScatterReduce`: no new output tensor object is required; the computation result is stored directly in the memory of the input tensor.

Each operator uses a [two-phase API](../../../docs/en/context/two_phase_api.md). Call "aclnnScatterReduceGetWorkspaceSize" or "aclnnInplaceScatterReduceGetWorkspaceSize" first to obtain the workspace size required for the computation and the executor that contains the computation process of the operator, and then call "aclnnScatterReduce" or "aclnnInplaceScatterReduce" to perform the computation.

```Cpp
aclnnStatus aclnnScatterReduceGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  bool             includeSelf,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterReduce(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  const aclrtStream   stream)
```

```Cpp
aclnnStatus aclnnInplaceScatterReduceGetWorkspaceSize(
  aclTensor*       selfRef,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  bool             includeSelf,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatterReduce(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterReduceGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
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
        <td>self (aclTensor*)</td>
        <td>Input</td>
        <td>The `self` in the formula.</td>
        <td>The data type must be the same as that of src.<br>The number of dimensions must be the same as those of index and src.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>The dimension to scatter along.</td>
        <td>The value range is [-number of dimensions of self, number of dimensions of self - 1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index(aclTensor*)</td>
        <td>Input</td>
        <td>Index tensor.</td>
        <td>The number of dimensions must be the same as self and src.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src(aclTensor*)</td>
        <td>Input</td>
        <td>`src` in the formula.</td>
        <td>The data type must be the same as that of self.<br>The number of dimensions must be the same as those of self and index.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce (int64_t)</td>
        <td>Input</td>
        <td>The reduction operation to apply.</td>
        <td>Available operation options and the corresponding int values are (none, 0), (add, 1), (mul, 2), (max, 3), (min, 4), and (mean, 5).<br>0 indicates replacement, 1 indicates accumulation, 2 indicates multiplication, 3 indicates maximum, 4 indicates minimum, and 5 indicates average.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeSelf (bool)</td>
        <td>Input</td>
        <td>Whether the original values in self participate in the reduction computation.</td>
        <td>true includes self in the reduction; false sums src at updated positions and preserves self elsewhere. Empty index preserves self.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out (aclTensor*)</td>
        <td>Output</td>
        <td>Computation result.</td>
        <td>The data type must be the same as that of self.<br>The shape must be the same as that of self.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the size of the workspace to be allocated on the device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor containing the operator calculation process.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
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
        <td>self, index, src, or out is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="8">161002</td>
        <td>The data type of self, index, src, or out is out of the supported range.</td>
      </tr>
      <tr>
        <td>The data types of self, src, and out are different.</td>
      </tr>
      <tr>
        <td>The number of dimensions of self, index, and src is inconsistent.</td>
      </tr>
      <tr>
        <td>The shapes of self and out are inconsistent.</td>
      </tr>
      <tr>
        <td>The value of dim is not in the range [-number of dimensions of self, number of dimensions of self - 1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of self exceeds 8.</td>
      </tr>
      <tr>
        <td>The shape does not meet the constraints: for each dimension d, index.size(d) <= src.size(d); and when d != dim, index.size(d) <= self.size(d).</td>
      </tr>
      <tr>
        <td>reduce is out of the supported range. It must be in {0, 1, 2, 3, 4, 5}.</td>
      </tr>
    </tbody></table>

## aclnnScatterReduce

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Workspace memory address allocated on the Device side.</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Input</td>
        <td>Workspace size allocated on the Device side, obtained by the first-phase API aclnnScatterReduceGetWorkspaceSize.</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Input</td>
        <td>Executor of the op, containing the computation flow of the operator.</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>Input</td>
        <td>Stream on which the task is executed.</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## aclnnInplaceScatterReduceGetWorkspaceSize

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
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
        <td>selfRef (aclTensor*)</td>
        <td>Input/Output</td>
        <td>Target tensor of scatter_reduce.</td>
        <td>shape must have the same number of dimensions as index and src.<br>The data type must be the same as that of src.<br>Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim (int64_t)</td>
        <td>Input</td>
        <td>Specifies the dimension along which the scatter_reduce operation is performed.</td>
        <td>The value range is [-number of dimensions of selfRef, number of dimensions of selfRef-1].</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index (aclTensor*)</td>
        <td>Input</td>
        <td>Index tensor.</td>
        <td>Specifies the positions where elements of the src tensor are scattered into the selfRef tensor. <br>Empty tensors are supported.</td>
        <td>INT32, INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src(aclTensor*)</td>
        <td>Input</td>
        <td>Source tensor.</td>
        <td>The data type must be the same as that of selfRef.<br>Empty tensors are supported.</td>
        <td>UINT8, INT8, INT16, INT32, INT64, BOOL, FLOAT16, FLOAT32, DOUBLE, COMPLEX64, COMPLEX128, BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce (int64_t)</td>
        <td>Input</td>
        <td>Specifies the reduction operation to apply.</td>
        <td>Available operations: 0 indicates replacement, 1 indicates sum, 2 indicates product, 3 indicates maximum, 4 indicates minimum, and 5 indicates mean.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeSelf (bool)</td>
        <td>Input</td>
        <td>Whether the original values in selfRef participate in the reduction computation.</td>
        <td>true includes self in the reduction; false sums src at updated positions and preserves self elsewhere. Empty index preserves self.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize (uint64_t*)</td>
        <td>Output</td>
        <td>Returns the workspace size to be requested on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor (aclOpExecutor**)</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the calculation process of the operator.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The first-phase API performs input parameter validation and returns an error in the following scenarios:

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
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
        <td>selfRef, index, or src is a null pointer.</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>The data types of selfRef, index, and src are out of the supported range.</td>
      </tr>
      <tr>
        <td>The data types of selfRef and src are different.</td>
      </tr>
      <tr>
        <td>The number of dimensions of selfRef, index, and src is inconsistent.</td>
      </tr>
      <tr>
        <td>The value of dim is not in the range [-the number of dimensions of selfRef, the number of dimensions of selfRef - 1].</td>
      </tr>
      <tr>
        <td>The number of dimensions of selfRef exceeds 8.</td>
      </tr>
      <tr>
        <td>The shape does not meet the restrictions: for each dimension d, index.size(d) <= src.size(d); and when d != dim, index.size(d) <= selfRef.size(d).</td>
      </tr>
      <tr>
        <td>reduce is out of the supported range. It must be one of {0, 1, 2, 3, 4, 5}.</td>
      </tr>
    </tbody></table>

## aclnnInplaceScatterReduce

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>Workspace size allocated on the Device, obtained by calling the first-phase API aclnnInplaceScatterReduceGetWorkspaceSize.</td>
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
    </tbody></table>

- **Return value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

## Constraints

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT products</term>:
  - The AI Core path supports only `reduce=0(none)`, `reduce=1(add)`, and `reduce=2(mul)`.
  - When `reduce` is 1 (add) or 2 (mul), only `includeSelf=true` is supported.
  - The implementation is nondeterministic by default. When `reduce` is 0 (none) or 1 (add), deterministic computation can be enabled through `aclrtSetSysParamOpt`; it cannot be enabled when `reduce` is 2 (mul).
<!-- end id7 -->

<!-- npu="A3,910b" id8 -->
- <term>Atlas A3 products</term> and <term>Atlas A2 products</term>:
  - The implementation is deterministic by default.
  - `reduce` supports only 1 (add).
  - `reduce=1(add)`:
    - The data type of `self/src/out` must be `FLOAT32`.
    - For nonempty inputs with `includeSelf=false`, normalize `dim` to a nonnegative axis `d`. If `d=0`, `self/index/src` sizes must match on every axis except the first and last axes. If `d>0`, their sizes must match on every axis except axis 0 and axis `d`.
<!-- end id8 -->

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

**aclnnScatterReduce sample code:**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_reduce.h"

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
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHost = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indexHost = {0, 2, 1, 1, 0, 2};
  std::vector<float> srcHost = {10, 20, 30, 40, 50, 60};
  std::vector<float> outHost(6, 0);

  void *selfDev = nullptr, *indexDev = nullptr, *srcDev = nullptr, *outDev = nullptr, *workspace = nullptr;
  aclTensor *self = nullptr, *index = nullptr, *src = nullptr, *out = nullptr;

  CHECK_RET(CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT, &self) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(indexHost, shape, &indexDev, ACL_INT64, &index) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(srcHost, shape, &srcDev, ACL_FLOAT, &src) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out) == ACL_SUCCESS, return -1);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int64_t dim = 1;
  int64_t reduce = 1;   // add
  bool includeSelf = true;

  ret = aclnnScatterReduceGetWorkspaceSize(self, dim, index, src, reduce, includeSelf, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnScatterReduce(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  LOG_PRINT("aclnnScatterReduce run success.\n");
  return 0;
}
```
