# MaskedScatter

English | [简体中文](./README.md)

## Applicable Products

| Product                                                         | Support |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT products</term>                          |    √  |
| <term>Atlas A3 products</term>    |    √     |
| <term>Atlas A2 products</term>     |    √     |
| <term>Atlas 200I/500 A2 inference products</term>                      |    ×     |
| <term>Atlas inference products</term>                             |    ×     |
| <term>Atlas training products</term>                              |    ×   |

## Function

Copies the elements in source to the corresponding positions in selfRef at the positions where the elements in the mask tensor are True.

## Function Prototype

Each operator is a [two-phase API](../../docs/en/context/two_phase_api.md). You must first call the "aclnnInplaceMaskedScatterGetWorkspaceSize" API to obtain the workspace size required for the computation and the executor that contains the operator computation flow, and then call the "aclnnInplaceMaskedScatter" API to perform the computation.

```Cpp
aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
 aclTensor*         selfRef,
 const aclTensor*   mask,
 const aclTensor*   source,
 uint64_t*          workspaceSize,
 aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnInplaceMaskedScatter(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnInplaceMaskedScatterGetWorkspaceSize

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1477px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 255px">
    <col style="width: 270px">
    <col style="width: 143px">
    <col style="width: 164px">
    <col style="width: 145px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter name</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Usage instructions</th>
        <th>Data type</th>
        <th>Data format</th>
        <th>Dimensions (shape)</th>
        <th>Non-contiguous tensor</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>selfRef</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>-</td>
        <td>FLOAT, FLOAT16, DOUBLE, INT8, INT16, INT32, INT64, UINT8, BOOL, BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>mask</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>The shape must not be greater than selfRef, and mask and selfRef must satisfy the <a href="../../docs/en/context/broadcast_relationship.md">broadcast relationship</a>.</td>
        <td>BOOL</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>source</td>
        <td>Input</td>
        <td>Input tensor.</td>
        <td>The number of elements must be greater than or equal to the number of elements that are true in mask.</td>
        <td>Same as selfRef</td>
        <td>ND</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>Output</td>
        <td>Returns the workspace size that needs to be allocated on the Device.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>Output</td>
        <td>Returns the op executor, which contains the operator computation flow.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody></table>

- **Return Values**

  aclnnStatus: return status code. For details, see [aclnn return codes](../../docs/en/context/aclnn_return_code.md).

  The first-phase API completes the parameter validation and reports an error in the following scenarios:
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>Return value</th>
      <th>Error code</th>
      <th>Description</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>The passed-in selfRef, mask, or source is a null pointer.</td>
      </tr>
      <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>The data types of selfRef and mask are not within the supported range.</td>
      </tr>
      <tr>
      <td>The shapes of selfRef and mask cannot be broadcast.</td>
      </tr>
      <tr>
      <td>The shape dimensions of mask are greater than those of selfRef.</td>
      </tr>
      <tr>
      <td>The data type of source is different from the data type of selfRef.</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceMaskedScatter

- **Parameters**
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
      </colgroup>
      <thead>
      <tr>
      <th>Parameter name</th>
      <th>Input/Output</th>
      <th>Description</th>
      </tr></thead>
      <tbody>
      <tr>
      <td>workspace</td>
      <td>Input</td>
      <td>Memory address of the workspace allocated on the Device.</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>Input</td>
      <td>Workspace size allocated on the Device, obtained by the first-phase API aclnnInplaceMaskedScatterGetWorkspaceSize.</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>Input</td>
      <td>Op executor, which contains the operator computation flow.</td>
      </tr>
      <tr>
      <td>stream</td>
      <td>Input</td>
      <td>Stream for executing the task.</td>
      </tr>
      </tbody>
    </table>

- **Return Values**

  aclnnStatus: return status code. For details, see [aclnn return codes](../../docs/en/context/aclnn_return_code.md).

## Constraints

None

## Invocation Sample

| Invocation method   | Sample code           | Description                                         |
| ---------------- | --------------------------- | --------------------------------------------------- |
| aclnn API | [test_aclnn_masked_scatter](./examples/test_aclnn_masked_scatter.cpp) | Invoke the MaskedScatter operator through the [aclnnInplaceMaskedScatter](docs/aclnnInplaceMaskedScatter_en.md) API. |
