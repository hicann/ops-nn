# aclnnTransQuantParam

[📄 View source code](https://gitcode.com/cann/ops-nn/tree/master/quant/trans_quant_param)

## Supported Products

|Product            |  Supported |
|:-------------------------|:----------:|
|  <term>Atlas A3 training series products/Atlas A3 inference series products</term>  |     √    |
|  <term>Atlas A2 training series products/Atlas A2 inference series products</term>    |     √    |
|  <term>Atlas 200I/500 A2 inference products</term>   |     ×    |
|  <term>Atlas inference series products</term>   |     ×    |
|  <term>Atlas training series products</term>   |     ×    |

## Function

- Description: Converts the input **scale** data from FLOAT32 type to UINT64 type required by the hardware, and stores the data in **quantParam**.
- Formula:
  1. `out` is in 64-bit format and is initialized to 0.

  2. The higher 19 bits of `scale` are truncated and stored in bit 32 of `out`, and bit 46 is changed to 1.

     $$
     out = out\ |\ (scale\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$

  3. The subsequent computation is performed based on the value of `offset`.
     - If `offset` does not exist, no subsequent computation is performed.
     - If `offset` exists:
       1. The value of `offset` is converted to an int value in the range of [–256, 255].

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. Nine bits of `offset` are retained and stored in bits 37 to 45 of out.

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
          $$

## Prototype

```Cpp
aclnnStatus aclnnTransQuantParam(
  const float  *scaleArray,
  uint64_t      scaleSize,
  const float  *offsetArray,
  uint64_t      offsetSize,
  uint64_t    **quantParam,
  uint64_t     *quantParamSize)
```

## aclnnTransQuantParam

- **Parameters:**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>scaleArray</td>
      <td>Input</td>
      <td>Pointer to the memory for storing the data of scale, corresponding to `scale` in the formula.</td>
      <td>Ensure that NaN and inf do not exist in the data of scale.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleSize</td>
      <td>Input</td>
      <td>Number of elements in the scale data.</td>
      <td>Ensure that `scaleSize` and `scaleArray` have the same number of elements.</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetArray</td>
      <td>Input</td>
      <td>Pointer to the memory for storing the data of offset, corresponding to `offset` in the formula.</td>
      <td>Ensure that NaN and inf do not exist in the data of offset.</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetSize</td>
      <td>Input</td>
      <td>Number of elements in the offset data.</td>
      <td>Ensure that `offsetSize` and `offsetArray` have the same number of elements.</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>Output</td>
      <td>Pointer to the memory for storing the quantParam data obtained after conversion, corresponding to `out` in the formula.</td>
      <td>-</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParamSize</td>
      <td>Output</td>
      <td>Number of elements in the stored quantParam data.</td>
      <td>Ensure that `quantParamSize` and `quantParam` have the same number of elements.</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **Returns:**

  **aclnnStatus**: status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_codes_nn.md).
  
  The API implements input parameter verification. The following errors may be thrown:

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>Return</th>
      <th>Error Code</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>quantParam is a null pointer.</td>
    </tr>
    <tr>
      <td>scaleArray is a null pointer.</td>
    </tr>
    <tr>
      <td>quantParamSize is a null pointer.</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>When scaleArray is not a null pointer, scaleSize is less than 1.</td>
    </tr>
    <tr>
      <td>When offsetArray is not a null pointer, offsetSize is less than 1.</td>
    </tr>
    <tr>
      <td>When offsetArray is a null pointer, offsetSize is not 0.</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>quantParam is a null pointer.</td>
    </tr>
  </tbody></table>

## Constraints

- Deterministic compute:
  - **aclnnTransQuantParam** defaults to a deterministic implementation.

## Example

The following example is for reference only. For details, see [Compilation and Running Sample](../../../docs/en/context/compilation_running_sample_nn.md).

```Cpp
#include <iostream>
#include "acl/acl.h"
#include "aclnnop/aclnn_trans_quant_param.h"

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

int main() {
  float scaleArray[3] = {1.0, 1.0, 1.0};
  uint64_t scaleSize = 3;
  float offsetArray[3] = {1.0, 1.0, 1.0};
  uint64_t offsetSize = 3;
  uint64_t *result = nullptr;
  uint64_t resultSize = 0;
  auto ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransQuantParam failed. ERROR: %d\n", ret); return ret);
  for (auto i = 0; i < resultSize; i++) {
    LOG_PRINT("result[%ld] is: %ld\n", i, result[i]);
  }
  free(result);
  return 0;
}
```
