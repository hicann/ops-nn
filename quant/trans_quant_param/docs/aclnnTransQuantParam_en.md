# aclnnTransQuantParam

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
- <term>Atlas 200I/500 A2 inference products</term>: Not supported
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas inference products</term>: Not supported
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas training products</term>: Not supported
<!-- end id6 -->

## Function

- Function: Encodes FLOAT32 scale and optional offset values into UINT64 quantization parameters on the Host. The interface allocates the output array, returns its address through quantParam, and returns its length, including padding, through quantParamSize.
- Formula:

  1. Let $L=\max(\mathrm{scaleSize},\mathrm{offsetSize})$. Calculate the output element count $P$ using the following formula and write it to `*quantParamSize`. The interface allocates $P$ UINT64 elements and initializes all of them to 0. In the following description, $q_i$ denotes `(*quantParam)[i]`.

     $$
     P=\begin{cases}
     1, & L=1\\
     16\lceil L/16\rceil, & L>1
     \end{cases}
     $$

  2. For $0\le i<\mathrm{scaleSize}$, reinterpret `scaleArray[i]` bitwise as UINT32, preserve all 32 bits, widen it to UINT64, and set bit 46 to 1. Bit positions are numbered from 0. This step does not clear the lowest 13 bits of scale.

     $$
     q_i=\operatorname{UINT64}\left(\operatorname{BitCast}_{\mathrm{UINT32}}(\mathrm{scaleArray}[i])\right)\mathbin{|}(1\ll46)
     $$

  3. If `offsetArray` is nullptr, do not write offset bits. Otherwise, for $0\le i<\mathrm{offsetSize}$, first round to the nearest integer using `std::round`, resolving ties away from zero; for example, 0.5 becomes 1 and -0.5 becomes -1. The rounded value must be representable as a 32-bit signed integer. Then saturate it to [-256,255]:

     $$
     o_i=\max\left(-256,\min\left(255,\operatorname{Round}_{\mathrm{away}}(\mathrm{offsetArray}[i])\right)\right)
     $$

     Convert $o_i$ to UINT64 and write its lowest 9 bits to bits 37–45 of $q_i$, preserving the previously written scale bits and bit 46:

     $$
     q_i=(q_i\mathbin{\&}0x4000FFFFFFFF)\mathbin{|}
     \left((\operatorname{UINT64}(o_i)\mathbin{\&}0x1FF)\ll37\right)
     $$

The two arrays are processed independently according to their own lengths, without broadcasting. Unwritten scale or offset bit fields remain 0, and padding elements at $L\le i<P$ are 0.

## Function Prototype

This interface completes the Host computation synchronously and returns the results directly. It does not use workspace, executor, or stream.

```cpp
aclnnStatus aclnnTransQuantParam(
  const float* scaleArray,
  uint64_t     scaleSize,
  const float* offsetArray,
  uint64_t     offsetSize,
  uint64_t**   quantParam,
  uint64_t*    quantParamSize)
```

## aclnnTransQuantParam

- **Parameters**

  Both input and output arrays reside on the Host, and aclTensor is not used. The data format, dimension, and non-contiguous tensor columns in the parameter table are not applicable and are shown as "-".

  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
    </colgroup>
    <thead>
      <tr>
        <th>Parameter</th>
        <th>Input/Output</th>
        <th>Description</th>
        <th>Instruction</th>
        <th>Data Type</th>
        <th>Data Format</th>
        <th>Dimensions (Shape)</th>
        <th>Non-contiguous Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>scaleArray (const float*)</td>
        <td>Input</td>
        <td>Pointer to the scale array to encode, corresponding to scaleArray in the formula.</td>
        <td>Must contain scaleSize contiguous float elements. NaN and Inf are not allowed.</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleSize (uint64_t)</td>
        <td>Input</td>
        <td>Number of elements in scaleArray, corresponding to scaleSize in the formula.</td>
        <td>Must be greater than 0 and equal to the actual length of scaleArray.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetArray (const float*)</td>
        <td>Optional input</td>
        <td>Pointer to the offset array to encode, corresponding to offsetArray in the formula.</td>
        <td><ul><li>If unused, pass nullptr and set offsetSize to 0. The offset bit field in the output remains 0.</li><li>If provided, it must contain offsetSize contiguous float elements. NaN and Inf are not allowed, and rounded values must be representable as 32-bit signed integers.</li></ul></td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetSize (uint64_t)</td>
        <td>Input</td>
        <td>Number of elements in offsetArray, corresponding to offsetSize in the formula.</td>
        <td>Must be 0 when offsetArray is nullptr. Otherwise, it must be greater than 0 and equal to the actual length of offsetArray.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantParam (uint64_t**)</td>
        <td>Output</td>
        <td>Receives the address of the encoded output array. Array elements correspond to q in the formula.</td>
        <td><ul><li>Pass the address of a uint64_t* variable. There is no need to allocate the output array in advance.</li><li>The interface allocates Host memory using calloc and writes its address to *quantParam. The caller must release this memory using std::free after use.</li></ul></td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantParamSize (uint64_t*)</td>
        <td>Output</td>
        <td>Receives the number of elements in the output array, corresponding to P in the formula.</td>
        <td>Written back by the interface and includes padding elements. The count is in elements, not bytes. For the calculation, see <a href="#function">Function</a>.</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **Return Value**

  aclnnStatus: return status code. For details, see [aclnn Return Codes](../../../docs/en/context/aclnn_return_code.md).

  The API performs input parameter validation and returns an error in the following scenarios:

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>Return/Status Code</th>
        <th>Value</th>
        <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>quantParam or quantParamSize is a null pointer.</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>scaleArray is a null pointer, or scaleSize is 0.</li><li>offsetArray is not a null pointer, but offsetSize is 0.</li><li>offsetArray is a null pointer, but offsetSize is not 0.</li></ul></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>Host memory allocation for the output array failed. In this case, *quantParam is nullptr and *quantParamSize is 0.</td>
      </tr>
    </tbody>
  </table>

## Constraints

- The actual lengths of scaleArray and offsetArray must match their declared element counts. The interface does not check the actual memory capacity of the arrays.
- The two input arrays may have different lengths and are not broadcast. For example, when scaleSize is 3 and offsetSize is 1, only output element 0 contains an offset, while elements 1 and 2 contain only scale. To share the same offset across all elements, expand offsetArray to 3 elements before calling the interface.
- The output length includes padding to a multiple of 16 elements. For example, when scaleSize and offsetSize are both 3, quantParamSize is 16: the first 3 elements contain encoded results, and the last 13 elements are 0. When the maximum of the two sizes is 1, quantParamSize is 1.
- The returned quantParam array resides in Host memory. If a subsequent interface requires Device data, copy the data and construct the input as required by that interface.
- Deterministic computation: aclnnTransQuantParam uses a deterministic implementation by default.

## Example

The following sample code is for reference only. For details about compilation and execution, see [Compile and Run Samples](../../../docs/en/context/compile_and_run_sample.md).

```cpp
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "aclnnop/aclnn_trans_quant_param.h"

int main()
{
    // Inputs are Host arrays. Pass nullptr and 0 to omit offset.
    const float scaleArray[] = {1.0F, 1.0F, 1.0F};
    const float offsetArray[] = {1.0F, 1.0F, 1.0F};
    const uint64_t scaleSize = sizeof(scaleArray) / sizeof(scaleArray[0]);
    const uint64_t offsetSize = sizeof(offsetArray) / sizeof(offsetArray[0]);
    uint64_t* result = nullptr;
    uint64_t resultSize = 0;
    const auto ret = aclnnTransQuantParam(scaleArray, scaleSize, offsetArray, offsetSize, &result, &resultSize);
    if (ret != ACLNN_SUCCESS) {
        std::cerr << "aclnnTransQuantParam failed. ERROR: " << ret << '\n';
        return 1;
    }

    // This example returns 16 elements: 3 encoded results followed by 13 padding zeros.
    for (uint64_t i = 0; i < resultSize; ++i) {
        std::cout << "result[" << i << "] is: " << result[i] << '\n';
    }
    std::free(result);
    return 0;
}
```
