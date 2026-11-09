# TransQuantParam

English | [简体中文](./README.md)

## Applicable Products

| Product | Support |
| --- | --- |
| <term>Ascend 950PR&950DT products</term> | √ |
| <term>Atlas A3 products</term> | √ |
| <term>Atlas A2 products</term> | √ |
| <term>Atlas 200I/500 A2 inference products</term> | × |
| <term>Atlas inference products</term> | × |
| <term>Atlas training products</term> | × |

## Function

- Function: Encodes FLOAT32 scale and optional offset values into UINT64 quantization parameters on the Host for subsequent quantized computation.
- Computation flow:
  1. Allocate and zero the output array based on the element counts of the two input arrays. Keep the output element count at 1 if the maximum input count is 1; otherwise, round it up to a multiple of 16.
  2. Write the complete 32-bit floating-point representation of each scale into the lowest 32 bits of the corresponding output element, and set bit 46 to 1. Bit positions are numbered from 0.
  3. Round each provided offset to the nearest integer, resolving ties away from zero, and saturate it to [-256,255]. Write its lowest 9 bits to bits 37–45 of the corresponding output element.

scale and offset are encoded independently by their respective array indices, without broadcasting. Unwritten bits and padding elements remain 0. For the full encoding formulas, see [aclnnTransQuantParam Function](docs/aclnnTransQuantParam_en.md#function).

## Parameters

This directory provides a Host-side aclnn interface. The parameters below are listed in interface order. All inputs and outputs are Host arrays or scalars.

| Parameter | Input/Output/Attribute | Description | Data Type | Data Format |
| --- | --- | --- | --- | --- |
| scaleArray | Input | Scale array to encode. Its length is specified by scaleSize. | FLOAT32 | - |
| scaleSize | Input | Number of elements in scaleArray. Must be greater than 0. | UINT64 | - |
| offsetArray | Optional input | Offset array to encode. If unused, pass nullptr and set offsetSize to 0. | FLOAT32 | - |
| offsetSize | Input | Number of elements in offsetArray. Must be greater than 0 if the array is provided, or 0 if omitted. | UINT64 | - |
| quantParam | Output | Encoded output array allocated by the interface. The caller must release it using std::free after use. | UINT64 | - |
| quantParamSize | Output | Number of elements in the encoded output array, including padding. Written back by the interface. | UINT64 | - |

## Constraints

- Input arrays must be stored contiguously in Host memory, and declared element counts must match actual array lengths.
- scale and offset must not contain NaN or Inf. Rounded offset values must be representable as 32-bit signed integers before saturation to [-256,255].
- The two arrays may have different lengths, but an input of length 1 is not automatically broadcast. To share a scale or offset across elements, expand the array before calling the interface.
- This interface completes the Host computation synchronously and returns the results. Its output memory differs from the Device tensor output of [TransQuantParamV2](../trans_quant_param_v2/README_en.md). Prepare inputs for subsequent calls according to the target interface's memory requirements.

## Invocation Description

| Invocation method | Sample code | Description |
| --- | --- | --- |
| aclnn API | [test_aclnn_trans_quant_param](examples/test_aclnn_trans_quant_param.cpp) | Calls [aclnnTransQuantParam](docs/aclnnTransQuantParam_en.md) to synchronously encode Host arrays and then releases the returned Host memory. |
