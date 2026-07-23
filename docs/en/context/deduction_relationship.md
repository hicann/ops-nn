# Type Promotion Relationships

## Promotion Rules

When the **input aclTensor data types** of an API (such as aclnnAdd, aclnnMul, etc.) are inconsistent, the API internally deduces a data type and converts the input data to that data type for calculation.

For the data types supported by aclTensor, refer to [Data Types](./data_type.md). Some of these types satisfy the following promotion rules, and the promotion principle is similar to PyTorch's [Type Promotion](https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc).

> Note:
>
> - For convenience of description, the data types used in the table are **abbreviated forms**, representing: ACL\_FLOAT(f32), ACL\_FLOAT16(f16), ACL\_DOUBLE(f64), ACL\_BF16(bf16), ACL\_INT8(s8), ACL\_UINT8(u8), ACL\_INT16(s16), ACL\_UINT16(u16), ACL\_INT32(s32), ACL\_UINT32(u32), ACL\_INT64(s64), ACL\_UINT64(u64), ACL\_BOOL(bool), ACL\_COMPLEX32(c32), ACL\_COMPLEX64(c64), ACL\_COMPLEX128(c128).
> - The table header and the leftmost column represent the two input data types to be deduced, and the corresponding position in the table represents the deduced data type.
> - The cross mark (×) in the table indicates that these two types cannot perform promotion calculation.

**Table 1**  Data Type Promotion Relationships

| Data Type  | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  | u16  | s32  | u32  | s64  | u64  | bool | c32  | c64  | c128 |
| :------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| **f32**  | f32  | f32  | f64  | f32  | f32  | f32  | f32  |  ×   | f32  |  ×   | f32  |  ×   | f32  | c64  | c64  | c128 |
| **f16**  | f32  | f16  | f64  | f32  | f16  | f16  | f16  |  ×   | f16  |  ×   | f16  |  ×   | f16  | c32  | c64  | c128 |
| **f64**  | f64  | f64  | f64  | f64  | f64  | f64  | f64  |  ×   | f64  |  ×   | f64  |  ×   | f64  | c128 | c128 | c128 |
| **bf16** | f32  | f32  | f64  | bf16 | bf16 | bf16 | bf16 |  ×   | bf16 |  ×   | bf16 |  ×   | bf16 | c32  | c64  | c128 |
|  **s8**  | f32  | f16  | f64  | bf16 |  s8  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  s8  | c32  | c64  | c128 |
|  **u8**  | f32  | f16  | f64  | bf16 | s16  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   |  u8  | c32  | c64  | c128 |
| **s16**  | f32  | f16  | f64  | bf16 | s16  | s16  | s16  |  ×   | s32  |  ×   | s64  |  ×   | s16  | c32  | c64  | c128 |
| **u16**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u16  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s32**  | f32  | f16  | f64  | bf16 | s32  | s32  | s32  |  ×   | s32  |  ×   | s64  |  ×   | s32  | c32  | c64  | c128 |
| **u32**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u32  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |
| **s64**  | f32  | f16  | f64  | bf16 | s64  | s64  | s64  |  ×   | s64  |  ×   | s64  |  ×   | s64  | c32  | c64  | c128 |
| **u64**  |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   |  ×   | u64  |  ×   |  ×   |  ×   |  ×   |
| **bool** | f32  | f16  | f64  | bf16 |  s8  |  u8  | s16  |  ×   | s32  |  ×   | s64  |  ×   | bool | c32  | c64  | c128 |
| **c32**  | c64  | c32  | c128 | c32  | c32  | c32  | c32  |  ×   | c32  |  ×   | c32  |  ×   | c32  | c32  | c64  | c128 |
| **c64**  | c64  | c64  | c128 | c64  | c64  | c64  | c64  |  ×   | c64  |  ×   | c64  |  ×   | c64  | c64  | c64  | c128 |
| **c128** | c128 | c128 | c128 | c128 | c128 | c128 | c128 |  ×   | c128 |  ×   | c128 |  ×   | c128 | c128 | c128 | c128 |

## Promotion Examples

- When calling the aclnnAdd interface, if the data types of the input parameters are inconsistent, one is float16 and one is float32, the API internally converts the float16 data type to float32 data type and then performs the calculation.
- When calling the aclnnAdd interface, if the data types of the input parameters are inconsistent, one is float32 and one is bool, the API internally converts the bool data type to float32 data type and then performs the calculation.
