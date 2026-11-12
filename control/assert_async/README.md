# AssertAsync

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR&950DT系列产品</term>                         |    √     |
| <term>Atlas A3系列产品</term>     |    ×     |
| <term>Atlas A2系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2推理产品</term>                      |    ×     |
| <term>Atlas推理系列产品</term>                             |    ×     |
| <term>Atlas训练系列产品</term>                              |    ×     |

## 功能说明

- 算子功能：断言输入张量 `input_condition`（要求 `numel == 1`）的数值非 0。若其数值为 0（含 bool 的 `False`），
  设备侧打印属性 `assert_msg` 指定的消息并触发算子失败；数值非 0 直接通过。本算子**无输出张量**，
  语义对齐 PyTorch 的 `torch._assert_async`（含 `.msg` 重载语义）。

- 判定口径（bit-exact，无数值计算）：
  - 浮点类型（FLOAT16、FLOAT、DOUBLE、BFLOAT16）按符号位掩码判定：FLOAT 取 `& 0x7FFFFFFF`，FLOAT16/BFLOAT16 取 `& 0x7FFF`，
    DOUBLE 取 `& 0x7FFFFFFFFFFFFFFF`。`+0.0` 与 `-0.0` 均判失败，`NaN`、`+inf`、`-inf` 判通过。
  - 复数类型（COMPLEX64、COMPLEX128）实部与虚部精确同时为 0（含 `-0.0` 分量）判失败；`NaN`/`±inf` 分量通过。
  - 整型与布尔类型判全字节零：`0`（含 bool 的 `False`）判失败，非 0（含 `True`）判通过。

- 失败通路：kernel 内以 `AscendC::printf` 打印消息后调用 `AscendC::Trap()` 中断 AI Core，使算子执行失败；
  失败状态经 stream 同步（如 `aclrtSynchronizeStream`）回传给上层。判定通过时不产生任何打印或输出。

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|--------------|------|---------|---------|
| input_condition | 输入 | 待断言的条件张量，要求 `numel == 1`（rank-0 或任意收缩为单元素的 shape），维度不超过 8 维，否则返回 ACLNN_ERR_PARAM_INVALID（161002）。数值为 0（含 bool 的 `False`）判失败。 | BOOL、INT8、INT16、INT32、INT64、UINT8、FLOAT16、FLOAT、DOUBLE、BFLOAT16、COMPLEX64、COMPLEX128 | ND |
| assert_msg | 可选属性 | 断言失败时打印的消息字符串，默认 `""`。可用长度 254 字节（长度须小于 255；缓冲 256 字节，含结尾 `'\0'`），超长时返回 ACLNN_ERR_PARAM_INVALID（161002）。 | String | - |

> 本算子无输出张量。

## 约束说明

- 本算子无输出张量、无浮点计算，判定按行为 bit-exact 进行，不适用输出张量的浮点容差（atol/rtol）精度标准。
- 断言失败发生在设备侧 kernel 内，需通过 stream 同步感知算子失败。
- 本算子仅支持 aclnn 接口直调通路，不支持动态图（动态 shape）场景。

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|--------------|------------------------------------------------------------------------|----------------------------------------------------------------|
| aclnn 调用 | [test_aclnn_assert_async](./examples/arch35/test_aclnn_assert_async.cpp) | 通过 [aclnnAssertAsync](./docs/aclnnAssertAsync.md) 接口方式调用 AssertAsync 算子。 |
