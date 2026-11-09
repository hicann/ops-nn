# TransQuantParam

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| <term>Ascend 950PR&950DT系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | √ |
| <term>Atlas A2系列产品</term> | √ |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

- 算子功能：在Host侧将FLOAT32格式的scale及可选offset编码为UINT64量化参数，供后续量化计算使用。
- 计算流程：
  1. 根据两个输入数组的元素数申请并清零输出数组。元素数为1时保持为1，其他情况向上补齐到16的倍数。
  2. 将每个scale的完整32位浮点表示写入对应输出元素的低32位，并将bit 46置1；位编号从0开始。
  3. 对传入的offset舍入到最近整数，中间值向远离0的方向舍入，再饱和到[-256,255]，将其低9位写入对应输出元素的bit 37～45。

scale与offset按各自数组下标独立编码，不进行广播；未写入的位及补齐元素保持为0。完整编码公式见[aclnnTransQuantParam功能说明](docs/aclnnTransQuantParam.md#功能说明)。

## 参数说明

本目录提供Host侧aclnn接口，以下按该接口的参数顺序说明。输入输出均为Host数组或标量。

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
| --- | --- | --- | --- | --- |
| scaleArray | 输入 | 待编码的scale数组，长度由scaleSize指定。 | FLOAT32 | - |
| scaleSize | 输入 | scaleArray的元素数，须大于0。 | UINT64 | - |
| offsetArray | 可选输入 | 待编码的offset数组；不使用时传入nullptr，并将offsetSize设为0。 | FLOAT32 | - |
| offsetSize | 输入 | offsetArray的元素数；提供数组时须大于0，不提供时须为0。 | UINT64 | - |
| quantParam | 输出 | 接口申请的编码结果数组，使用后由调用者通过std::free释放。 | UINT64 | - |
| quantParamSize | 输出 | 编码结果数组的元素数，包含补齐元素，由接口写回。 | UINT64 | - |

## 约束说明

- 输入数组须连续存放在Host内存中，声明的元素数须与实际数组长度一致。
- scale和offset不能包含NaN或Inf；offset舍入后的值须可用32位有符号整数表示，再执行[-256,255]范围的饱和处理。
- 两个数组的长度可以不同，但长度为1的输入不会自动广播。需要逐元素共用同一个scale或offset时，调用前须自行扩展数组。
- 本接口同步完成Host计算并返回结果。输出内存与[TransQuantParamV2](../trans_quant_param_v2/README.md)的Device侧Tensor输出不同，后续调用须按目标接口的内存要求准备输入。

## 调用说明

| 调用方式 | 样例代码 | 说明 |
| --- | --- | --- |
| aclnn API | [test_aclnn_trans_quant_param](examples/test_aclnn_trans_quant_param.cpp) | 通过[aclnnTransQuantParam](docs/aclnnTransQuantParam.md)同步编码Host数组，并释放返回的Host内存。 |
