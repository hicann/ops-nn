# aclnnTransQuantParam

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：在Host侧将FLOAT32格式的scale及可选offset编码为UINT64量化参数。接口申请输出数组，并通过quantParam返回数组地址，通过quantParamSize返回包含补齐元素的数组长度。
- 计算公式：

  1. 令$L=\max(\mathrm{scaleSize},\mathrm{offsetSize})$，输出数组的元素数$P$按下式计算，并写入`*quantParamSize`。接口申请$P$个UINT64元素并全部初始化为0；下文用$q_i$表示`(*quantParam)[i]`。

     $$
     P=\begin{cases}
     1, & L=1\\
     16\lceil L/16\rceil, & L>1
     \end{cases}
     $$

  2. 对$0\le i<\mathrm{scaleSize}$，将`scaleArray[i]`按位重解释为UINT32，保留完整32位，再扩展为UINT64并将bit 46置1。位编号从0开始；这一步不清除scale的低13位。

     $$
     q_i=\operatorname{UINT64}\left(\operatorname{BitCast}_{\mathrm{UINT32}}(\mathrm{scaleArray}[i])\right)\mathbin{|}(1\ll46)
     $$

  3. 若`offsetArray`为nullptr，不写入offset；否则，对$0\le i<\mathrm{offsetSize}$，先按`std::round`规则舍入到最近整数，中间值向远离0的方向舍入，例如0.5变为1、-0.5变为-1。舍入结果须可用32位有符号整数表示，再饱和到[-256,255]：

     $$
     o_i=\max\left(-256,\min\left(255,\operatorname{Round}_{\mathrm{away}}(\mathrm{offsetArray}[i])\right)\right)
     $$

     将$o_i$转换为UINT64，取低9位写入$q_i$的bit 37～45，保留已写入的scale及bit 46：

     $$
     q_i=(q_i\mathbin{\&}0x4000FFFFFFFF)\mathbin{|}
     \left((\operatorname{UINT64}(o_i)\mathbin{\&}0x1FF)\ll37\right)
     $$

两个数组按各自长度独立处理，不进行广播。未写入scale或offset的位置保持相应位域为0；$L\le i<P$的补齐元素为0。

## 函数原型

该接口同步完成Host计算，直接返回结果，不使用workspace、executor或stream。

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

- **参数说明**

  输入数组和输出数组均位于Host侧，不使用aclTensor。参数表中的数据格式、维度及非连续Tensor列不适用，统一以“-”表示。

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
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度(shape)</th>
        <th>非连续Tensor</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>scaleArray（const float*）</td>
        <td>输入</td>
        <td>指向待编码的scale数组，对应公式中的scaleArray。</td>
        <td>连续存放scaleSize个float元素；数值不能包含NaN或Inf。</td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>scaleSize（uint64_t）</td>
        <td>输入</td>
        <td>scaleArray的元素数，对应公式中的scaleSize。</td>
        <td>须大于0，且与scaleArray的实际长度一致。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetArray（const float*）</td>
        <td>可选输入</td>
        <td>指向待编码的offset数组，对应公式中的offsetArray。</td>
        <td><ul><li>不使用时传入nullptr，并将offsetSize设为0，输出中的offset位域保持为0。</li><li>提供时须连续存放offsetSize个float元素；数值不能包含NaN或Inf，舍入结果须可用32位有符号整数表示。</li></ul></td>
        <td>FLOAT32</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>offsetSize（uint64_t）</td>
        <td>输入</td>
        <td>offsetArray的元素数，对应公式中的offsetSize。</td>
        <td>offsetArray为nullptr时须为0；否则须大于0，且与offsetArray的实际长度一致。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantParam（uint64_t**）</td>
        <td>输出</td>
        <td>接收编码结果数组的地址，数组元素对应公式中的q。</td>
        <td><ul><li>传入一个uint64_t*变量的地址，无需预先申请输出数组。</li><li>接口通过calloc申请Host内存并写入*quantParam，调用者使用后须通过std::free释放该内存。</li></ul></td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantParamSize（uint64_t*）</td>
        <td>输出</td>
        <td>接收输出数组的元素数，对应公式中的P。</td>
        <td>由接口写回，包含补齐元素；计数单位为元素，不是字节。计算方法见<a href="#功能说明">功能说明</a>。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码/状态码</th>
        <th>码值</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>ACLNN_ERR_PARAM_NULLPTR</td>
        <td>161001</td>
        <td>quantParam或quantParamSize是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>scaleArray为空指针，或scaleSize为0。</li><li>offsetArray不为空指针，但offsetSize为0。</li><li>offsetArray为空指针，但offsetSize不为0。</li></ul></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>输出数组的Host内存申请失败，此时*quantParam为nullptr，*quantParamSize为0。</td>
      </tr>
    </tbody>
  </table>

## 约束说明

- scaleArray、offsetArray的实际长度须与声明的元素数一致，接口不校验数组的实际内存容量。
- 两个输入数组的长度可以不同，不进行广播。例如，scaleSize为3、offsetSize为1时，仅第0个输出元素包含offset，第1、2个元素仅包含scale；若要共用同一个offset，须在调用前将offsetArray扩展为3个元素。
- 输出长度包含按16个元素对齐的补齐部分。例如，scaleSize与offsetSize均为3时，quantParamSize为16，前3个元素为编码结果，后13个元素为0；两者的最大值为1时，quantParamSize为1。
- 返回的quantParam数组位于Host内存；后续接口若需要Device侧数据，须按该接口要求拷贝并构造输入。
- 确定性计算：aclnnTransQuantParam默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "aclnnop/aclnn_trans_quant_param.h"

int main()
{
    // 输入为Host数组；传入nullptr、0可省略offset。
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

    // 本例返回16个元素：前3个为编码结果，后13个为补齐的0。
    for (uint64_t i = 0; i < resultSize; ++i) {
        std::cout << "result[" << i << "] is: " << result[i] << '\n';
    }
    std::free(result);
    return 0;
}
```
