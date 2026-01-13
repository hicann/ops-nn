# aclnnTransQuantParam

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：将输入scale数据从FLOAT32类型转换为硬件需要的UINT64类型，并存储到quantParam中。
- 计算公式：
  1. `out`为64位格式，初始为0。

  2. `scale`按bit位取高19位截断，存储于`out`的bit位32位处，并将46位修改为1。

     $$
     out = out\ |\ (scale\ \&\ 0xFFFFE000)\ |\ (1\ll46)
     $$

  3. 根据`offset`取值进行后续计算：
     - 若`offset`不存在，不再进行后续计算。
     - 若`offset`存在：
       1. 将`offset`值处理为int，范围为[-256, 255]。

          $$
          offset = Max(Min(INT(Round(offset)),255),-256)
          $$

       2. 再将`offset`按bit位保留9位并存储于out的37到45位。

          $$
          out = (out\ \&\ 0x4000FFFFFFFF)\ |\ ((offset\ \&\ 0x1FF)\ll37)
          $$

## 函数原型

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

- **参数说明：**

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
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th>维度(shape)</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>scaleArray</td>
      <td>输入</td>
      <td>表示指向存储scale数据的内存，对应公式中的`scale`。</td>
      <td>需要保证scale数据中不存在NaN和inf。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleSize</td>
      <td>输入</td>
      <td>表示scale数据的数量。</td>
      <td>需要自行保证`scaleSize`与`scaleArray`包含的元素个数相同。</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetArray</td>
      <td>输入</td>
      <td>表示指向存储offset数据的内存，对应公式中的`offset`。</td>
      <td>需要保证offset数据中不存在NaN和inf。</li></ul></td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>offsetSize</td>
      <td>输入</td>
      <td>表示offset数据的数量。</td>
      <td>需要自行保证`offsetSize`与`offsetArray`包含的元素个数相同。</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParam</td>
      <td>输出</td>
      <td>表示指向存储转换得到的quantParam数据的内存的地址，对应公式中的`out`。</td>
      <td>-</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantParamSize</td>
      <td>输出</td>
      <td>表示存储quantParam数据的数量。</td>
      <td>需要自行保证`quantParamSize`与`quantParam`包含的元素个数相同。</li></ul></td>
      <td>UINT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="3">161001</td>
      <td>参数quantParam是空指针。</td>
    </tr>
    <tr>
      <td>参数scaleArray是空指针。</td>
    </tr>
    <tr>
      <td>参数quantParamSize是空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>当scaleArray不为空指针时，参数scaleSize < 1。</td>
    </tr>
    <tr>
      <td>当offsetArray不为空指针时，参数offsetSize < 1。</td>
    </tr>
    <tr>
      <td>当offsetArray为空指针时，参数offsetSize不等于0。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>quantParam为空指针。</td>
    </tr>
  </tbody></table>

## 约束说明

- 确定性计算：
  - aclnnTransQuantParam默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

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
