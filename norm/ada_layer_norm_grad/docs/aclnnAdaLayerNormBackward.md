# aclnnAdaLayerNormBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/ada_layer_norm_grad)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |


## 功能说明

- 接口功能：[aclnnAdaLayerNormV2](../../ada_layer_norm_v2/docs/aclnnAdaLayerNormV2.md)的反向传播。用于计算输入张量的梯度，以便在反向传播过程中更新模型参数。

- 计算公式：
  正向：

  $$
  out = LayerNorm(input)\times(1+scale)+shift
  $$

  反向：

  $$
  z = (input - mean) \times rstd
  $$
  
  $$
  dy\_g = gradOut \times weightOptional \times (1 + scale)
  $$
  
  $$
  temp_1 = 1/N \times \sum_{reduce\_axis\_2} (gradOut \times weightOptional \times (1 + scale))
  $$
  
  $$
  temp_2 = 1/N \times (input - mean) \times rstd \times \sum_{reduce\_axis\_2}(gradOut \times weightOptional \times (1 + scale) \times (input - mean) \times rstd)
  $$

  $$
  gradInputOut = (gradOut \times weightOptional \times (1 + scale) - (temp_1 + temp_2)) \times rstd
  $$
  
  $$
  gradScaletOut =  \sum_{reduce\_axis\_1}(gradOut \times ((input - mean) \times rstd \times weightOptional + biasOptional))
  $$

  $$
  gradShiftOut =  \sum_{reduce\_axis\_1}gradOut
  $$

  $$
  gradWeightOut =  \sum_{reduce\_axis\_0,1}gradOut \times (1 + scale) \times (input - mean) \times rstd
  $$
  
  $$
  gradBiasOut = \sum_{reduce\_axis\_0,1}gradOut \times (1 + scale)
  $$

  其中，N为进行归一化计算的轴的维度，即归一化轴维度的大小。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAdaLayerNormBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAdaLayerNormBackward”接口执行计算。

```Cpp
aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(
    const aclTensor*    gradOut, 
    const aclTensor*    input, 
    const aclIntArray*  normalizedShape, 
    const aclTensor*    rstd,
    const aclTensor*    mean, 
    const aclTensor*    scale, 
    const aclTensor*    shift, 
    const aclTensor*    weightOptional, 
    const aclTensor*    biasOptional, 
    aclTensor*          gradInputOut, 
    aclTensor*          gradScaleOut, 
    aclTensor*          gradShiftOut,
    aclTensor*          gradWeightOut, 
    aclTensor*          gradBiasOut, 
    uint64_t*           workspaceSize, 
    aclOpExecutor**     executor)
```

```Cpp
aclnnStatus aclnnAdaLayerNormBackward(
  void*          workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor* executor, 
  aclrtStream    stream)
```

## aclnnAdaLayerNormBackwardGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 430px">
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
      <td>gradOut（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向计算的梯度Tensor，对应计算公式中的`gradOut`。</td>
      <td><ul><li>不支持空Tensor。<li>与输入input的数据类型相同。<li>shape为[B, S, H]，其中B支持0-6维。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向计算的首个输入，对应计算公式中的`input`。</td>
      <td><ul><li>不支持空Tensor。<li>与输入gradOut的数据类型相同。<li>shape与gradOut的shape相等，为[B, S, H], 其中B支持0-6维。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>normalizedShape（aclIntArray*）</td>
      <td>输入</td>
      <td>表示需要进行norm计算的维度，对应计算公式中的reduce_axis_2。</td>
      <td><ul><li>公式中的reduce_axis_0对应维度B，reduce_axis_1对应维度S，B和S维度不进行norm计算的维度。<li>在此处值固定为长度为1的数组，数组固定元素为[H], 数组长度小于输入input的shape维度，不支持为空。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>mean（aclTensor*）</td>
      <td>输入</td>
      <td>正向计算的第二个输出，表示input的均值，对应计算公式中的mean。</td>
      <td><ul><li>不支持空Tensor。</li><li>与输入rstd的数据类型相同且位宽不低于输入input的数据类型位宽。</li><li>shape与rstd的shape相等，为[B, S, 1]，最后一维固定为1，其他维度与`input`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输入</td>
      <td>正向计算的第三个输出，表示input的标准差的倒数，对应计算公式中的rstd。</td>
      <td><ul><li>不支持空Tensor。<li>与输入mean的数据类型相同且位宽不低于输入input的数据类型位宽。<li>shape与mean的shape相等，为[B, S, 1]，最后一维固定为1，其他维度与`x`一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scale（aclTensor*）</td>
      <td>输入</td>
      <td>表示自适应缩放参数。对应计算公式中的`scale`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与`gradOut`的数据类型一致。</li><li>shape为[B, H]或[B, 1, H]，其中B支持0-6维，维度数量和大小与`gradOut`中的B保持一致，H与`gradOut`中H维一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>shift（aclTensor*）</td>
      <td>输入</td>
      <td>表示自适应偏移参数。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`gradOut`的数据类型一致。</li><li>shape为[B, H]或[B, 1, H]，其中B支持0-6维，维度数量和大小与`gradOut`中的B保持一致，H与`gradOut`中H维一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示归一化缩放参数。对应计算公式中的`weightOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[H]，H与`gradOut`中H维一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional（aclTensor*）</td>
      <td>输入</td>
      <td>可选输入参数，表示归一化偏移参数。对应计算公式中的`biasOptional`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与可选参数`weightOptional`一致。</li><li>shape为[H]，H与`gradOut`中H维一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradInputOut（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示反向传播的输出梯度，对应计算公式中的`gradInputOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当指针gradInputOut（aclTensor*）不为空时，输出`input`梯度计算结果，且其与输入`input`的数据类型相同。</li><li>shape与`input`的shape相等，为[B, S, H]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradScaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示反向传播自适应缩放系数的梯度，对应计算公式中的`gradWeightOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当指针gradScaleOut（aclTensor*）不为空时，输出`scale`梯度计算结果，且其与输入`input`的数据类型相同。</li><li>shape与`scale`的shape相等，为[B, H]或[B, 1, H]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradShiftOut（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示反向传播自适应偏移系数的梯度，对应计算公式中的`gradBiasOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当指针gradShiftOut（aclTensor*）不为空时，输出`shift`梯度计算结果，且其与输入`input`的数据类型相同。</li><li>shape与`shift`的shape相等，为[B, H]或[B, 1, H]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOut（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示反向传播归一化缩放系数的梯度，对应计算公式中的`gradWeightOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当指针gradWeightOut（aclTensor*）不为空时，输出`weightOptional`梯度计算结果，且其与输入`weightOptional`的数据类型相同。</li><li>shape与`weightOptional`的shape相等，为[H]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBiasOut（aclTensor*）</td>
      <td>输出</td>
      <td>可选输出，表示反向传播归一化偏置系数的梯度，对应计算公式中的`gradBiasOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>当指针gradBiasOut（aclTensor*）不为空时，输出`biasOptional`梯度计算结果，且其与输入`weightOptional`的数据类型相同。</li><li>shape与`gradWeightOut`的shape相等，为[H]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

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
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOut、input、normalizedShape、mean、rstd、scale、shift为空指针。</td>
    </tr>
   <tr>
      <td rowspan="14">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="14">161002</td>
      <td>gradOut、input、mean、rstd、scale、shift、weightOptional（非空时）或biasOptional（非空时）的数据类型不在支持范围内。</td>
    </tr>
    <tr>
      <td>input的维度小于2。</td>
    </tr>
    <tr>
      <td>normalizedShape数组长度不为1。</td>
    </tr>
    <tr>
      <td>input的shape数组与normalizedShape数组右对齐时对应维度不相等。</td>
    </tr>
    <tr>
      <td>gradOut的shape与input的shape不相等。</td>
    </tr>
    <tr>
      <td>mean的shape乘积与input从第0轴到第len(input) - len(normalizedShape)轴的乘积不相等。</td>
    </tr>
    <tr>
      <td>rstd的shape乘积与input从第0根轴到第len(input) - len(normalizedShape)轴的乘积不相等。</td>
    </tr>
    <tr>
      <td>scale的shape不等于[B, 1, H]或[B, H]。</td>
    </tr>
    <tr>
      <td>shift的shape不等于[B, 1, H]或[B, H]。</td>
    </tr>
    <tr>
      <td>gradScaleOut（非空时）shape与scale不相等。</td>
    </tr>
    <tr>
      <td>gradShiftOut（非空时）shape与shift不相等。</td>
    </tr>
    <tr>
      <td>weightOptional非空且shape数组与normalizedShape数组不相等。</td>
    </tr>
    <tr>
      <td>biasOptional非空且shape数组与normalizedShape数组不相等。</td>
    </tr>
    <tr>
      <td> gradOut，input，mean，rstd，scale，shift，weightOptional（非空时），biasOptional（非空时），gradInputOut（非空时），gradScaleOut（非空时），gradShiftOut（非空时），gradWeightOut（非空时），gradBiasOut（非空时）的shape维度大于8维。</td>
    </tr>
  </tbody></table>

## aclnnAdaLayerNormBackward

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th>输入/输出</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>workspace</td>
      <td>输入</td>
      <td>在Device侧申请的workspace内存地址。</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>输入</td>
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAdaLayerNormBackwardGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream。</td>
    </tr>
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnAdaLayerNormBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_ada_layer_norm_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 2, 2};
    std::vector<int64_t> meanShape = {2, 2, 1};
    std::vector<int64_t> scaleShape = {2, 2};
    std::vector<int64_t> shiftShape = {2, 2};
    std::vector<int64_t> normShape = {2};
    void* dyDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* shiftDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* dsDeviceAddr = nullptr;
    void* dtDeviceAddr = nullptr;
    void* dwDeviceAddr = nullptr;
    void* dbDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclIntArray* norm = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* shift = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* ds = nullptr;
    aclTensor* dt = nullptr;
    aclTensor* dw = nullptr;
    aclTensor* db = nullptr;
    std::vector<float> dyHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<float> xHostData = {2, 3, 4, 5, 2, 3, 4, 5};
    std::vector<int64_t> normData = {2};
    std::vector<float> meanHostData = {2, 3, 4, 5};
    std::vector<float> rstdHostData = {4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 1, 2, 2};
    std::vector<float> shiftHostData = {2, 2, 1, 1};
    std::vector<float> weightHostData = {1, 1};
    std::vector<float> biasHostData = {0, 0};
    std::vector<float> outHostData(8, 0);
    std::vector<float> dsHostData(4, 0);
    std::vector<float> dtHostData(4, 0);
    std::vector<float> dwHostData(2, 0);
    std::vector<float> dbHostData(2, 0);

    // 创建dy aclTensor
    ret = CreateAclTensor(dyHostData, xShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建normalizedShape aclIntArray
    norm = aclCreateIntArray(normData.data(), 1);
    CHECK_RET(ret == ACL_SUCCESS, return false);
    // 创建rstd aclTensor
    ret = CreateAclTensor(rstdHostData, meanShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mean aclTensor
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建shift aclTensor
    ret = CreateAclTensor(shiftHostData, shiftShape, &shiftDeviceAddr, aclDataType::ACL_FLOAT, &shift);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, normShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, normShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, xShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建ds aclTensor
    ret = CreateAclTensor(dsHostData, scaleShape, &dsDeviceAddr, aclDataType::ACL_FLOAT, &ds);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dt aclTensor
    ret = CreateAclTensor(dtHostData, shiftShape, &dtDeviceAddr, aclDataType::ACL_FLOAT, &dt);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建dw aclTensor
    ret = CreateAclTensor(dwHostData, normShape, &dwDeviceAddr, aclDataType::ACL_FLOAT, &dw);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建db aclTensor
    ret = CreateAclTensor(dbHostData, normShape, &dbDeviceAddr, aclDataType::ACL_FLOAT, &db);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAdaLayerNormBackward第一段接口
    ret = aclnnAdaLayerNormBackwardGetWorkspaceSize(
        dy, x, norm, rstd, mean, scale, shift, weight, bias, out, ds, dt, dw, db, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnAdaLayerNormBackward第二段接口
    ret = aclnnAdaLayerNormBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnLayerNormBackward failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(xShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("out result[%ld] is: %f\n", i, resultData[i]);
    }

    auto size1 = GetShapeSize(scaleShape);
    std::vector<float> resultData1(size1, 0);
    ret = aclrtMemcpy(
        resultData1.data(), resultData1.size() * sizeof(resultData1[0]), dsDeviceAddr, size1 * sizeof(resultData1[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("ds result[%ld] is: %f\n", i, resultData1[i]);
    }

    auto size2 = GetShapeSize(shiftShape);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(
        resultData2.data(), resultData2.size() * sizeof(resultData2[0]), dtDeviceAddr, size2 * sizeof(resultData2[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("dt result[%ld] is: %f\n", i, resultData2[i]);
    }

    auto size3 = GetShapeSize(normShape);
    std::vector<float> resultData3(size3, 0);
    ret = aclrtMemcpy(
        resultData3.data(), resultData3.size() * sizeof(resultData3[0]), dwDeviceAddr, size3 * sizeof(resultData3[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size3; i++) {
        LOG_PRINT("dw result[%ld] is: %f\n", i, resultData3[i]);
    }

    auto size4 = GetShapeSize(normShape);
    std::vector<float> resultData4(size4, 0);
    ret = aclrtMemcpy(
        resultData4.data(), resultData4.size() * sizeof(resultData4[0]), dbDeviceAddr, size4 * sizeof(resultData4[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size4; i++) {
        LOG_PRINT("db result[%ld] is: %f\n", i, resultData4[i]);
    }

    // 6. 释放aclTensor、aclIntArray和aclBoolArray，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(x);
    aclDestroyIntArray(norm);
    aclDestroyTensor(mean);
    aclDestroyTensor(rstd);
    aclDestroyTensor(scale);
    aclDestroyTensor(shift);
    aclDestroyTensor(weight);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(ds);
    aclDestroyTensor(dt);
    aclDestroyTensor(dw);
    aclDestroyTensor(db);

    // 7. 释放device 资源
    aclrtFree(dyDeviceAddr);
    aclrtFree(xDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(shiftDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(dsDeviceAddr);
    aclrtFree(dtDeviceAddr);
    aclrtFree(dwDeviceAddr);
    aclrtFree(dbDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
