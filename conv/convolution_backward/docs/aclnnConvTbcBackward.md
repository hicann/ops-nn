# aclnnConvTbcBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 接口功能：用于计算时序卷积的反向传播

- 计算公式：

  假定输入Conv_tbc正向的输入$input$的shape是$(H_{\text{in}},N,C_{\text{in}})$，输出梯度$gradOutput$的shape是$(H_{\text{out}},N,C_{\text{out}})$，卷积核$weight$的shape是$(K,C_{\text{in}},C_{\text{out}})$，偏置$bias$的shape为$(C_{\text{out}})$。

  $$
  H_{out} = {H_{in} + 2*pad + 1 - K}
  $$

  输入张量input的梯度输出$gradInput_{(t,b,c)}$将被表示为：

  $$
  gradInput_{(t,b,c)} = \sum_{k=0}^{K-1} gradOutput_{(t+k,b,d)} \cdot   weight_{(k,c,d)}
  $$

  卷积核weight的梯度输出$gradWeight_{(t,b,c)}$将被表示为：

  $$
  gradWeight_{(t,b,c)} = \sum_{k=0}^{K-1} gradOutput_{(t+k,b,d)} \cdot input_{(k,c,d)}
  $$

  偏置bias的梯度输出$gradBias$将被表示为：

  $$
  gradBias_j = \sum_{i=1}^{N}\sum_{t=1}^{H_{\text{out}}} gradOutput_{i,t,j}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnConvTbcBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvTbcBackward”接口执行计算。

```cpp
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(
    const aclTensor*     self, 
    const aclTensor*     input, 
    const aclTensor*     weight, 
    const aclTensor*     bias, 
    int64_t              pad, 
    int8_t               cubeMathType, 
    aclTensor*           gradInput, 
    aclTensor*           gradWeight, 
    aclTensor*           gradBias, 
    uint64_t*            workspaceSize, 
    aclOpExecutor**      executor)
```

```cpp
aclnnStatus aclnnConvTbcBackward(
    void*                workspace, 
    uint64_t             workspaceSize, 
    aclOpExecutor*       executor, 
    const aclrtStream    stream)
```

## aclnnConvTbcBackwardGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1567px"><colgroup>
    <col style="width:170px">
    <col style="width:120px">
    <col style="width:300px">
    <col style="width:330px">
    <col style="width:212px">
    <col style="width:100px">
    <col style="width:190px">
    <col style="width:145px">
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
     <td>self</td>
     <td>输入</td>
     <td>计算公式中的gradOutput。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>shape可用公式一表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>参见<a href="#约束说明" target="_blank">约束说明</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>input</td>
     <td>输入</td>
     <td>计算公式中的input。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>shape可用公式二表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>参见<a href="#约束说明" target="_blank">约束说明</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>weight</td>
     <td>输入</td>
     <td>计算公式中的weight。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>shape可用公式三表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>参见<a href="#约束说明" target="_blank">约束说明</a></td>
     <td>√</td>
    </tr>
    <tr>
     <td>bias</td>
     <td>输入</td>
     <td>计算公式中的bias。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>shape可用公式四表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>√</td>
    </tr>
    <tr>
     <td>pad</td>
     <td>输入</td>
     <td>表示T维度上左右填充的个数。</td>
     <td>
       <ul><li>其值应该在[0,255]的范围内。</li></ul>
     </td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>cubeMathType</td>
     <td>输入</td>
     <td>用于判断Cube单元应该使用哪种计算逻辑进行运算。</td>
     <td>
       <ul><li>INT8类型的枚举值，支持的枚举值如下：</li>
       <ul><li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
       <li>1：ALLOW_FP32_DOWN_PRECISION，允许将输入数据降精度计算。</li>
       <li>2：USE_FP16，允许转换为数据类型FLOAT16进行计算。</li>
       <li>3：USE_HF32，允许转换为数据类型HFLOAT32计算。</li>
     </td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradInput</td>
     <td>输出</td>
     <td>计算公式中的gradInput。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与input类型一致。</li>
       <li>shape可用公式五表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradWeight</td>
     <td>输出</td>
     <td>计算公式中的gradWeight。</td>
     <td>
       <ul><li>支持空Tensor。</li>
       <li>数据类型与weight类型一致。</li>
       <li>shape可用公式六表示（见表格下方说明）。</li>
     </td>
     <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>gradBias</td>
     <td>输出</td>
     <td>计算公式中的gradBias。</td>
     <td>
      <ul><li>支持空Tensor。</li>
      <li>数据类型与bias类型一致。</li>
      <li>shape可用公式七表示（见表格下方说明）。</li>
    </td>
     <td>FLOAT、FLOAT16、BFLOAT16</td>
     <td>ND、NCL</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>workspaceSize</td>
     <td>输出</td>
     <td>返回需要在Device侧申请的workspace大小。</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    <tr>
     <td>executor</td>
     <td>输出</td>
     <td>返回op执行器，包含了算子计算流程。</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>-</td>
     <td>×</td>
    </tr>
    </tbody>
  </table> 

      - 不支持BFLOAT16、HIFLOAT8、FLOAT8_E4M3FN。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
      - 不支持HIFLOAT8、FLOAT8_E4M3FN。
      - 不支持空Tensor。
      - 只有在gradWeight参数中，才支持HIFLOAT8、FLOAT8_E4M3FN，在其他参数中不支持HIFLOAT8、FLOAT8_E4M3FN。
  - 公式一：$(H_{\text{out}},N,C_{\text{out}})$
  - 公式二：$(H_{\text{in}},N,C_{\text{in}})$
  - 公式三：$(K,C_{\text{in}},C_{\text{out}})$
  - 公式四：$(C_{\text{out}})$
  - 公式五：$(H_{\text{in}},N,C_{\text{in}})$
  - 公式六：$(K,C_{\text{in}},C_{\text{out}})$
  - 公式七：$(C_{\text{out}})$
  - cubeMathType说明：
      - 枚举值为0
      - 枚举值为1
          - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入是FLOAT，处理器转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
      - 枚举值为2
          - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入是BFLOAT16时不支持该选项。
      - 枚举值为3
          - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入是FLOAT，转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。


- **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。  
    第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:650px">
    </colgroup>
    <thead>
     <tr>
       <th>返回值</th>
       <th>错误码</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入指针类型入参是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>self，input，weight，bias，gradInput，gradWeight，gradBias的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self，input，weight，bias数据类型不一致。</td>
    </tr>
    <tr>
      <td>gradInput，gradWeight，gradBias的shape不满足infershape结果。</td>
    </tr>
    <tr>
      <td>gradInput，gradWeight，gradBias的shape中存在小于0的值。</td>
    </tr>
    <tr>
      <td>self，input，weight的dim不为3。</td>
    </tr>
    <tr>
      <td>bias的dim不为1。</td>
    </tr>
    <tr>
      <td>input的第三个维度值不等于weight的第2个维度值。</td>
    </tr>
    <tr>
      <td>bias的值不等于weight的第三个维度值。</td>
    </tr>
    <tr>
      <td>pad值不满足要求。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_NULLPTR</td>
      <td>561103</td>
      <td>API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
      </tr>
      </tbody>
  </table>
      

## aclnnConvTbcBackward

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
       <col style="width:100px">
       <col style="width:100px">
       <col style="width:550px">
       </colgroup>
    <thead>
     <tr>
       <th>参数名</th>
       <th>输入/输出</th>
       <th>描述</th>
     </tr>
    </thead>
    <tbody>
     <tr>
       <td>workspace</td>
       <td>输入</td>
       <td>在Device侧申请的workspace内存地址。</td>
     </tr>
     <tr>
       <td>workspaceSize</td>
       <td>输入</td>
       <td>在Device侧申请的workspace大小，由第一段接口aclnnConvTbcBackwardGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

<table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width:200px">
    <col style="width:800px">
    </colgroup>
   <thead>
    <tr>
     <th>约束类型</th>
     <th><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">gradOutput约束</th>   
     <td>
        <ul><li>各个维度的大小应在[1,2147483646]的范围内。</li></ul>
     </td>
   </tr>
   <tr>
     <th scope="row">input约束</th>   
     <td>
        <ul><li>各个维度的大小应在[1,2147483646]的范围内。</li></ul>
     </td>
   </tr>
   <tr>
     <th scope="row">weight约束</th>   
     <td>
        <ul><li>L的大小应该在[1,255]的范围内，其他维度的大小应该在[1,2147483646]的范围内。</li></ul>
     </td>
   </tr>
   </tbody>
</table>


由于硬件资源限制，算子在部分参数取值组合场景下会执行失败，请根据日志信息提示分析并排查问题。若无法解决，请单击[Link](https://www.hiascend.com/support)获取技术支持。


## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream *stream)
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
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
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
    if (shape.size() == 4) {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                                  shape.data(), shape.size(), *deviceAddr);
    } else {
        *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                  shape.data(), shape.size(), *deviceAddr);
    }

    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnConvTbcBackwardTest(int32_t deviceId, aclrtStream &stream)
{
    // 1. 初始化
    auto ret = Init(deviceId, &stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {5, 1, 2};
    std::vector<int64_t> inputShape = {5, 1, 2};
    std::vector<int64_t> weightShape = {1, 2, 2};
    std::vector<int64_t> biasShape = {2};
    const int64_t pad = 0;
    int8_t cubeMathType = 1;

    std::vector<int64_t> gradInputShape = {5, 1, 2};
    std::vector<int64_t> gradWeightShape = {1, 2, 2};
    std::vector<int64_t> gradBiasShape = {2};

    // 创建self aclTensor
    std::vector<float> selfData(GetShapeSize(selfShape), 1);
    aclTensor *self = nullptr;
    void *selfdeviceAddr = nullptr;
    ret = CreateAclTensor(selfData, selfShape, &selfdeviceAddr, aclDataType::ACL_FLOAT, &self);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> selfdeviceAddrPtr(selfdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建input aclTensor
    std::vector<float> inputData(GetShapeSize(inputShape), 1);
    aclTensor *input = nullptr;
    void *inputdeviceAddr = nullptr;
    ret = CreateAclTensor(inputData, inputShape, &inputdeviceAddr, aclDataType::ACL_FLOAT, &input);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> inputDeviceAddrPtr(inputdeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建weight aclTensor
    std::vector<float> weightData(GetShapeSize(weightShape), 1);
    aclTensor *weight = nullptr;
    void *weightDeviceAddr = nullptr;
    ret = CreateAclTensor(weightData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建bias aclTensor
    std::vector<float> biasData(GetShapeSize(biasShape), 1);
    aclTensor *bias = nullptr;
    void *biasDeviceAddr = nullptr;
    ret = CreateAclTensor(biasData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> biasDeviceAddrPtr(biasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradInput aclTensor
    std::vector<float> gradInputData(GetShapeSize(inputShape), 1);
    aclTensor *gradInput = nullptr;
    void *gradInputDeviceAddr = nullptr;
    ret = CreateAclTensor(gradInputData, inputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradInputTensorPtr(gradInput, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradInputDeviceAddrPtr(gradInputDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradWeight aclTensor
    std::vector<float> gradWeightData(GetShapeSize(weightShape), 1);
    aclTensor *gradWeight = nullptr;
    void *gradWeightDeviceAddr = nullptr;
    ret = CreateAclTensor(gradWeightData, weightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradWeightTensorPtr(gradWeight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradWeightDeviceAddrPtr(gradWeightDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 创建gradBias aclTensor
    std::vector<float> gradBiasData(GetShapeSize(gradBiasShape), 1);
    aclTensor *gradBias = nullptr;
    void *gradBiasDeviceAddr = nullptr;
    ret = CreateAclTensor(gradBiasData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> gradBiasTensorPtr(gradBias, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void *)> gradBiasDeviceAddrPtr(gradBiasDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor *executor;
    // 调用aclnnConvTbcBackward第一段接口
    ret = aclnnConvTbcBackwardGetWorkspaceSize(self, input, weight, bias, pad, cubeMathType, gradInput, gradWeight,
                                               gradBias, &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                   return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void *workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }
    // 调用aclnnConvTbcBackward第二段接口
    ret = aclnnConvTbcBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> gradInputResult(size, 0);
    ret = aclrtMemcpy(gradInputResult.data(), gradInputResult.size() * sizeof(gradInputResult[0]), gradInputDeviceAddr,
                      size * sizeof(gradInputResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradInputResult[%ld] is: %f\n", i, gradInputResult[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightResult(size, 0);
    ret = aclrtMemcpy(gradWeightResult.data(), gradWeightResult.size() * sizeof(gradWeightResult[0]), gradWeightDeviceAddr,
                      size * sizeof(gradWeightResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradWeightResult[%ld] is: %f\n", i, gradWeightResult[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasResult(size, 0);
    ret = aclrtMemcpy(gradBiasResult.data(), gradBiasResult.size() * sizeof(gradBiasResult[0]), gradBiasDeviceAddr,
                      size * sizeof(gradBiasResult[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
                   return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("gradBiasResult[%ld] is: %f\n", i, gradBiasResult[i]);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnConvTbcBackwardTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
