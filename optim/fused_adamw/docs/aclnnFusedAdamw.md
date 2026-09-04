# aclnnFusedAdamw

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/optim/fused_adamw)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：实现融合AdamW优化器功能，支持多组参数列表（TensorList）一次调用完成参数更新、动量更新等操作。

- 计算公式：

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t}
  $$

  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$

  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}}
  $$

  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}}
  $$

  $$
  if(amsgrad) : maxGradNorm = max(maxGradNorm,\hat{v}_{t})
  $$

  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnFusedAdamwGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedAdamw”接口执行计算。

```cpp
aclnnStatus aclnnFusedAdamwGetWorkspaceSize(
    const aclTensorList* paramsRef,
    const aclTensorList* grads,
    const aclTensorList* expAvgsRef,
    const aclTensorList* expAvgSqsRef,
    const aclTensorList* maxExpAvgSqsRef,
    const aclTensorList* stateSteps,
    const aclTensor*     gradScaleOptional,
    const aclTensor*     foundInfOptional,
    double                lr,
    double                beta1,
    double                beta2,
    double                weightDecay,
    double                eps,
    bool                 amsgrad,
    bool                 maximize,
    uint64_t*            workspaceSize,
    aclOpExecutor**      executor)
```

```cpp
aclnnStatus aclnnFusedAdamw(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnFusedAdamwGetWorkspaceSize

- **参数说明：**

    <table style="undefined;table-layout: fixed; width: 1520px"><colgroup>
    <col style="width: 230px">
    <col style="width: 120px">
    <col style="width: 330px">
    <col style="width: 220px">
    <col style="width: 230px">
    <col style="width: 115px">
    <col style="width: 130px">
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
        <td>paramsRef（aclTensorList*）</td>
        <td>输入/输出</td>
        <td>待计算的权重列表，公式中的θ。</td>
        <td>-</td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>grads（aclTensorList*）</td>
        <td>输入</td>
        <td>梯度数据列表，公式中的g<sub>t</sub>。</td>
        <td>-</td>
        <td>与“paramsRef”参数一致。</td>
        <td>ND</td>
        <td>与“paramsRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expAvgsRef（aclTensorList*）</td>
        <td>输入/输出</td>
        <td>一阶动量列表，公式中的m。</td>
        <td>-</td>
        <td>与“paramsRef”参数一致。</td>
        <td>ND</td>
        <td>与“paramsRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>expAvgSqsRef（aclTensorList*）</td>
        <td>输入/输出</td>
        <td>二阶动量列表，公式中的v，不能为负数。</td>
        <td>-</td>
        <td>与“paramsRef”参数一致。</td>
        <td>ND</td>
        <td>与“paramsRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>maxExpAvgSqsRef（aclTensorList*）</td>
        <td>输入/输出</td>
        <td>保存最大二阶矩列表，与更新后的expAvgSqsRef比较后取最大值输出。</td>
        <td>此参数在amsgrad参数为true时必选，在amsgrad参数为false时可选。</td>
        <td>与“paramsRef”参数一致。</td>
        <td>ND</td>
        <td>与“paramsRef”参数一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>stateSteps（aclTensorList*）</td>
        <td>输入</td>
        <td>迭代次数列表，公式中的t，需要大于0。</td>
        <td>每个Tensor元素个数为1。</td>
        <td>INT64、FLOAT32</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>gradScaleOptional（aclTensor*）</td>
        <td>输入</td>
        <td>可选输入，梯度缩放因数。</td>
        <td>当foundInfOptional=1时，梯度清零并停止参数更新。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>foundInfOptional（aclTensor*）</td>
        <td>输入</td>
        <td>可选输入，标识是否出现Inf/NaN。</td>
        <td>等于1时停止更新。</td>
        <td>FLOAT</td>
        <td>ND</td>
        <td>-</td>
        <td>x</td>
      </tr>
      <tr>
        <td>lr（double）</td>
        <td>输入</td>
        <td>学习率，公式中的η。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta1（double）</td>
        <td>输入</td>
        <td>β<sub>1</sub>参数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>beta2（double）</td>
        <td>输入</td>
        <td>β<sub>2</sub>参数。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>weightDecay（double）</td>
        <td>输入</td>
        <td>权重衰减系数，公式中的λ。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>eps（double）</td>
        <td>输入</td>
        <td>防止除数为0。</td>
        <td>-</td>
        <td>DOUBLE</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>amsgrad（bool）</td>
        <td>输入</td>
        <td>是否使用算法的AMSGrad变量。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>maximize（bool）</td>
        <td>输入</td>
        <td>是否最大化参数。</td>
        <td>-</td>
        <td>BOOL</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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
    </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
    </colgroup>
    <thead>
      <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
      </tr></thead>
    <tbody>
      <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的paramsRef、grads、expAvgsRef、expAvgSqsRef、stateSteps是空指针时。</td>
      </tr>
      <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>paramsRef、grads、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef的数据类型不在支持的范围内时。</td>
      </tr>
      <tr>
      <td>paramsRef、grads、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef的数据类型不同时。</td>
      </tr>
      <tr>
      <td>grads、expAvgsRef、expAvgSqsRef和paramsRef的shape不一致时。</td>
      </tr>
      <tr>
      <td>当amsgrad为true时，maxExpAvgSqsRef和paramsRef的shape不一致时。</td>
      </tr>
      <tr>
      <td>stateSteps中每个Tensor的元素个数不为1时。</td>
      </tr>
    </tbody></table>

## aclnnFusedAdamw

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
  <col style="width: 200px">
  <col style="width: 162px">
  <col style="width: 882px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedAdamwGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 输入张量中paramsRef、grads、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef的数据类型必须一致，且数据类型支持FLOAT16、BFLOAT16、FLOAT32。
- 输入张量中paramsRef、grads、expAvgsRef、expAvgSqsRef、maxExpAvgSqsRef对应位置的shape必须保持一致。
- stateSteps支持INT64、FLOAT32，每个Tensor元素个数为1。
- amsgrad为false时，maxExpAvgSqsRef可为空；amsgrad为true时，maxExpAvgSqsRef必选且shape需与paramsRef一致。
- 确定性计算：
  - aclnnFusedAdamw默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_adamw.h"
#include <iostream>
#include <vector>

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

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return );
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> gradsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> expavgsqsHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
    std::vector<float> maxexpavgsqsHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> stepsHostData1 = {3};


    std::vector<float> gradScaleOptionalHostData = {2};

    std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
    std::vector<float> gradsHostData2 = {0.9, 1.0, 1.1, 1.2};
    std::vector<float> expavgsHostData2 = {-1, -2, -3, -4};
    std::vector<float> expavgsqsHostData2 = {1, 2, 3, 4};
    std::vector<float> maxexpavgsqsHostData2 = {2, 2, 2, 2};
    std::vector<float> stepsHostData2 = {4};


    std::vector<int64_t> inputShape1 = {2, 2, 2};
    std::vector<int64_t> inputShape2 = {2, 2};
    std::vector<int64_t> scalarShape = {1};

    void* paramsRef1DeviceAddr = nullptr;
    void* grads1DeviceAddr = nullptr;
    void* expavgs1DeviceAddr = nullptr;
    void* expavgsqs1DeviceAddr = nullptr;
    void* maxexpavgsqs1DeviceAddr = nullptr;
    void* steps1DeviceAddr = nullptr;

    void* paramsRef2DeviceAddr = nullptr;
    void* grads2DeviceAddr = nullptr;
    void* expavgs2DeviceAddr = nullptr;
    void* expavgsqs2DeviceAddr = nullptr;
    void* maxexpavgsqs2DeviceAddr = nullptr;
    void* steps2DeviceAddr = nullptr;

    void* gradScaleOptionalDeviceAddr = nullptr;

    aclTensor* paramsRef1 = nullptr;
    aclTensor* grads1 = nullptr;
    aclTensor* expavgs1 = nullptr;
    aclTensor* expavgsqs1 = nullptr;
    aclTensor* maxexpavgsqs1 = nullptr;
    aclTensor* steps1 = nullptr;

    aclTensor* paramsRef2 = nullptr;
    aclTensor* grads2 = nullptr;
    aclTensor* expavgs2 = nullptr;
    aclTensor* expavgsqs2 = nullptr;
    aclTensor* maxexpavgsqs2 = nullptr;
    aclTensor* steps2 = nullptr;

    aclTensor* gradScaleOptional = nullptr;

    aclTensor* foundInfOptional = nullptr;

    ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData1, inputShape1, &grads1DeviceAddr, aclDataType::ACL_FLOAT, &grads1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData1, inputShape1, &expavgs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData1, inputShape1, &expavgsqs1DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData1, inputShape1, &maxexpavgsqs1DeviceAddr, aclDataType::ACL_FLOAT, &maxexpavgsqs1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData1, inputShape1, &steps1DeviceAddr, aclDataType::ACL_FLOAT, &steps1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gradsHostData2, inputShape2, &grads2DeviceAddr, aclDataType::ACL_FLOAT, &grads2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsHostData2, inputShape2, &expavgs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expavgsqsHostData2, inputShape2, &expavgsqs2DeviceAddr, aclDataType::ACL_FLOAT, &expavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxexpavgsqsHostData2, inputShape2, &maxexpavgsqs2DeviceAddr, aclDataType::ACL_FLOAT, &maxexpavgsqs2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(stepsHostData2, inputShape2, &steps2DeviceAddr, aclDataType::ACL_FLOAT, &steps2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT,
                          &gradScaleOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
    std::vector<aclTensor*> gradsListData = {grads1, grads2};
    std::vector<aclTensor*> expavgsListData = {expavgs1, expavgs2};
    std::vector<aclTensor*> expavgsqsListData = {expavgsqs1, expavgsqs2};
    std::vector<aclTensor*> maxexpavgsqsData = {maxexpavgsqs1, maxexpavgsqs2};
    std::vector<aclTensor*> stepsListData = {steps1, steps2};
    aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
    aclTensorList* gradsList = aclCreateTensorList(gradsListData.data(), gradsListData.size());
    aclTensorList* expavgsList = aclCreateTensorList(expavgsListData.data(), expavgsListData.size());
    aclTensorList* expavgsqsListList = aclCreateTensorList(expavgsqsListData.data(), expavgsqsListData.size());
    aclTensorList* maxexpavgsqsList = aclCreateTensorList(maxexpavgsqsData.data(), maxexpavgsqsData.size());
    aclTensorList* stepsList = aclCreateTensorList(stepsListData.data(), stepsListData.size());

    double lr = 0.001f;
    double beta1 = 0.9f;
    double beta2 = 0.999f;
    double weightDecay = 0.0f;
    double eps = 1e-8;
    bool amsgrad = true;
    bool maximize = false;

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    ret = aclnnFusedAdamwGetWorkspaceSize(paramsRefList, gradsList, expavgsList, expavgsqsListList, maxexpavgsqsList,
                                        stepsList, gradScaleOptional, foundInfOptional, lr, beta1, beta2, weightDecay,
                                        eps, amsgrad, maximize, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamwGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    ret = aclnnFusedAdamw(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedAdamw failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    LOG_PRINT("====== Tensor 1 paramsRef1 results ======\n");
    PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
    LOG_PRINT("====== Tensor 2 expavgs1 results ======\n");
    PrintOutResult(inputShape1, &expavgs1DeviceAddr);
    LOG_PRINT("====== Tensor 3 expavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &expavgsqs1DeviceAddr);
    LOG_PRINT("===== Tensor 4 maxexpavgsqs1 results ======\n");
    PrintOutResult(inputShape1, &maxexpavgsqs1DeviceAddr);

    LOG_PRINT("====== Tensor 5 paramsRef results ======\n");
    PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
    LOG_PRINT("====== Tensor 6 expavgs2 results ======\n");
    PrintOutResult(inputShape2, &expavgs2DeviceAddr);
    LOG_PRINT("====== Tensor 7 expavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &expavgsqs2DeviceAddr);
    LOG_PRINT("====== Tensor 8 maxexpavgsqs2 results ======\n");
    PrintOutResult(inputShape2, &maxexpavgsqs2DeviceAddr);

    aclDestroyTensorList(paramsRefList);
    aclDestroyTensorList(gradsList);
    aclDestroyTensorList(expavgsList);
    aclDestroyTensorList(expavgsqsListList);
    aclDestroyTensorList(maxexpavgsqsList);
    aclDestroyTensorList(stepsList);
    aclDestroyTensor(gradScaleOptional);
    aclDestroyTensor(foundInfOptional);

    aclrtFree(paramsRef1DeviceAddr);
    aclrtFree(grads1DeviceAddr);
    aclrtFree(expavgs1DeviceAddr);
    aclrtFree(expavgsqs1DeviceAddr);
    aclrtFree(maxexpavgsqs1DeviceAddr);
    aclrtFree(steps1DeviceAddr);

    aclrtFree(paramsRef2DeviceAddr);
    aclrtFree(grads2DeviceAddr);
    aclrtFree(expavgs2DeviceAddr);
    aclrtFree(expavgsqs2DeviceAddr);
    aclrtFree(maxexpavgsqs2DeviceAddr);
    aclrtFree(steps2DeviceAddr);

    aclrtFree(gradScaleOptionalDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    ret = aclrtDestroyStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("destroy stream failed. ERROR: %d\n", ret); return ret);
    ret = aclrtResetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("reset device failed. ERROR: %d\n", ret); return ret);
    ret = aclFinalize();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("finalize acl failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
