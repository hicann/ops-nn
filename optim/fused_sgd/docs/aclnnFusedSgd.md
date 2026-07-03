# aclnnFusedSgd

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 算子功能：实现fusedSgd算子。将传统 SGD 更新过程中原本分散的多个细粒度操作（如梯度缩放、权重衰减、动量更新、参数赋值等）融合为单个NPU Kernel执行。
- 计算公式：

  $$
  \begin{aligned} 
  &\tilde{g}_t = \begin{cases} 
  g_t / s & s \neq \text{None} \\ 
  g_t & \text{otherwise} 
  \end{cases} \\ 

  &\hat{g}_t = \begin{cases} 
  -\tilde{g}_t & \text{maximize} \\ 
  \tilde{g}_t & \text{otherwise} 
  \end{cases} \\ 

  &\bar{g}_t = \hat{g}_t + weightDecay \cdot \theta_t \\ 

  &v_{t+1} = \begin{cases} 
  \bar{g}_t & \text{first step} \\ 
  \mu v_t + (1-dampening)\bar{g}_t & \text{otherwise} 
  \end{cases} \\ 

  &g_t^{\text{final}} = \begin{cases} 
  \bar{g}_t + \mu v_{t+1} & \text{nesterov} \\ 
  v_{t+1} & v_t \neq \text{None} \\ 
  \bar{g}_t & \text{otherwise}
  \end{cases} \\ 

  &\theta_{t+1} = \theta_t - lr \cdot g_t^{\text{final}} \\
  &g_{t+1} = \tilde{g}_t

  \end{aligned}
  $$

## 函数原型

算子执行接口为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedSgdGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnFusedSgd”接口执行计算。

```cpp
aclnnStatus aclnnFusedSgdGetWorkspaceSize(
    const aclTensorList   *paramsRef,
    const aclTensorList   *gradsRef,
    const aclTensorList   *momentumBufferListOptionalRef,
    const aclTensor       *gradScaleOptional,
    float                  weightDecay,
    float                  momentum,
    float                  lr,
    float                  dampening,
    bool                   nesterov,
    bool                   maximize,
    bool                   isFirstStep,
    uint64_t              *workspaceSize,
    aclOpExecutor        **executor);
```

```cpp
aclnnStatus aclnnFusedSgd(
  void              *workspace,
  uint64_t           workspace_size,
  aclOpExecutor     *executor,
  const aclrtStream  stream)
```

## aclnnFusedSgdGetWorkspaceSize

- **参数说明**
  
  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1497px"><colgroup> 
     <col style="width: 150px"> 
     <col style="width: 120px"> 
     <col style="width: 300px"> 
     <col style="width: 330px"> 
     <col style="width: 212px"> 
     <col style="width: 100px">  
     <col style="width: 140px">  
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
      <td>paramsRef</td>
      <td>输入/输出</td>
      <td>更新参数。</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>小于等于8维</td>
      <td>√</td>
      </tr>
      <tr>
      <td>gradsRef</td>
      <td>输入/输出</td>
      <td>更新梯度</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>小于等于8维</td>
      <td>√</td>
      </tr>
      <tr>
      <td>momentumBufferListOptionalRef</td>
      <td>输入/输出</td>
      <td>更新动量</td>
      <td>支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>小于等于8维</td>
      <td>√</td>
      </tr>
      <tr>
      <td>gradScaleOptional</td>
      <td>输入</td>
      <td>梯度缩放值</td>
      <td>支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>shape等于[1]</td>
      <td>√</td>
      </tr>
      <tr>
      <td>weightDecay</td>
      <td>属性</td>
      <td>权重衰减值。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>momentum</td>
      <td>属性</td>
      <td>动量值。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>lr</td>
      <td>属性</td>
      <td>学习率。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>dampening</td>
      <td>属性</td>
      <td>动量的阻尼系数。</td>
      <td>-</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>nesterov</td>
      <td>属性</td>
      <td>是否启用 Nesterov 动量。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>maximize</td>
      <td>属性</td>
      <td>是否为最大化目标函数。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>isFirstStep</td>
      <td>属性</td>
      <td>是否第一步更新。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
      <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
      </tr>
      <tr>
      <td>executor</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>1</td>
      <td>-</td>
      </tr>
    </tbody></table>
    </div>
- **返回值**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 250px">
    <col style="width: 100px">
    <col style="width: 650px">
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
      <td>传入的paramsRef、gradsRef、momentumBufferListOptionalRef、grad_scale是空指针。</td>
      </tr>
      <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td><ul><li>paramsRef、gradsRef、momentumBufferListOptionalRef、grad_scale的数据类型不在支持的范围之内。</li>
      <li>paramsRef、gradsRef与momentumBufferListOptionalRef及其中各个tensor的数据类型不同。</li>
      <li>paramsRef、gradsRef与momentumBufferListOptionalRef中tensor的shape维度小于等于8。</li>
      <li>paramsRef、gradsRef与momentumBufferListOptionalRef（momentumBufferListOptionalRef不为空时）中相同索引tensor的shape不同。</li>
      <li>paramsRef、gradsRef与momentumBufferListOptionalRef（momentumBufferListOptionalRef不为空时）中有空指针。</li>
      <li>lr、momentum、weightDecay、dampening的值小于0。</li>
      <li>gradScale等于0。</li>
      </ul></td>
      </tr>
  </tbody>
    </table>
    </div>

## aclnnFusedSgd

* ​**参数说明**​：

<div style="overflow-x: auto;">
    <table style="undefined;table-layout: fixed; width: 900px"><colgroup>
    <col style="width: 150px">
    <col style="width: 100px">
    <col style="width: 650px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedSgd获取。</td>
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
    </div>

* ​**返回值**​：
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

1. paramsRef、gradsRef、momentumBufferListOptionalRef、grad_scale的数据类型在支持的范围之内。
2. paramsRef、gradsRef与momentumBufferListOptionalRef及其中各个tensor具有相同的数据类型。
3. paramsRef、gradsRef与momentumBufferListOptionalRef中tensor的shape维度小于等于8。
4. paramsRef、gradsRef与momentumBufferListOptionalRef（momentumBufferListOptionalRef不为空时）中相同索引tensor的shape相同。
5. paramsRef、gradsRef与momentumBufferListOptionalRef（momentumBufferListOptionalRef不为空时）中不能有空指针。
6. lr、momentum、weightDecay、dampening的值大于等于0。
7. gradScale不等于0。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_sgd.h"
#include <iostream>
#include <vector>

#define CHECK_RET(cond, return_expr)                                           \
  do {                                                                         \
    if (!(cond)) {                                                             \
      return_expr;                                                             \
    }                                                                          \
  } while (0)

#define LOG_PRINT(message, ...)                                                \
  do {                                                                         \
    printf(message, ##__VA_ARGS__);                                            \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

void PrintOutResult(std::vector<int64_t> &shape, void **deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(
      resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
      return );
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData,
                    const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  std::vector<float> paramsRefHostData1 = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> gradsRefHostData1 = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> momentumHostData1 = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> paramsRefHostData2 = {9, 10, 11, 12};
  std::vector<float> gradsRefHostData2 = {0.9, 1.0, 1.1, 1.2};
  std::vector<float> momentumHostData2 = {0, 0, 0, 0};
  std::vector<float> gradScaleOptionalHostData = {1.0};
  std::vector<int64_t> inputShape1 = {2, 2, 2};
  std::vector<int64_t> inputShape2 = {2, 2};
  std::vector<int64_t> scalarShape = {1};

  void *paramsRef1DeviceAddr = nullptr;
  void *gradsRef1DeviceAddr = nullptr;
  void *momentum1DeviceAddr = nullptr;
  void *paramsRef2DeviceAddr = nullptr;
  void *gradsRef2DeviceAddr = nullptr;
  void *momentum2DeviceAddr = nullptr;
  void *gradScaleOptionalDeviceAddr = nullptr;

  aclTensor *paramsRef1 = nullptr;
  aclTensor *gradsRef1 = nullptr;
  aclTensor *momentum1 = nullptr;
  aclTensor *paramsRef2 = nullptr;
  aclTensor *gradsRef2 = nullptr;
  aclTensor *momentum2 = nullptr;
  aclTensor *gradScaleOptional = nullptr;

  ret = CreateAclTensor(paramsRefHostData1, inputShape1, &paramsRef1DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData1, inputShape1, &gradsRef1DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData1, inputShape1, &momentum1DeviceAddr, aclDataType::ACL_FLOAT, &momentum1);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(paramsRefHostData2, inputShape2, &paramsRef2DeviceAddr, aclDataType::ACL_FLOAT, &paramsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradsRefHostData2, inputShape2, &gradsRef2DeviceAddr, aclDataType::ACL_FLOAT, &gradsRef2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(momentumHostData2, inputShape2, &momentum2DeviceAddr, aclDataType::ACL_FLOAT, &momentum2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(gradScaleOptionalHostData, scalarShape, &gradScaleOptionalDeviceAddr, aclDataType::ACL_FLOAT, &gradScaleOptional);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<aclTensor*> paramsRefListData = {paramsRef1, paramsRef2};
  std::vector<aclTensor*> gradsRefListData = {gradsRef1, gradsRef2};
  std::vector<aclTensor*> momentumListData = {momentum1, momentum2};
  aclTensorList* paramsRefList = aclCreateTensorList(paramsRefListData.data(), paramsRefListData.size());
  aclTensorList* gradsRefList = aclCreateTensorList(gradsRefListData.data(), gradsRefListData.size());
  aclTensorList* momentumList = aclCreateTensorList(momentumListData.data(), momentumListData.size());

  float weightDecay = 0.01f;
  float momentumVal = 0.9f;
  float lr = 0.001f;
  float dampening = 0.0f;
  bool nesterov = false;
  bool maximize = false;
  bool isFirstStep = true;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  ret = aclnnFusedSgdGetWorkspaceSize(paramsRefList, gradsRefList, momentumList, gradScaleOptional,
                                       weightDecay, momentumVal, lr, dampening,
                                       nesterov, maximize, isFirstStep,
                                       &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnFusedSgdGetWorkspaceSize failed. ERROR: %d\n", ret);
      return ret);

  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  ret = aclnnFusedSgd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnFusedSgd failed. ERROR: %d\n", ret);
            return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  LOG_PRINT("====== Tensor 1 paramsRef results ======\n");
  PrintOutResult(inputShape1, &paramsRef1DeviceAddr);
  LOG_PRINT("====== Tensor 1 gradsRef results ======\n");
  PrintOutResult(inputShape1, &gradsRef1DeviceAddr);
  LOG_PRINT("------ Momentum buffer 1 ------\n");
  PrintOutResult(inputShape1, &momentum1DeviceAddr);
  LOG_PRINT("====== Tensor 2 paramsRef results ======\n");
  PrintOutResult(inputShape2, &paramsRef2DeviceAddr);
  LOG_PRINT("====== Tensor 2 gradsRef results ======\n");
  PrintOutResult(inputShape2, &gradsRef2DeviceAddr);
  LOG_PRINT("------ Momentum buffer 2 ------\n");
  PrintOutResult(inputShape2, &momentum2DeviceAddr);

  aclDestroyTensorList(paramsRefList);
  aclDestroyTensorList(gradsRefList);
  aclDestroyTensorList(momentumList);
  aclDestroyTensor(gradScaleOptional);

  aclrtFree(paramsRef1DeviceAddr);
  aclrtFree(gradsRef1DeviceAddr);
  aclrtFree(momentum1DeviceAddr);
  aclrtFree(paramsRef2DeviceAddr);
  aclrtFree(gradsRef2DeviceAddr);
  aclrtFree(momentum2DeviceAddr);
  aclrtFree(gradScaleOptionalDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  ret = aclrtDestroyStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("destroy stream failed. ERROR: %d\n", ret);
            return ret);
  ret = aclrtResetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("reset device failed. ERROR: %d\n", ret);
            return ret);
  ret = aclFinalize();
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("finalize acl failed. ERROR: %d\n", ret);
            return ret);
  return 0;
}

```
