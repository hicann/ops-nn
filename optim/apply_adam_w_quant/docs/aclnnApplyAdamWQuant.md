# aclnnApplyAdamWQuant

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |

## 功能说明

- **算子功能：** 对优化器输入的m和v作为索引，取出各自qmap中的值，乘以每个blockSize对应的absmax进行反量化，而后实现adamW优化器功能，更新后的m和v每blockSize中取一个最大值，每blockSize个m和v对应一个absmax，进行一次norm归一化，利用二分法找到对应m和v对应qmap中的索引作为输出，absmax也作为下一轮量化的输入。

- **优化器计算公式：**

  $$
  m_{t}=\beta_{1} m_{t-1}+\left(1-\beta_{1}\right) g_{t} \\
  $$


  $$
  v_{t}=\beta_{2} v_{t-1}+\left(1-\beta_{2}\right) g_{t}^{2}
  $$


  $$
  \hat{m}_{t}=\frac{m_{t}}{1-\beta_{1}^{t}} \\
  $$


  $$
  \hat{v}_{t}=\frac{v_{t}}{1-\beta_{2}^{t}} \\
  $$


  $$
  \theta_{t+1}=\theta_{t}-\frac{\eta}{\sqrt{\hat{v}_{t}}+\epsilon} \hat{m}_{t}-\eta \cdot \lambda \cdot \theta_{t-1}
  $$

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnApplyAdamWQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnApplyAdamWQuant”接口执行计算。

* `aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor *varRef, const aclTensor *grad, aclTensor *mRef, aclTensor *vRef, const aclTensor *qmapM, const aclTensor *qmapV, aclTensor *absmaxMRef, aclTensor *absmaxVRef, const aclTensor *step, double lr, double beta1, double beta2, double weightDecay, double eps, double gnormScale, char *quantModeOptional, int64_t blockSize, uint64_t *workspaceSize, aclOpExecutor **executor);`
* `aclnnStatus aclnnApplyAdamWQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);`

## aclnnApplyAdamWQuantGetWorkspaceSize

- **参数说明：**

  * varRef（const aclTensor*，计算输入/计算输出）：Device侧的aclTensor，公式中的theta，待计算的权重输入同时也是输出。数据类型支持FLOAT、FLOAT16、BFLOAT16。不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * grad（aclTensor*， 计算输入）：Device侧的aclTensor，梯度，公式中的gt，数据类型支持FLOAT16、BFLOAT16、FLOAT32，且与varRef保持一致，shape要求与“varRef”参数一致，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * mRef（aclTensor*， 计算输入/计算输出）：Device侧的aclTensor，adamw优化器公式中m参数量化前的索引值，根据索引导出qmapM中具体的值，数据类型支持UINT8，shape要求与“varRef”参数一致，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * vRef（aclTensor*， 计算输入/计算输出）：Device侧的aclTensor，adamw优化器公式中v参数量化前的索引值，根据索引导出qmapV中具体的值，数据类型支持UINT8，shape要求与“varRef”参数一致，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * qmapM（aclTensor*， 计算输入）：Device侧的aclTensor，量化映射表升序排列，数据类型支持FLOAT32，数据格式要求为ND，shape要求为[256,]，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * qmapV（aclTensor*， 计算输入）：Device侧的aclTensor，量化映射表升序排列，数据类型支持FLOAT32，数据格式要求为ND，shape要求为[256,]，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * absmaxMRef（aclTensor*， 计算输入/计算输出） ：Device侧的aclTensor，计算输入为本次反量化阶段将分位数反归一化（乘以absmaxMRef），计算输出为本次量化和下一轮反量化的参数，数据类型支持FLOAT32，shape要求为每256个m对应一个最值，shape要求为“absmaxMRef.size = m.size/blockSize”，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * absmaxVRef（aclTensor*， 计算输入/计算输出） ：Device侧的aclTensor，计算输入为本次反量化阶段将分位数反归一化（乘以absmaxVRef），计算输出为本次量化和下一轮反量化的参数，数据类型支持FLOAT32，shape要求为每256个v对应一个最值，shape要求为“absmaxVRef.size = m.size/blockSize”，不支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * step（aclTensor*）：Device侧的aclTensor，公式中的t，迭代次数，数据类型支持INT64，shape为[1,]，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
  * lr（float\*, 计算输入）：学习率，公式中的eta，推荐1e-3，1e-5，1e-8，范围0~1，数据类型支持float。
  * beta1（float\*, 计算输入）：adamw优化器公式中beta1参数，推荐0.9，范围0~1，数据类型支持float。
  * beta2（float\*, 计算输入）：adamw优化器公式中beta2参数，推荐0.99，范围0~1，数据类型支持float。
  * weightDecay（float\*, 计算输入）：权重衰减系数，adamw优化器公式中lambda参数，推荐0.999，范围0~1，数据类型支持float。
  * eps（float\*, 计算输入）：adamw优化器公式中epsilon参数，加在分母中用来防止除0，推荐1e-8，数据类型支持float。
  * gnormScale（float\*, 计算输入）： 对输入参数grad进行缩放的参数，推荐0.999，范围0~1，数据类型支持float。
  * blockSize（int64\*, 计算输入）：每个block参与计算的大小，固定为256，数据类型支持int64。
  * quantModeOptional（char\*, 计算输入）：保留参数。
  * workspaceSize（uint64_t\*, 出参）：返回用户需要在npu device侧申请的workspace大小。
  * executor（aclOpExecutor\*\*, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ``` 
    第一段接口完成入参校验，出现以下场景时报错：
    返回161001（ACLNN_ERR_PARAM_NULLPTR）：1.输入和输出的Tensor是空指针。
    返回161002（ACLNN_ERR_PARAM_INVALID）：1.输入和输出的数据类型和数据格式不在支持的范围之内。
  ```

## aclnnApplyAdamWQuant

- **参数说明：**

  * workspace（void \*， 入参）： 在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t， 入参）： 在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamWQuantGetWorkspaceSize获取。
  * executor（aclOpExecutor \*， 入参）： op执行器，包含了算子计算流程。
  * stream（aclrtStream， 入参）： 指定执行任务的Stream。

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  varRef的shape满足约束：
  - varRef.shape = grad.shape
  - varRef.shape = mRef.shape
  - varRef.shape = vRef.shape
  - varRef.size/blockSize = absmaxMRef.size
  - varRef.size/blockSize = absmaxVRef.size
  
  - 确定性计算：
    - aclnnApplyAdamWQuant默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_adam_w_quant.h"
#define FAILED 1

#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR] " fmt "\n", ##args)
#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)

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

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
}


int Init(int32_t deviceId, aclrtStream* stream) {
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
aclDataType dataType, aclTensor** tensor) {
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
*tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
shape.data(), shape.size(), *deviceAddr);
return 0;

}

int main() {
// 1. 固定写法，device/stream初始化, 参考acl API手册
// 根据自己的实际device填写deviceId
int32_t deviceId = 0;
aclrtStream stream;
auto ret = Init(deviceId, &stream);
// check根据自己的需要处理
CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
// 2. 构造输入与输出，需要根据API的接口定义构造
std::vector<int64_t> VarRefShape = {1,256};
std::vector<int64_t> GradShape = {1,256};
std::vector<int64_t> mRefShape = {1,256};
std::vector<int64_t> vRefShape = {1,256};
std::vector<int64_t> mapMShape = {256,};
std::vector<int64_t> mapVShape = {256,};
std::vector<int64_t> absmaxMRefShape = {1,};
std::vector<int64_t> absmaxVRefShape = {1,};
std::vector<int64_t> stepShape = {1,};

void *VarRefDeviceAddr = nullptr;
void *GradDeviceAddr = nullptr;
void *mRefDeviceAddr = nullptr;
void *vRefDeviceAddr = nullptr;
void *qmapMDeviceAddr = nullptr;
void *qmapVDeviceAddr = nullptr;
void *absmaxMRefDeviceAddr = nullptr;
void *absmaxVRefDeviceAddr = nullptr;
void *stepDeviceAddr = nullptr;

aclTensor *varRef = nullptr;
aclTensor *grad = nullptr;
aclTensor *mRef = nullptr;
aclTensor *vRef = nullptr;
aclTensor *qmapM = nullptr;
aclTensor *qmapV = nullptr;
aclTensor *absmaxMRef = nullptr;
aclTensor *absmaxVRef = nullptr;
aclTensor *step = nullptr;

std::vector<float> inputVarHostData(256);
std::vector<float> inputGradHostData(256);
std::vector<uint8_t> inputMHostData(256);
std::vector<uint8_t> inputVHostData(256);
std::vector<float> inputmapMHostData(256);
std::vector<float> inputmapVHostData(256);
std::vector<float> inputabsmaxMHostData = {5};
std::vector<float> inputabsmaxVHostData = {3};
std::vector<int64_t> inputstepHostData(1);

const float lr = 0.1;
const float beta1 = 0.1;
const float beta2 = 0.1;
const float weightDecay = 0.1;
const float eps = 0.01;
const float gnormScale = 0.1;
const int64_t blockSize = 256;
char* quantModeOptional = "BLOCKWISE";

// 创建gradOutput aclTensor
ret = CreateAclTensor(inputVarHostData, VarRefShape, &VarRefDeviceAddr, aclDataType::ACL_FLOAT, &varRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputGradHostData, GradShape, &GradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputMHostData, mRefShape, &mRefDeviceAddr, aclDataType::ACL_UINT8, &mRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputVHostData, vRefShape, &vRefDeviceAddr, aclDataType::ACL_UINT8, &vRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapMHostData, mapMShape, &qmapMDeviceAddr, aclDataType::ACL_FLOAT, &qmapM);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputmapVHostData, mapVShape, &qmapVDeviceAddr, aclDataType::ACL_FLOAT, &qmapV);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxMHostData, absmaxMRefShape, &absmaxMRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxMRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputabsmaxVHostData, absmaxVRefShape, &absmaxVRefDeviceAddr, aclDataType::ACL_FLOAT, &absmaxVRef);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

ret = CreateAclTensor(inputstepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
CHECK_RET(ret == ACL_SUCCESS, return FAILED);

// 3. 调用CANN算子库API，需要修改为具体的API
uint64_t workspaceSize = 0;
aclOpExecutor* executor;
// 调用aclnnApplyAdamWQuantGetWorkspaceSize第一段接口
ret = aclnnApplyAdamWQuantGetWorkspaceSize(varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, lr, beta1, beta2, weightDecay, eps, gnormScale, quantModeOptional, blockSize, &workspaceSize, &executor);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
// 根据第一段接口计算出的workspaceSize申请device内存
void* workspaceAddr = nullptr;
if (workspaceSize > 0) {
ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
}
// 调用aclnnApplyAdamWQuant第二段接口
ret = aclnnApplyAdamWQuant(workspaceAddr, workspaceSize, executor, stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnApplyAdamWQuant failed. ERROR: %d\n", ret); return ret);
// 4. 固定写法，同步等待任务执行结束
ret = aclrtSynchronizeStream(stream);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return FAILED);

// 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
auto size = GetShapeSize(VarRefShape);
std::vector<float> resultData(size, 0);
ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), VarRefDeviceAddr,
                size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
for (int64_t i = 0; i < size; i++) {
LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }
// 6. 释放aclTensor，需要根据具体API的接口定义修改
aclDestroyTensor(varRef);
aclDestroyTensor(grad);
aclDestroyTensor(mRef);
aclDestroyTensor(vRef);
aclDestroyTensor(qmapM);
aclDestroyTensor(qmapV);
aclDestroyTensor(absmaxMRef);
aclDestroyTensor(absmaxVRef);
aclDestroyTensor(step);

// 7. 释放device资源，需要根据具体API的接口定义修改
aclrtFree(VarRefDeviceAddr);
aclrtFree(GradDeviceAddr);
aclrtFree(mRefDeviceAddr);
aclrtFree(vRefDeviceAddr);
aclrtFree(qmapMDeviceAddr);
aclrtFree(qmapVDeviceAddr);
aclrtFree(absmaxMRefDeviceAddr);
aclrtFree(absmaxVRefDeviceAddr);
aclrtFree(stepDeviceAddr);
if (workspaceSize > 0) {
aclrtFree(workspaceAddr);
}
aclrtDestroyStream(stream);
aclrtResetDevice(deviceId);
aclFinalize();
return 0;
}
```