# aclnnApplyAdamWQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_adam_w_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     x    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **接口功能：** 对优化器输入的m和v作为索引，取出各自qmap中的值，乘以每个blockSize对应的absmax进行反量化，而后实现adamW优化器功能，更新后的m和v每blockSize中取一个最大值，每blockSize个m和v对应一个absmax，进行一次norm归一化，利用二分法找到对应m和v对应qmap中的索引作为输出，absmax也作为下一轮量化的输入。

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

```Cpp
aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(
    const aclTensor *varRef,
    const aclTensor *grad,
    const aclTensor *mRef,
    const aclTensor *vRef,
    const aclTensor *qmapM,
    const aclTensor *qmapV,
    const aclTensor *absmaxMRef,
    const aclTensor *absmaxVRef,
    const aclTensor *step,
    double          lr,
    double          beta1,
    double          beta2,
    double          weightDecay,
    double          eps,
    double          gnormScale,
    char *          quantModeOptional,
    int64_t         blockSize,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnApplyAdamWQuant(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnApplyAdamWQuantGetWorkspaceSize

- **参数说明：**

  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1235px"><colgroup>
  <col style="width: 271px">
  <col style="width: 88px">
  <col style="width: 289px">
  <col style="width: 156px">
  <col style="width: 120px">
  <col style="width: 95px">
  <col style="width: 108px">
  <col style="width: 108px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">参数名</th>
      <th class="tg-0pky">输入/输出</th>
      <th class="tg-0pky">描述</th>
      <th class="tg-0pky">使用说明</th>
      <th class="tg-0pky">数据类型</th>
      <th class="tg-0pky">数据格式</th>
      <th class="tg-0pky">维度(shape)</th>
      <th class="tg-0pky">非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">varRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">待计算的权重输入同时也是输出，公式中的θ。</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">FLOAT16、BFLOAT16、FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">grad（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">输入梯度，公式中的g。</td>
      <td class="tg-0pky">shape要求与输入varRef保持一致。</td>
      <td class="tg-0pky">与varRef保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">mRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">adamw优化器公式中m参数量化前的索引值，根据索引导出qmapM中具体的值。</td>
      <td class="tg-0pky">shape要求与输入varRef保持一致。</td>
      <td class="tg-0pky">uin8_t</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">adamw优化器公式中v参数量化前的索引值，根据索引导出qmapV中具体的值。</td>
      <td class="tg-0pky">shape要求与输入varRef保持一致。</td>
      <td class="tg-0pky">uin8_t</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">qmapM（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">量化映射表升序排列，每个m根据mRef中的索引值进行选择。</td>
      <td class="tg-0pky">shape要求时[256,]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">qmapV（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">量化映射表升序排列，每个v根据vRef中的索引值进行选择。</td>
      <td class="tg-0pky">shape要求时[256,]。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">absmaxMRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">每blockSize(256)个vRef对应一个最大值，对于用于对mRef索引选择qmapM中的值乘以对应的absmaxMRef进行反量化。再通过更新后的mRef每blockSize(256)个选择出一个最大值，作为absmaxMRef的输出。</td>
      <td class="tg-0pky">shape要求为“absmaxMRef.size = mRef.size/blockSize”。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">absmaxVRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">每blockSize(256)个vRef对应一个最大值，对于用于对vRef索引选择qmapV中的值乘以对应的absmaxVRef进行反量化。再通过更新后的vRef每blockSize(256)个选择出一个最大值，作为absmaxVRef的输出。</td>
      <td class="tg-0pky">shape要求为“absmaxVRef.size = vRef.size/blockSize”。</td>
      <td class="tg-0pky">FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">step（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">公式中的t，迭代次数。</td>
      <td class="tg-0pky">shape要求为[1]。</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">lr（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">学习率，公式中的η，推荐1e-3，1e-5，1e-8，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">adamw优化器公式中beta1参数，推荐0.9，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">adamw优化器公式中beta2参数，推荐0.999，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">weightDeacy（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">权重衰减系数，adamw优化器公式中λ参数，推荐0.999，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">eps（double）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">adamw优化器公式中ϵ参数，加在分母中用来防止除0，推荐1e-8。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">gnormScale（double）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">对输入参数grad进行缩放的参数，推荐0.999，范围0~1。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">blockSize（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">每个block参与计算的size大小，固定为256。为上述absmax中所说，每blockSize选择一个最大值。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">quantModeOptional（char*）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">保留参数暂无意义</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">workspaceSize（uint64_t*）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回需要在Device侧申请的workspace大小。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">executor（aclOpExecutor**）</td>
      <td class="tg-0pky">输出</td>
      <td class="tg-0pky">返回op执行器，包含了算子计算流程。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

    第一段接口完成入参校验，出现以下场景时报错：
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 728px"><colgroup>
  <col style="width: 267px">
  <col style="width: 87px">
  <col style="width: 374px">
  </colgroup>
  <thead>
    <tr>
      <th class="tg-0pky">返回码</th>
      <th class="tg-0pky">错误码</th>
      <th class="tg-0pky">描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_NULLPTR</td>
      <td class="tg-0pky">161001</td>
      <td class="tg-0pky">输入和输出的Tensor是空指针。</td>
    </tr>
    <tr>
      <td class="tg-0pky">ACLNN_ERR_PARAM_INVALID</td>
      <td class="tg-0pky">161002</td>
      <td class="tg-0pky">输入和输出的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnApplyAdamWQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAdvanceStepGetWorkspaceSize获取。</td>
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

  aclnnStatus： 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  varRef的shape满足约束：
  - varRef.shape = grad.shape
  - varRef.shape = mRef.shape
  - varRef.shape = vRef.shape
  - varRef.size/blockSize = absmaxMRef.size
  - varRef.size/blockSize = absmaxVRef.size

  确定性计算：
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