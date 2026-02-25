# aclnnApplyFusedEmaAdam

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/optim/apply_fused_ema_adam)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>     |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- **接口功能**：实现FusedEmaAdam融合优化器功能。
- **计算公式**：

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$

  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$

  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用 “aclnnApplyFusedEmaAdamGetWorkspaceSize” 接口获取入参并根据计算流程计算所需workspace大小以及包含了算子计算流程的执行器，再调用 “aclnnApplyFusedEmaAdam” 接口执行计算。

```cpp
aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(
    const aclTensor       *grad,
    aclTensor       *varRef,
    aclTensor       *mRef,
    aclTensor       *vRef,
    const aclTensor *sRef,
    const aclTensor *step,
    double           lr,
    double           emaDecay,
    double           beta1,
    double           beta2,
    double           eps,
    int64_t          mode,
    bool             biasCorrection,
    double           weightDecay,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnApplyFusedEmaAdam(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnApplyFusedEmaAdamGetWorkspaceSize

- **参数说明：**
  </style>
  <table class="tg" style="undefined;table-layout: fixed; width: 1137px"><colgroup>
  <col style="width: 201px">
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
      <td class="tg-0pky">grad（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">待更新参数对应的梯度，对应公式中的grad。</td>
      <td class="tg-0pky"></td>
      <td class="tg-0pky">FLOAT16、BFLOAT16、FLOAT32</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">varRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">待更新参数，对应公式中的var。</td>
      <td class="tg-0pky">shape要求与输入grad保持一致。</td>
      <td class="tg-0pky">与grad保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">mRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">待更新参数对应的一阶动量，对应公式中的m。</td>
      <td class="tg-0pky">shape要求与输入grad保持一致。</td>
      <td class="tg-0pky">与grad保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">vRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">待更新参数对应的二阶动量，对应公式中的v。</td>
      <td class="tg-0pky">shape要求与输入grad保持一致。</td>
      <td class="tg-0pky">与grad保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">sRef（aclTensor*）</td>
      <td class="tg-0pky">输入/输出</td>
      <td class="tg-0pky">待更新参数对应的EMA权重，对应公式中的s。</td>
      <td class="tg-0pky">shape要求与输入grad保持一致。</td>
      <td class="tg-0pky">与grad保持一致</td>
      <td class="tg-0pky">ND</td>
      <td class="tg-0pky">1-8</td>
      <td class="tg-0pky">x</td>
    </tr>
    <tr>
      <td class="tg-0pky">step（aclTensor*）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">优化器当前的更新次数，对应公式中的step。</td>
      <td class="tg-0pky">shape要求是[1,]。</td>
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
      <td class="tg-0pky">emaDecay（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">指数移动平均（EMA）的衰减速率，对应公式中的emaDecay。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta1（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">计算一阶动量的系数，公式中beta1参数，推荐0.9，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">beta2（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">计算二阶动量的系数，公式中beta1参数，推荐0.9，范围0~1。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">eps（double）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">加到分母上的项，用于数值稳定性，对应公式中的eps。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">DOUBLE</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">mode（int64_t）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">控制应用L2正则化还是权重衰减，对应公式中的mode，1为adamw，0为L2。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">INT64</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0pky">biasCorrection（bool）</td>
      <td class="tg-0pky">输入</td>
      <td class="tg-0pky">控制是否进行偏差校正，对应公式中的biasCorrection，true表示进行校正，false表示不做校正。</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">BOOL</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
      <td class="tg-0pky">-</td>
    </tr>
    <tr>
      <td class="tg-0lax">weightDecay（double）</td>
      <td class="tg-0lax">输入</td>
      <td class="tg-0lax">权重衰减，对应公式中的weightDecay。</td>
      <td class="tg-0lax">-</td>
      <td class="tg-0lax">DOUBLE</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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

## aclnnApplyFusedEmaAdam

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnApplyAdamWV2GetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

输入grad、var、m、v、s的数据类型和shape需要保持一致。

- 确定性计算：
  - aclnnApplyFusedEmaAdam默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_apply_fused_ema_adam.h"
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
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
}

int Init(int32_t deviceId, aclrtStream *stream) {
  // 固定写法，资源初始化
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
  // 调用aclrtMalloc申请device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret);
            return ret);
  // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size,
                    ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret);
            return ret);

  // 计算连续tensor的strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用aclCreateTensor接口创建aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
                            strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret);
            return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  // input
  std::vector<float> gradHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> varHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> mHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> vHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> sHostData = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> stepHostData = {10, 10, 10, 10};
  std::vector<int64_t> inputShape = {2, 2, 2};
  std::vector<int64_t> stepShape = {2, 2};
  void *gradDeviceAddr = nullptr;
  void *varDeviceAddr = nullptr;
  void *mDeviceAddr = nullptr;
  void *vDeviceAddr = nullptr;
  void *sDeviceAddr = nullptr;
  void *stepDeviceAddr = nullptr;
  aclTensor *grad = nullptr;
  aclTensor *var = nullptr;
  aclTensor *m = nullptr;
  aclTensor *v = nullptr;
  aclTensor *s = nullptr;
  aclTensor *step = nullptr;
  ret = CreateAclTensor(gradHostData, inputShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(varHostData, inputShape, &varDeviceAddr, aclDataType::ACL_FLOAT, &var);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(mHostData, inputShape, &mDeviceAddr, aclDataType::ACL_FLOAT, &m);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(vHostData, inputShape, &vDeviceAddr, aclDataType::ACL_FLOAT, &v);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(sHostData, inputShape, &sDeviceAddr, aclDataType::ACL_FLOAT, &s);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(stepHostData, stepShape, &stepDeviceAddr, aclDataType::ACL_INT64, &step);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // out, inplace
  std::vector<int64_t> outShape = {2, 2, 2};

  // attr
  float lr = 0.001f;
  float emaDecay = 0.5f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  int64_t mode = 1;
  bool bias = true;
  float weightDecay = 0.5f;

  uint64_t workspaceSize = 0;
  aclOpExecutor *executor;

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  // 调用aclnnApplyFusedEmaAdam第一段接口
  ret = aclnnApplyFusedEmaAdamGetWorkspaceSize(grad, var, m, v, s, step, lr, emaDecay, beta1, beta2, eps,
                                               mode, bias, weightDecay, &workspaceSize, &executor);
  CHECK_RET(
      ret == ACL_SUCCESS,
      LOG_PRINT("aclnnApplyFusedEmaAdamGetWorkspaceSize failed. ERROR: %d\n",
                ret);
      return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void *workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              return ret);
  }

  // 调用aclnnApplyFusedEmaAdam第二段接口
  ret = aclnnApplyFusedEmaAdam(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclnnApplyFusedEmaAdam failed. ERROR: %d\n", ret);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
            LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            return ret);

  // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(outShape, &varDeviceAddr);
  PrintOutResult(outShape, &mDeviceAddr);
  PrintOutResult(outShape, &vDeviceAddr);
  PrintOutResult(outShape, &sDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(grad);
  aclDestroyTensor(var);
  aclDestroyTensor(m);
  aclDestroyTensor(v);
  aclDestroyTensor(s);
  aclDestroyTensor(step);

  // 7. 释放device资源
  aclrtFree(gradDeviceAddr);
  aclrtFree(varDeviceAddr);
  aclrtFree(mDeviceAddr);
  aclrtFree(vDeviceAddr);
  aclrtFree(sDeviceAddr);
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
