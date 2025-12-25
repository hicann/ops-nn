# aclnnCrossEntropyLoss

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 接口功能：计算输入的交叉熵损失。
- 计算公式：
  
  reductionOptional = mean时，交叉熵损失loss的计算公式为：

  $$
  l_n = -weight_{y_n}*log\frac{exp(x_{n,y_n})}{\sum_{c=1}^Cexp(x_{n,c})}*1\{y_n\ !=\ ignoreIndex \}
  $$

  $$
  loss=\begin{cases}\sum_{n=1}^N\frac{1}{\sum_{n=1}^Nweight_{y_n}*1\{y_n\ !=\ ignoreIndex \}}l_n,&\text{if reductionOptional = ‘mean’} \\\sum_{n=1}^Nl_n,&\text {if reductionOptional = ‘sum’ }\\\{l_0,l_1,...,l_n\},&\text{if reductionOptional = ‘None’ }\end{cases}
  $$

  log\_prob计算公式为：

  $$
  lse_n = log*\sum_{c=1}^{C}exp(x_{n,c})
  $$

  $$
  logProb_{n,c} = x_{n,c} - lse_n
  $$

  zloss计算公式为：

  $$
  zloss_n = lseSquareScaleForZloss * （lse_n）^2 
  $$

  其中，N为batch数，C为标签数。
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnCrossEntropyLossGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnCrossEntropyLoss”接口执行计算。

```Cpp
aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(
 const aclTensor *input,
 const aclTensor *target,
 const aclTensor *weightOptional,
 char            *reductionOptional,
 int64_t          ignoreIndex,
 double           labelSmoothing,
 double           lseSquareScaleForZloss,
 bool             returnZloss,
 const aclTensor *lossOut,
 const aclTensor *logProbOut,
 const aclTensor *zlossOut,
 const aclTensor *lseForZlossOut,
 uint64_t        *workspaceSize,
 aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnCrossEntropyLoss(
 void          *workspace,
 uint64_t       workspaceSize,
 aclOpExecutor *executor,
 aclrtStream    stream)
```

## aclnnCrossEntropyLossGetWorkspaceSize

- **参数说明**
    <table style="undefined;table-layout: fixed; width: 1370px"><colgroup>
    <col style="width: 208px">
    <col style="width: 120px">
    <col style="width: 256px">
    <col style="width: 226px">
    <col style="width: 149px">
    <col style="width: 111px">
    <col style="width: 155px">
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
    <td>input</td>
    <td>输入</td>
    <td>公式中的input。</td>
    <td>-</td>
    <td>FLOAT、FLOAT16、BFLOAT16</td>
    <td>ND</td>
    <td>(N,C)<br>N为批处理大小，C为标签数，必须大于0</td>
    <td>-</td>
    </tr>
    <tr>
    <td>target</td>
    <td>输入</td>
    <td>表示标签，公式中的y。</td>
    <td>-</td>
    <td>INT32、INT64</td>
    <td>ND</td>
    <td>(N)<br>N与input第零维相等，数值在[0, C)之间.</td>
    <td>-</td>
    </tr>
    <tr>
    <td>weightOptional</td>
    <td>输入</td>
    <td>表示为每个类别指定的缩放权重，公式中的weight。</td>
    <td>如果不给定，则不对target加权.</td>
    <td>FLOAT</td>
    <td>ND</td>
    <td>(C)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>reductionOptional</td>
    <td>输入</td>
    <td>表示loss的归约方式。</td>
    <td>支持["mean", "sum", "none"]。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>ignoreIndex</td>
    <td>输入</td>
    <td>指定忽略的标签。</td>
    <td>数值必须小于C，当小于零时视为无忽略标签。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>labelSmoothing</td>
    <td>输入</td>
    <td>表示计算loss时的平滑量。</td>
    <td>数值在[0.0, 1.0]之间。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseSquareScaleForZloss</td>
    <td>输入</td>
    <td>表示zloss计算所需的scale。公式中的lse_square_scale_for_zloss。</td>
    <td>数值在[0, 1)之间。当前暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>returnZloss</td>
    <td>输入</td>
    <td>控制是否返回zloss输出。</td>
    <td>需要输出zLoss时传入True，否则传入False。当前暂不支持。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lossOut</td>
    <td>输出</td>
    <td>表示输出损失。</td>
    <td>reductionOptional为“None”时，shape为[N]，与input第零维一致；否则shape为[1]。</td>
    <td>与input相同</td>
    <td>ND</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>logProbOut</td>
    <td>输出</td>
    <td>输出给反向计算的输出。</td>
    <td>-</td>
    <td>与input相同</td>
    <td>ND</td>
    <td>(N,C)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>zlossOut</td>
    <td>输出</td>
    <td>表示辅助损失。</td>
    <td>当前暂不支持。</td>
    <td>与input相同</td>
    <td>ND</td>
    <td>与lossOut一致</td>
    <td>-</td>
    </tr>
    <tr>
    <td>lseForZlossOut</td>
    <td>输出</td>
    <td>表示zloss场景输出给反向的Tensor，lseSquareScaleForZloss为0时输出为None。</td>
    <td>当前暂不支持。</td>
    <td>与input相同</td>
    <td>ND</td>
    <td>(N)</td>
    <td>-</td>
    </tr>
    <tr>
    <td>workspaceSize</td>
    <td>输出</td>
    <td>返回需要在Device侧申请的workspace大小。</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    </tr>
    <tr>
    <td>executor</td>
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

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
    <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 319px">
    <col style="width: 125px">
    <col style="width: 800px">
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
        <td>传入的input、target、lossOut、logProbOut、zlossOut、lseForZlossOut是空指针。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td>传入的input、target、lossOut、logProbOut、zlossOut、lseForZlossOut的数据类型不在支持范围内。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_INNER_TILING_ERROR</td>
        <td rowspan="2">561002</td>
        <td>传入的input、target、weightOptional的shape不在支持范围内。</td>
      </tr>
      <tr>
        <td>传入的reductionOptional、labelSmoothing的值不在支持范围内。</td>
      </tr>
    </tbody>
    </table>

## aclnnCrossEntropyLoss

- **参数说明**
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCrossEntropyLossGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

  - target仅支持类标签索引，不支持概率输入。
  - 当前暂不支持zloss相关功能。传入相关输入，即lseSquareScaleForZloss、returnZloss，不会生效。
  - 确定性计算： 
    - aclnnCrossEntropyLoss默认确定性实现。
## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```c++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_loss.h"

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
  // 固定写法，初始化
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
    // 1. （固定写法）device/stream初始化, 参考acl API对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {2, 5};
    std::vector<int64_t> targetShape = {2,};
    std::vector<int64_t> weightShape = {5,};
    std::vector<int64_t> lossOutShape = {1,};
    std::vector<int64_t> logProbOutShape = {2,5};
    std::vector<int64_t> zlossOutShape = {1,};
    std::vector<int64_t> lseForZlossOutShape = {2,};

    void* inputDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;

    void* lossOutDeviceAddr = nullptr;
    void* logProbOutDeviceAddr = nullptr;
    void* zlossDeviceAddr = nullptr;
    void* lseForZlossDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* target = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* lossOut = nullptr;
    aclTensor* logProbOut = nullptr;
    aclTensor* zloss = nullptr;
    aclTensor* lseForZloss = nullptr;
    
    // data
    std::vector<float> inputHostData = {5, 0, 3, 3, 7,
                                            9, 3, 5, 2, 4};
    std::vector<int64_t> targetHostData = {0, 0};
    std::vector<float> lossOutHostData = {1.0937543};
    std::vector<float> logProbOutHostData = {
        -2.159461, -7.159461, -4.159461, -4.159461, -0.159461,
        -0.0280476, -6.0280476, -4.0280476, -7.0280476, -5.0280476};
    std::vector<float> zlossOutHostData = {0};
    std::vector<float> lseForZlossOutHostData = {0, 0};

    // attr
    char* reduction = "mean";
    int64_t ignoreIndex = -100;
    float labelSmoothing = 0.0;
    float lseSquareScaleForZloss = 0.0;
    bool returnZloss = 0;

    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建target aclTensor
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建lossOut aclTensor
    ret = CreateAclTensor(lossOutHostData, lossOutShape, &lossOutDeviceAddr, aclDataType::ACL_FLOAT, &lossOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建logProbOut aclTensor
    ret = CreateAclTensor(logProbOutHostData, logProbOutShape, &logProbOutDeviceAddr, aclDataType::ACL_FLOAT, &logProbOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建zloss aclTensor
    ret = CreateAclTensor(zlossOutHostData, zlossOutShape, &zlossDeviceAddr, aclDataType::ACL_FLOAT, &zloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // lseForZloss aclTensor
    ret = CreateAclTensor(lseForZlossOutHostData, lseForZlossOutShape, &lseForZlossDeviceAddr, aclDataType::ACL_FLOAT, &lseForZloss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnCrossEntropyLoss第一段接口
    ret = aclnnCrossEntropyLossGetWorkspaceSize(input, target, weight, reduction, ignoreIndex, labelSmoothing, lseSquareScaleForZloss, returnZloss, lossOut, logProbOut, zloss, lseForZloss, &workspaceSize, &executor);

    CHECK_RET(
        ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropyLossGetWorkspaceSize failed. ERROR: %d\n",
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

    // 调用aclnnCrossEntropyLoss第二段接口
    ret = aclnnCrossEntropyLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnCrossEntropyLoss failed. ERROR: %d\n", ret);
                return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
                return ret);

    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改]

    auto size1 = GetShapeSize(lossOutShape);
    auto size2 = GetShapeSize(logProbOutShape);
    std::vector<float> resultData1(size1, 0);
    std::vector<float> resultData2(size2, 0);
    ret = aclrtMemcpy(resultData1.data(), resultData1.size() * sizeof(resultData1[0]), lossOutDeviceAddr,
                        size1 * sizeof(resultData1[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: \n[");
    for (int64_t i = 0; i < size1; i++) {
        LOG_PRINT("%f, ", i, resultData1[i]);
    }
    LOG_PRINT("]\n");

    ret = aclrtMemcpy(resultData2.data(), resultData2.size() * sizeof(resultData2[0]), logProbOutDeviceAddr,
                        size2 * sizeof(resultData2[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy logProb result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("logprob is: \n [");
    for (int64_t i = 0; i < size2; i++) {
        LOG_PRINT("%f,", i, resultData2[i]);
    }
    LOG_PRINT("]\n");

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(target);
    aclDestroyTensor(lossOut);
    aclDestroyTensor(logProbOut);

    // 7. 释放device资源
    aclrtFree(inputDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(lossOutDeviceAddr);
    aclrtFree(logProbOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
