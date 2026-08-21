# aclnnCrossEntropySumExpAndIndexLogit

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/loss/cross_entropy_sum_exp_and_index_logit)

## 产品支持情况

|产品      | 是否支持 |
|:----------------------------|:-----------:|
|<term>Ascend 950PR/Ascend 950DT</term>|      √     |
|<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>|    ×     |
|<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>|    ×     |
|<term>Atlas 200I/500 A2 推理产品</term>|      ×     |
|<term>Atlas 推理系列产品</term>|      ×     |
|<term>Atlas 训练系列产品</term>|      ×     |

## 功能说明

- **接口功能**：本算子为vocab并行（Tensor Parallel）场景下CrossEntropy本地计算段的融合算子。对按vocab维切分后的TP rank本地logits，在`all_reduce(MAX)`之后、`predictedLogits` / `sumExpLogits`的`all_reduce(SUM)`之前，一次性完成logits平移（减全局最大值）、target越界mask判定、本地target offset计算、target对应logit的gather，以及`exp`与沿vocab维的本地求和，降低小算子launch、GM读写和中间tensor materialization开销。面向超大词表大模型训练场景。

- **计算公式**：

  设当前rank本地vocab shard为`vocabParallelLogits`，全局最大logit为`globalLogitsMax`，vocab分片范围为`[vocabStartIndex, vocabEndIndex)`。对每个token `i`、每个本地vocab位置`j`：

  **1. target mask：**

  $$
  targetMask[i] = \begin{cases} 1, & target[i] < vocabStartIndex \text{ 或 } target[i] \geq vocabEndIndex \\ 0, & \text{otherwise} \end{cases}
  $$

  **2. 本地target offset：**

  $$
  targetOffset[i] = \begin{cases} 0, & targetMask[i] = 1 \\ target[i] - vocabStartIndex, & targetMask[i] = 0 \end{cases}
  $$

  **3. predicted logit gather（logits平移）：**

  $$
  predictedLogits[i] = \begin{cases} 0, & targetMask[i] = 1 \\ vocabParallelLogits[i, targetOffset[i]] - globalLogitsMax[i], & targetMask[i] = 0 \end{cases}
  $$

  **4. 指数计算：**

  $$
  expLogits[i, j] = \exp\big(vocabParallelLogits[i, j] - globalLogitsMax[i]\big)
  $$

  **5. 本地求和：**

  $$
  sumExpLogits[i] = \sum_{j=0}^{V\_local-1} expLogits[i, j]
  $$

- **特殊处理说明**：中间计算强制使用FLOAT（`BFLOAT16`输入自动升精度），exp操作数已减去全局最大值以抑制上溢，下溢按FLOAT自然返回0。本算子不涉及除法，无需除零保护。`sumExpLogits`沿V_local维的累加在单核内完成（无跨核规约、无AtomicAdd），累加顺序固定，相同输入多次调用结果一致，为确定性实现。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnCrossEntropySumExpAndIndexLogit`接口执行计算。

```cpp
aclnnStatus aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
  const aclTensor *vocabParallelLogits,
  const aclTensor *target,
  const aclTensor *globalLogitsMax,
  int64_t          vocabStartIndex,
  int64_t          vocabEndIndex,
  const aclTensor *predictedLogitsOut,
  const aclTensor *sumExpLogitsOut,
  const aclTensor *expLogitsOut,
  const aclTensor *targetOffsetOut,
  const aclTensor *targetMaskOut,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnCrossEntropySumExpAndIndexLogit(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1200px"><colgroup>
  <col style="width: 160px">
  <col style="width: 100px">
  <col style="width: 260px">
  <col style="width: 260px">
  <col style="width: 140px">
  <col style="width: 80px">
  <col style="width: 280px">
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
    </tr></thead>
  <tbody>
    <tr>
      <td>vocabParallelLogits (aclTensor*)</td>
      <td>输入</td>
      <td>当前TP rank的本地vocab shard logits，公式中的vocabParallelLogits。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT、BFLOAT16</td>
      <td>ND</td>
      <td><ul><li>仅支持2维 [N, V_local] 或3维 [S, B, V_local]。</li><li>shape[:-1] 需与target.shape完全一致。</li><li>最后一维V_local范围：[16, 200K]；BFLOAT16时需为16的倍数，FLOAT时需为8的倍数。</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>target (aclTensor*)</td>
      <td>输入</td>
      <td>全局vocab索引，公式中的target。</td>
      <td>不支持空Tensor。取值为非负整数。</td>
      <td>INT32</td>
      <td>ND</td>
      <td><ul><li>shape与vocabParallelLogits.shape[:-1] 完全一致。</li><li>展平后N = prod(shape) 范围：[1, 32K]。</li></ul></td>
      <td>√</td>
    </tr>
    <tr>
      <td>globalLogitsMax (aclTensor*)</td>
      <td>输入</td>
      <td>all_reduce(MAX) 后得到的全局最大logit，公式中的globalLogitsMax。</td>
      <td>不支持空Tensor。数据类型需与vocabParallelLogits一致。</td>
      <td>同vocabParallelLogits</td>
      <td>ND</td>
      <td>shape与target完全一致。</td>
      <td>√</td>
    </tr>
    <tr>
      <td>vocabStartIndex (int64_t)</td>
      <td>输入</td>
      <td>当前rank vocab分片起始索引（全局），公式中的vocabStartIndex。</td>
      <td>需满足vocabEndIndex > vocabStartIndex，且vocabStartIndex大于等于0。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>vocabEndIndex (int64_t)</td>
      <td>输入</td>
      <td>当前rank vocab分片结束索引（全局），公式中的vocabEndIndex。</td>
      <td>需满足vocabEndIndex - vocabStartIndex == vocabParallelLogits.size(-1)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>predictedLogits (aclTensor*)</td>
      <td>输出</td>
      <td>target对应的logit减去global_max的结果，公式中的predictedLogits。</td>
      <td>shape同target；target不在当前rank分片内时对应位置为0。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>同target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>sumExpLogits (aclTensor*)</td>
      <td>输出</td>
      <td>本地exp(logits - global_max) 沿最后一维的求和，公式中的sumExpLogits。</td>
      <td>shape同target。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>同target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>expLogits (aclTensor*)</td>
      <td>输出</td>
      <td>exp(vocabParallelLogits - globalLogitsMax)，公式中的expLogits。</td>
      <td>shape同vocabParallelLogits。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>同vocabParallelLogits.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>targetOffset (aclTensor*)</td>
      <td>输出</td>
      <td>target - vocabStartIndex，公式中的targetOffset。</td>
      <td>shape同target；target不在当前rank分片内时置0。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>同target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>targetMask (aclTensor*)</td>
      <td>输出</td>
      <td>vocab越界掩码，1表示target不在当前rank分片内、0表示在内，公式中的targetMask。</td>
      <td>shape同target。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>同target.shape</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize (uint64_t*)</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor (aclOpExecutor**)</td>
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

  <table style="undefined;table-layout: fixed; width: 970px"><colgroup>
  <col style="width: 263px">
  <col style="width: 88px">
  <col style="width: 619px">
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
      <td>vocabParallelLogits、target、globalLogitsMax、predictedLogits、sumExpLogits、expLogits、targetOffset、targetMask是空指针。</td>
  </tr>
  <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>vocabParallelLogits维度不是2或3。</td>
  </tr>
  <tr>
      <td>vocabParallelLogits.shape[:-1] 与target.shape不一致。</td>
  </tr>
  <tr>
      <td>globalLogitsMax.shape与target.shape不一致。</td>
  </tr>
  <tr>
      <td>vocabEndIndex <= vocabStartIndex。</td>
  </tr>
  <tr>
      <td>vocabEndIndex - vocabStartIndex != vocabParallelLogits.size(-1)。</td>
  </tr>
  <tr>
      <td>vocabParallelLogits或globalLogitsMax的数据类型不在支持范围内（FLOAT / BFLOAT16），或两者数据类型不一致。</td>
  </tr>
  <tr>
      <td>target的数据类型不是INT32。</td>
  </tr>
  <tr>
      <td>V_local不满足对齐约束（BFLOAT16时非16的倍数，FLOAT时非8的倍数）。</td>
  </tr>
  <tr>
      <td>N不在 [1, 32K] 范围或V_local不在 [16, 200K] 范围。</td>
  </tr>
  </tbody>
  </table>

## aclnnCrossEntropySumExpAndIndexLogit

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- **确定性计算**：
  aclnnCrossEntropySumExpAndIndexLogit默认确定性实现。

- **Batch一致性说明**：
  aclnnCrossEntropySumExpAndIndexLogit默认Batch一致性实现。

- **输入shape限制**：
    - vocabParallelLogits仅支持2维 [N, V_local] 或3维 [S, B, V_local]。
    - vocabParallelLogits.shape[:-1] 与target.shape完全一致。
    - globalLogitsMax.shape与target.shape完全一致。
    - V_local对齐约束：BFLOAT16输入时V_local需为16的倍数，FLOAT输入时V_local需为8的倍数（保证UB 32字节对齐）。
    - N（prod(target.shape)）范围：[1, 32K]。
    - V_local范围：[16, 200K]。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_cross_entropy_sum_exp_and_index_logit.h"

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
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }
}

void PrintOutIntResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<int32_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape,
    void** deviceAddr, aclDataType dataType, aclTensor** tensor)
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType,
        strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main()
{
    // 1. 固定写法，device/stream初始化, 参考acl API
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口定义构造
    // 示例使用 2 维 shape：N=4, V_local=16（FLOAT 时 V_local 需为 8 的倍数）
    int64_t N = 4;
    int64_t vLocal = 16;
    int64_t vocabStart = 0;
    int64_t vocabEnd = vLocal;   // target 均落在 [0, 16) 内，targetMask 全 0

    std::vector<int64_t> logitsShape = {N, vLocal};
    std::vector<int64_t> targetShape = {N};

    void* logitsDeviceAddr = nullptr;
    void* targetDeviceAddr = nullptr;
    void* maxDeviceAddr = nullptr;
    void* predictedDeviceAddr = nullptr;
    void* sumExpDeviceAddr = nullptr;
    void* expLogitsDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* maskDeviceAddr = nullptr;

    aclTensor* logitsTensor = nullptr;
    aclTensor* targetTensor = nullptr;
    aclTensor* maxTensor = nullptr;
    aclTensor* predictedTensor = nullptr;
    aclTensor* sumExpTensor = nullptr;
    aclTensor* expLogitsTensor = nullptr;
    aclTensor* offsetTensor = nullptr;
    aclTensor* maskTensor = nullptr;

    // 构造示例输入数据
    std::vector<float> logitsHostData(N * vLocal, 0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < vLocal; j++) {
            logitsHostData[i * vLocal + j] = static_cast<float>(i) * 1.0f + static_cast<float>(j) * 0.1f;
        }
    }
    std::vector<int32_t> targetHostData = {2, 5, 10, 15};
    std::vector<float> maxHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> predictedHostData(N, 0);
    std::vector<float> sumExpHostData(N, 0);
    std::vector<float> expLogitsHostData(N * vLocal, 0);
    std::vector<int32_t> offsetHostData(N, 0);
    std::vector<int32_t> maskHostData(N, 0);

    ret = CreateAclTensor(logitsHostData, logitsShape, &logitsDeviceAddr,
        aclDataType::ACL_FLOAT, &logitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr,
        aclDataType::ACL_INT32, &targetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxHostData, targetShape, &maxDeviceAddr,
        aclDataType::ACL_FLOAT, &maxTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(predictedHostData, targetShape, &predictedDeviceAddr,
        aclDataType::ACL_FLOAT, &predictedTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumExpHostData, targetShape, &sumExpDeviceAddr,
        aclDataType::ACL_FLOAT, &sumExpTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(expLogitsHostData, logitsShape, &expLogitsDeviceAddr,
        aclDataType::ACL_FLOAT, &expLogitsTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(offsetHostData, targetShape, &offsetDeviceAddr,
        aclDataType::ACL_INT32, &offsetTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maskHostData, targetShape, &maskDeviceAddr,
        aclDataType::ACL_INT32, &maskTensor);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用第一段接口
    ret = aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize(
        logitsTensor,
        targetTensor,
        maxTensor,
        vocabStart,
        vocabEnd,
        predictedTensor,
        sumExpTensor,
        expLogitsTensor,
        offsetTensor,
        maskTensor,
        &workspaceSize,
        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogitGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用第二段接口
    ret = aclnnCrossEntropySumExpAndIndexLogit(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnCrossEntropySumExpAndIndexLogit failed. ERROR: %d\n", ret); return ret);

    // 4. 固定写法，同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值
    LOG_PRINT("predictedLogits:\n");
    PrintOutResult(targetShape, &predictedDeviceAddr);
    LOG_PRINT("sumExpLogits:\n");
    PrintOutResult(targetShape, &sumExpDeviceAddr);
    LOG_PRINT("expLogits:\n");
    PrintOutResult(logitsShape, &expLogitsDeviceAddr);
    LOG_PRINT("targetOffset:\n");
    PrintOutIntResult(targetShape, &offsetDeviceAddr);
    LOG_PRINT("targetMask:\n");
    PrintOutIntResult(targetShape, &maskDeviceAddr);

    // 6. 释放aclTensor
    aclDestroyTensor(logitsTensor);
    aclDestroyTensor(targetTensor);
    aclDestroyTensor(maxTensor);
    aclDestroyTensor(predictedTensor);
    aclDestroyTensor(sumExpTensor);
    aclDestroyTensor(expLogitsTensor);
    aclDestroyTensor(offsetTensor);
    aclDestroyTensor(maskTensor);

    // 7. 释放device资源
    aclrtFree(logitsDeviceAddr);
    aclrtFree(targetDeviceAddr);
    aclrtFree(maxDeviceAddr);
    aclrtFree(predictedDeviceAddr);
    aclrtFree(sumExpDeviceAddr);
    aclrtFree(expLogitsDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(maskDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
