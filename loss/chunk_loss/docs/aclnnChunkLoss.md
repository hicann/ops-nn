# aclnnChunkLoss

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：融合计算语言模型头线性变换与稀疏交叉熵，并返回对隐状态的梯度。输入为一chunk的隐状态`hidden`、词表权重`weight`与标签`labels`，在Device侧按词表维流式计算，无需物化完整logits。输出包括标量`loss`、online状态`maxOut`/`sumOut`，以及对`hidden`的梯度`dH`。本接口不计算`weight`的梯度。调用方按`(M, H)`切块后逐段调用；各段使用同一`alpha`再将`loss`相加，结果与整段一次计算一致。本接口不接收chunkSize。
- 计算公式：

  `hidden`即为$X$，形状为$[M, H]$。`labels`即为$y$，形状为$[M]$。`weight`即为$W$，形状为$[V, H]$。

  $$
  Z = X W^{\mathsf T}
  $$

  第$i$个token的负对数似然为：

  $$
  \mathrm{nll}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignoreIndex} \\
  \mathrm{logsumexp}(Z_i) - Z_{i, y_i} & 0 \le y_i < V
  \end{cases}
  $$

  $$
  loss = \alpha \sum_{i=0}^{M-1} \mathrm{nll}_i
  $$

  $$
  \mathrm{dZ}_i =
  \begin{cases}
  0 & y_i = \mathrm{ignoreIndex} \\
  \alpha\left(\mathrm{softmax}(Z_i) - e_{y_i}\right) & 0 \le y_i < V
  \end{cases}
  $$

  $$
  \mathrm{dH} = \mathrm{dZ}\, W
  $$

  其中：

  - $M$为token数，等于`hidden`的第0维，也等于`labels`的长度。
  - $H$为`hidden`与`weight`的最后一维。
  - $V$为`weight`的第0维，即词表大小。
  - $e_{y_i}$为长度为$V$的one-hot向量：仅第$y_i$个位置为1，其余为0。
  - $\alpha$为入参`alpha`，在Device侧按FLOAT32与loss/`dZ`相乘做归一化（$\alpha>0$）。
  - $Z$、$\mathrm{logsumexp}$、$\mathrm{softmax}$与`loss`在Device侧按FLOAT32累加计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnChunkLossGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnChunkLoss`接口执行计算。

```cpp
aclnnStatus aclnnChunkLossGetWorkspaceSize(
    const aclTensor *hidden,
    const aclTensor *weight,
    const aclTensor *labels,
    double           alpha,
    int64_t          ignoreIndex,
    aclTensor       *loss,
    aclTensor       *maxOut,
    aclTensor       *sumOut,
    aclTensor       *dH,
    uint64_t        *workspaceSize,
    aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnChunkLoss(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnChunkLossGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px">
  <colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 340px">
  <col style="width: 180px">
  <col style="width: 100px">
  <col style="width: 140px">
  <col style="width: 140px">
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
      <td>hidden（const aclTensor*）</td>
      <td>输入</td>
      <td>一个chunk的隐状态，对应公式中的X。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>数据类型与weight保持一致。</li>
      <li>shape为[M, H]，M、H均大于等于1。</li>
      </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[M, H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>weight（const aclTensor*）</td>
      <td>输入</td>
      <td>语言模型头权重，对应公式中的W。按行存放，第v行对应词表第v项。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>数据类型与hidden保持一致。</li>
      <li>最后一维须与hidden最后一维相等。</li>
      <li>不包含bias。</li>
      </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[V, H]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>labels（const aclTensor*）</td>
      <td>输入</td>
      <td>已与hidden对齐的目标token，对应公式中的y。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>长度等于hidden第0维M。</li>
      <li>等于ignoreIndex的元素不计入loss，对应dH行为0。</li>
      <li>其余元素取值范围须为[0, V)。</li>
      </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>[M]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>loss的归一化乘子，对应公式中的alpha。</td>
      <td>必须是有限数，且大于0。作为归一化乘子：mean归约时通常取有效token数的倒数；sum归约时取1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ignoreIndex（int64_t）</td>
      <td>输入</td>
      <td>labels中需要忽略的取值，对应公式中的ignoreIndex。</td>
      <td>与该值相等的label不产生loss。调用方需显式传入（常见为-100）。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>loss（aclTensor*）</td>
      <td>输出</td>
      <td>本chunk的标量loss，对应公式中的loss。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>数据类型固定为FLOAT32。</li>
      <li>shape为0维标量。</li>
      </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>标量</td>
      <td>-</td>
    </tr>
    <tr>
      <td>maxOut（aclTensor*）</td>
      <td>输出</td>
      <td>每行logits的最大值，对应公式中的max。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>长度等于hidden第0维M。</li>
      </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[M]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sumOut（aclTensor*）</td>
      <td>输出</td>
      <td>每行对exp(Z - max)求和的结果，对应公式中的sum。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>长度等于hidden第0维M。</li>
      </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[M]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>dH（aclTensor*）</td>
      <td>输出</td>
      <td>loss对hidden的梯度，对应公式中的dH。</td>
      <td>
      <ul>
      <li>不支持空Tensor。</li>
      <li>dtype与hidden一致。</li>
      <li>shape为[M, H]；ignore行对应梯度为0。</li>
      </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[M, H]</td>
      <td>×</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td>传入的hidden、weight、labels、loss、maxOut、sumOut、dH、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>hidden、weight、labels、loss、maxOut、sumOut或dH的数据类型不在支持的范围之内，或彼此不匹配。</td>
    </tr>
    <tr>
      <td>hidden、weight或labels的数据格式不是ND，或Tensor非连续。</td>
    </tr>
    <tr>
      <td>hidden、weight、labels、loss、maxOut、sumOut或dH的维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>hidden、weight、labels、maxOut、sumOut或dH的shape不满足约束。</td>
    </tr>
    <tr>
      <td>alpha不是有限数，或alpha小于等于0。</td>
    </tr>
  </tbody>
  </table>

## aclnnChunkLoss

- **参数说明**

  <table style="table-layout: fixed; width: 1000px">
  <colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnChunkLossGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- `hidden`、`weight`、`labels`的shape满足：`hidden`为$[M, H]$，`weight`为$[V, H]$，`labels`为$[M]$；且$M$、$H$、$V$均大于等于1。
- 归约方式固定为先对全部token的nll求和，再乘以`alpha`。多chunk调用时每段传入同一`alpha`，再将各段`loss`相加。
- 不支持语言模型头bias，不支持label smoothing，不输出`weight`的梯度。
- `labels`中非`ignoreIndex`的取值须落在$[0, V)$；越界行为未定义，由调用方保证。aclnn层不对label取值做逐元素拦截。
- 支持输入/输出非连续Tensor：输入在内部转为连续后再计算，输出写回用户提供的Tensor视图。
- 确定性计算：

  <!-- npu="910b" id7 -->
  - <term>Atlas A2系列产品</term>：Kernel 默认按确定性归约实现（禁止多写者 AtomicAdd；loss/dH 为私有 partial + 固定序归约）。是否另支持通过 aclrtSetSysParamOpt 切换未单独分支。
  <!-- end id7 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR&950DT系列产品</term>：支持。开启确定性（如通过`aclrtSetSysParamOpt`设置`ACL_OPT_DETERMINISTIC`）后，Host tiling读取确定性开关，Kernel对`loss`/`dH`采用可复现的归约路径（私有 partial、有序归约或等价确定性累加），相同输入多次执行结果一致。
  <!-- end id9 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_loss.h"

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

uint16_t FloatToFp16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffU) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00U);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
    aclDataType dataType, aclTensor** tensor)
{
    auto size = static_cast<size_t>(GetShapeSize(shape)) * sizeof(T);
    if (shape.empty()) {
        size = sizeof(T);
    }
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor（0维标量时shape为空）
    *tensor = aclCreateTensor(shape.empty() ? nullptr : shape.data(),
        shape.size(),
        dataType,
        shape.empty() ? nullptr : strides.data(),
        0,
        aclFormat::ACL_FORMAT_ND,
        shape.empty() ? nullptr : shape.data(),
        shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1.（固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出：M=4，H=8，V=16。第3个label等于ignoreIndex
    std::vector<int64_t> hiddenShape = {4, 8};
    std::vector<int64_t> weightShape = {16, 8};
    std::vector<int64_t> labelsShape = {4};
    std::vector<int64_t> lossShape; // 0维标量
    std::vector<int64_t> maxOutShape = {4};
    std::vector<int64_t> sumOutShape = {4};
    std::vector<int64_t> dHShape = {4, 8};

    std::vector<uint16_t> hiddenHostData(32, FloatToFp16(0.1f));
    std::vector<uint16_t> weightHostData(128, FloatToFp16(0.01f));
    std::vector<int64_t> labelsHostData = {1, 2, -100, 0};
    std::vector<float> lossHostData = {0.0f};
    std::vector<float> maxOutHostData(4, 0.0f);
    std::vector<float> sumOutHostData(4, 0.0f);
    std::vector<uint16_t> dHHostData(32, 0);
    double alpha = 4.0;
    int64_t ignoreIndex = -100;

    void* hiddenDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* labelsDeviceAddr = nullptr;
    void* lossDeviceAddr = nullptr;
    void* maxOutDeviceAddr = nullptr;
    void* sumOutDeviceAddr = nullptr;
    void* dHDeviceAddr = nullptr;
    aclTensor* hidden = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* labels = nullptr;
    aclTensor* loss = nullptr;
    aclTensor* maxOut = nullptr;
    aclTensor* sumOut = nullptr;
    aclTensor* dH = nullptr;

    ret = CreateAclTensor(hiddenHostData, hiddenShape, &hiddenDeviceAddr, aclDataType::ACL_FLOAT16, &hidden);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT16, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(labelsHostData, labelsShape, &labelsDeviceAddr, aclDataType::ACL_INT64, &labels);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(maxOutHostData, maxOutShape, &maxOutDeviceAddr, aclDataType::ACL_FLOAT, &maxOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(sumOutHostData, sumOutShape, &sumOutDeviceAddr, aclDataType::ACL_FLOAT, &sumOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(dHHostData, dHShape, &dHDeviceAddr, aclDataType::ACL_FLOAT16, &dH);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    // 调用aclnnChunkLoss第一段接口
    ret = aclnnChunkLossGetWorkspaceSize(
        hidden, weight, labels, alpha, ignoreIndex, loss, maxOut, sumOut, dH, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnChunkLossGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnChunkLoss第二段接口
    ret = aclnnChunkLoss(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnChunkLoss failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧结果拷贝至host侧
    float result = 0.0f;
    ret = aclrtMemcpy(&result, sizeof(result), lossDeviceAddr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("loss is: %f\n", result);

    // 6. 释放aclTensor
    aclDestroyTensor(hidden);
    aclDestroyTensor(weight);
    aclDestroyTensor(labels);
    aclDestroyTensor(loss);
    aclDestroyTensor(maxOut);
    aclDestroyTensor(sumOut);
    aclDestroyTensor(dH);
    // 7. 释放device资源
    aclrtFree(hiddenDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(labelsDeviceAddr);
    aclrtFree(lossDeviceAddr);
    aclrtFree(maxOutDeviceAddr);
    aclrtFree(sumOutDeviceAddr);
    aclrtFree(dHDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
