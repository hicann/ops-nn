# aclnnGRU

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | :------:|
| <term>Ascend 950PR/Ascend 950DT</term>                            |     ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |     √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
| <term>Atlas 推理系列产品</term>                             |     ×     |
| <term>Atlas 训练系列产品</term>                              |     ×   |

## 功能说明

- 接口功能：实现门控循环单元（Gated Recurrent Unit, GRU）计算，支持多层堆叠、双向、定长序列和不定长序列（PackedSequence）两种输入模式。训练模式下可输出各门控中间结果（r、z、n、hn、h），用于反向传播。

- 输入输出shape：
  - 定长模式（batchSizes为nullptr）：如果输入张量 input 的shape为(T, B, I)（batchFirst=false）或(B, T, I)（batchFirst=true），则输出张量 output 的shape为(T, B, D *H)（batchFirst=false）或(B, T, D* H)（batchFirst=true），最终隐状态 hy 的shape为(L * D, B, H)。
  - 不定长模式（batchSizes非nullptr）：输入张量 input 的shape为(sum(batch_size), I)，输出张量 output 的shape为(sum(batch_size), D *H)，最终隐状态 hy 的shape为(L* D, B, H)。当前暂不支持该模式。

- 计算公式：

  $$
  r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
  $$

  $$
  z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
  $$

  $$
  n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{(t-1)} + b_{hn}))
  $$

  $$
  h_t = (1 - z_t) \odot n_t + z_t \odot h_{(t-1)}
  $$

  其中：
  - $\sigma$ 为 sigmoid 激活函数。
  - $\odot$ 表示逐元素乘法（Hadamard积）。
  - $r_t$、$z_t$、$n_t$ 分别为重置门、更新门和新门。
  - $h_t$ 为当前时刻的隐状态。
  - $W_{ir}$、$W_{iz}$、$W_{in}$ 为输入权重矩阵，形状为 $[H, I]$（转置存储为 $[3H, I]$）。
  - $W_{hr}$、$W_{hz}$、$W_{hn}$ 为隐状态权重矩阵，形状为 $[H, H]$（转置存储为 $[3H, H]$）。
  - 若 hasBias=true，则 $b_{ir}$、$b_{iz}$、$b_{in}$ 和 $b_{hr}$、$b_{hz}$、$b_{hn}$ 分别为输入偏置和隐状态偏置，形状均为 $[H]$（存储为 $[3H]$）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnGRUGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnGRU"接口执行计算。

```cpp
aclnnStatus aclnnGRUGetWorkspaceSize(
  const aclTensor     *input,
  const aclTensorList *params,
  const aclTensor     *hx,
  const aclTensor     *batchSizes,
  bool                 hasBias,
  int64_t              numLayers,
  double               dropout,
  bool                 train,
  bool                 bidirection,
  bool                 batchFirst,
  aclTensor           *output,
  aclTensor           *hy,
  aclTensorList       *rOut,
  aclTensorList       *zOut,
  aclTensorList       *nOut,
  aclTensorList       *hnOut,
  aclTensorList       *hOut,
  uint64_t            *workspaceSize,
  aclOpExecutor      **executor)
```

```cpp
aclnnStatus aclnnGRU(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnGRUGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
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
      <td>input（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入的序列数据，对应公式中的 $x_t$。</td>
      <td><ul><li>不支持空Tensor。</li><li>定长模式：shape为(T, B, I)或(batchFirst=true时为(B, T, I))。</li><li>不定长模式（batchSizes非nullptr）：shape为(sum(batch_size), I)。</li></ul></td>
      <td>FLOAT32、FLOAT16</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>params（aclTensorList*）</td>
      <td>输入</td>
      <td>表示权重和偏置参数列表，每个元素对应公式中的 $W_{ir}$、$W_{iz}$、$W_{in}$、$W_{hr}$、$W_{hz}$、$W_{hn}$ 及对应的偏置。</td>
      <td><ul><li>不支持空Tensor。</li><li>列表长度 = 2 * bScale * dScale * numLayers，其中 bScale = hasBias ? 2 : 1，dScale = bidirection ? 2 : 1。</li><li>排列顺序：先层后方向，每组内依次为 $W_{ih}$、$W_{hh}$、$b_{ih}$（有偏置时）、$b_{hh}$（有偏置时）。</li><li>$W_{ih}$ 形状为 $[3H, I]$（首层）或 $[3H, D*H]$（非首层），$W_{hh}$ 形状为 $[3H, H]$，偏置形状为 $[3H]$。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2（权重）、1（偏置）</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hx（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示初始隐状态，对应公式中的 $h_0$。</td>
      <td><ul><li>可选参数，传入空指针表示不使用该参数，此时初始隐状态为零向量。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>3 (L*D, B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>batchSizes（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示不定长序列的batch大小数组，对应PackedSequence模式。</td>
      <td><ul><li>可选参数，传入空指针表示使用定长模式。</li><li>传入非空指针时，进入PackedSequence模式，input形状必须为2D (sum(batch_size), I)。当前暂不支持传入非空指针。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1 (T)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hasBias（bool）</td>
      <td>输入</td>
      <td>表示是否使用偏置，对应公式中的 $b_{ir}$、$b_{iz}$、$b_{in}$、$b_{hr}$、$b_{hz}$、$b_{hn}$。</td>
      <td>hasBias为True，则params列表包含偏置参数，否则不包含。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>numLayers（int64_t）</td>
      <td>输入</td>
      <td>表示GRU的堆叠层数，对应公式中的 $L$。</td>
      <td>取值范围为大于等于1的整数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dropout（double）</td>
      <td>输入</td>
      <td>表示dropout比例。</td>
      <td>当前版本未实现dropout功能，传入任意值不影响计算结果。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>train（bool）</td>
      <td>输入</td>
      <td>表示是否为训练模式。</td>
      <td>train为True时，需要传入rOut、zOut、nOut、hnOut、hOut进行门控输出；train为False时，对应参数传入空指针。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>bidirection（bool）</td>
      <td>输入</td>
      <td>表示是否为双向GRU。</td>
      <td>bidirection为True时，每个时间步同时计算正向和反向，输出维度为2*H。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>batchFirst（bool）</td>
      <td>输入</td>
      <td>表示输入和输出的batch维度是否在第一维。</td>
      <td>batchFirst为True时，input和output的shape为(B, T, I)和(B, T, D*H)；为False时，shape为(T, B, I)和(T, B, D*H)。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output（aclTensor*）</td>
      <td>输出</td>
      <td>表示所有时间步的输出，对应公式中的 $h_t$。</td>
      <td><ul><li>不支持空Tensor。</li><li>定长模式：shape为(T, B, D*H)或(batchFirst=true时为(B, T, D*H))。</li><li>不定长模式：shape为(sum(batch_size), D*H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hy（aclTensor*）</td>
      <td>输出</td>
      <td>表示最后一层所有方向的最终隐状态，对应公式中最后时刻的 $h_t$。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为(L*D, B, H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>3 (L*D, B, H)</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rOut（aclTensorList*）</td>
      <td>可选输出</td>
      <td>表示训练模式下的重置门输出，对应公式中的 $r_t$。</td>
      <td><ul><li>训练模式下必传，推理模式下传入空指针。</li><li>列表长度 = dScale * numLayers。</li><li>每个元素的shape：定长模式为(T, B, H)，不定长模式为(sum(batch_size), H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>zOut（aclTensorList*）</td>
      <td>可选输出</td>
      <td>表示训练模式下的更新门输出，对应公式中的 $z_t$。</td>
      <td><ul><li>训练模式下必传，推理模式下传入空指针。</li><li>列表长度 = dScale * numLayers。</li><li>每个元素的shape：定长模式为(T, B, H)，不定长模式为(sum(batch_size), H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>nOut（aclTensorList*）</td>
      <td>可选输出</td>
      <td>表示训练模式下的新门输出，对应公式中的 $n_t$。</td>
      <td><ul><li>训练模式下必传，推理模式下传入空指针。</li><li>列表长度 = dScale * numLayers。</li><li>每个元素的shape：定长模式为(T, B, H)，不定长模式为(sum(batch_size), H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hnOut（aclTensorList*）</td>
      <td>可选输出</td>
      <td>表示训练模式下的隐状态-新门输出 $W_{hn} h_{(t-1)} + b_{hn}$。</td>
      <td><ul><li>训练模式下必传，推理模式下传入空指针。</li><li>列表长度 = dScale * numLayers。</li><li>每个元素的shape：定长模式为(T, B, H)，不定长模式为(sum(batch_size), H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>hOut（aclTensorList*）</td>
      <td>可选输出</td>
      <td>表示训练模式下的隐状态输出，即各时间步的 $h_t$。</td>
      <td><ul><li>训练模式下必传，推理模式下传入空指针。</li><li>列表长度 = dScale * numLayers。</li><li>每个元素的shape：定长模式为(T, B, H)，不定长模式为(sum(batch_size), H)。</li></ul></td>
      <td>数据类型与input保持一致。</td>
      <td>ND</td>
      <td>2-3</td>
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

- **返回值**

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
      <td>传入的input、params、output或hy是空指针，或训练模式下rOut、zOut、nOut、hnOut、hOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>input、params中元素、hx、output、hy或训练模式下的门控输出数据类型不在支持范围内（仅支持FLOAT32、FLOAT16）。</td>
    </tr>
    <tr>
      <td>input、params中元素、hx、output、hy或训练模式下的门控输出数据类型不一致。</td>
    </tr>
    <tr>
      <td>params列表长度不正确（应为 2 * bScale * dScale * numLayers）。</td>
    </tr>
    <tr>
      <td>训练模式下rOut、zOut、nOut、hnOut、hOut列表长度不正确（应为 dScale * numLayers）。</td>
    </tr>
    <tr>
      <td>input维度不是2维或3维。</td>
    </tr>
    <tr>
      <td>params中权重维度不是2维，或偏置维度不是1维。</td>
    </tr>
    <tr>
      <td>hx维度不是3维，或output、hy维度不是3维（定长模式）。</td>
    </tr>
    <tr>
      <td>params中权重或偏置的shape不符合预期：$W_{ih}$ 首层应为 $[3H, I]$、非首层 $[3H, D*H]$，$W_{hh}$ 应为 $[3H, H]$，偏置应为 $[3H]$。</td>
    </tr>
    <tr>
      <td>batchSizes不是空指针，当前暂不支持不定长模式。</td>
    </tr>
  </tbody></table>

## aclnnGRU

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGRUGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：aclnnGRU默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_gru.h"

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

void PrintOutResult(const std::string& name, const std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<float> resultData(size, 0);
    auto ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy %s from device to host failed. ERROR: %d\n", name.c_str(), ret); return);
    LOG_PRINT("=== %s shape=[", name.c_str());
    for (size_t i = 0; i < shape.size(); i++) {
        LOG_PRINT("%ld%s", shape[i], (i + 1 < shape.size()) ? "," : "");
    }
    LOG_PRINT("] ===\n");
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("  [%ld] = %f\n", i, resultData[i]);
    }
    LOG_PRINT("\n");
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    // 固定写法，AscendCL初始化
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
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

template <typename T>
int CreateAclTensorList(
    const std::vector<std::vector<int64_t>>& shapes, void** deviceAddr, aclDataType dataType, aclTensorList** tensor,
    T initVal = 1)
{
    int size = shapes.size();
    aclTensor* tensors[size];
    for (int i = 0; i < size; i++) {
        std::vector<T> hostData(GetShapeSize(shapes[i]), initVal);
        int ret = CreateAclTensor<float>(hostData, shapes[i], deviceAddr + i, dataType, tensors + i);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    *tensor = aclCreateTensorList(tensors, size);
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    int64_t timeStep = 2;
    int64_t batchSize = 3;
    int64_t inputSize = 4;
    int64_t hiddenSize = 5;
    int64_t gateNum = 3;
    int64_t numLayers = 1;
    bool isbias = false;
    bool batchFirst = false;
    bool bidirection = false;
    bool isTraining = true;
    double dropout = 0.0;
    int64_t dScale = bidirection ? 2 : 1;
    int64_t ldScale = numLayers * dScale;

    std::vector<int64_t> inputShape = {timeStep, batchSize, inputSize};
    std::vector<int64_t> outputShape = {timeStep, batchSize, dScale * hiddenSize};
    std::vector<int64_t> hyShape = {ldScale, batchSize, hiddenSize};
    std::vector<int64_t> hxShape = {ldScale, batchSize, hiddenSize};
    std::vector<std::vector<int64_t>> paramsListShape = {};

    auto curLayerInputSize = inputSize;
    for (int i = 0; i < numLayers; i++) {
        for (int64_t j = 0; j < dScale; j++) {
            paramsListShape.push_back({hiddenSize * gateNum, curLayerInputSize});
            paramsListShape.push_back({hiddenSize * gateNum, hiddenSize});
            if (isbias) {
                paramsListShape.push_back({hiddenSize * gateNum});
                paramsListShape.push_back({hiddenSize * gateNum});
            }
        }
        curLayerInputSize = dScale * hiddenSize;
    }

    std::vector<std::vector<int64_t>> rOutListShape;
    std::vector<std::vector<int64_t>> zOutListShape;
    std::vector<std::vector<int64_t>> nOutListShape;
    std::vector<std::vector<int64_t>> hnOutListShape;
    std::vector<std::vector<int64_t>> hOutListShape;

    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            rOutListShape.push_back({timeStep, batchSize, hiddenSize});
            zOutListShape.push_back({timeStep, batchSize, hiddenSize});
            nOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hnOutListShape.push_back({timeStep, batchSize, hiddenSize});
            hOutListShape.push_back({timeStep, batchSize, hiddenSize});
        }
    }

    void* inputDeviceAddr = nullptr;
    std::vector<void*> paramsListDeviceAddr(paramsListShape.size(), nullptr);
    void* outputDeviceAddr = nullptr;
    void* hyDeviceAddr = nullptr;
    void* hxDeviceAddr = nullptr;

    std::vector<void*> rOutDeviceAddr;
    std::vector<void*> zOutDeviceAddr;
    std::vector<void*> nOutDeviceAddr;
    std::vector<void*> hnOutDeviceAddr;
    std::vector<void*> hOutDeviceAddr;

    aclTensor* input = nullptr;
    aclTensorList* params = nullptr;
    aclTensor* output = nullptr;
    aclTensor* hy = nullptr;
    aclTensor* hx = nullptr;

    aclTensorList* rOut = nullptr;
    aclTensorList* zOut = nullptr;
    aclTensorList* nOut = nullptr;
    aclTensorList* hnOut = nullptr;
    aclTensorList* hOut = nullptr;

    std::vector<float> inputHostData(GetShapeSize(inputShape), 1.0);
    ret = CreateAclTensor<float>(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensorList<float>(paramsListShape, paramsListDeviceAddr.data(), aclDataType::ACL_FLOAT, &params, 1.0);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> outputHostData(GetShapeSize(outputShape), 0.0);
    ret = CreateAclTensor<float>(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    std::vector<float> hyHostData(GetShapeSize(hyShape), 0.0);
    ret = CreateAclTensor<float>(hyHostData, hyShape, &hyDeviceAddr, aclDataType::ACL_FLOAT, &hy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    bool useHx = false;
    if (useHx) {
        std::vector<float> hxHostData(GetShapeSize(hxShape), 0.5);
        ret = CreateAclTensor<float>(hxHostData, hxShape, &hxDeviceAddr, aclDataType::ACL_FLOAT, &hx);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    if (isTraining) {
        rOutDeviceAddr.resize(ldScale, nullptr);
        zOutDeviceAddr.resize(ldScale, nullptr);
        nOutDeviceAddr.resize(ldScale, nullptr);
        hnOutDeviceAddr.resize(ldScale, nullptr);
        hOutDeviceAddr.resize(ldScale, nullptr);

        ret = CreateAclTensorList<float>(rOutListShape, rOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &rOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(zOutListShape, zOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &zOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(nOutListShape, nOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &nOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hnOutListShape, hnOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hnOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        ret = CreateAclTensorList<float>(hOutListShape, hOutDeviceAddr.data(), aclDataType::ACL_FLOAT, &hOut, 0.0);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    // 3. 调用aclnnGRU第一段接口
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnGRUGetWorkspaceSize(
        input, params, (useHx ? hx : nullptr), nullptr, isbias, numLayers, dropout, isTraining, bidirection, batchFirst, output, hy,
        isTraining ? rOut : nullptr,
        isTraining ? zOut : nullptr,
        isTraining ? nOut : nullptr,
        isTraining ? hnOut : nullptr,
        isTraining ? hOut : nullptr,
        &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRUGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 4. 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 5. 调用aclnnGRU第二段接口
    ret = aclnnGRU(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGRU failed. ERROR: %d\n", ret); return ret);

    // 6. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 7. 获取输出的值
    PrintOutResult("output", outputShape, &outputDeviceAddr);
    PrintOutResult("hy", hyShape, &hyDeviceAddr);
    if (isTraining) {
        for (int64_t i = 0; i < ldScale; i++) {
            PrintOutResult("rOut[" + std::to_string(i) + "]", rOutListShape[i], &rOutDeviceAddr[i]);
            PrintOutResult("zOut[" + std::to_string(i) + "]", zOutListShape[i], &zOutDeviceAddr[i]);
            PrintOutResult("nOut[" + std::to_string(i) + "]", nOutListShape[i], &nOutDeviceAddr[i]);
            PrintOutResult("hnOut[" + std::to_string(i) + "]", hnOutListShape[i], &hnOutDeviceAddr[i]);
            PrintOutResult("hOut[" + std::to_string(i) + "]", hOutListShape[i], &hOutDeviceAddr[i]);
        }
    }

    // 释放资源
    aclDestroyTensor(input);
    aclDestroyTensorList(params);
    aclDestroyTensor(output);
    aclDestroyTensor(hy);
    if (hx != nullptr) {
        aclDestroyTensor(hx);
    }
    if (isTraining) {
        aclDestroyTensorList(rOut);
        aclDestroyTensorList(zOut);
        aclDestroyTensorList(nOut);
        aclDestroyTensorList(hnOut);
        aclDestroyTensorList(hOut);
    }
    aclrtFree(inputDeviceAddr);
    for (size_t i = 0; i < paramsListShape.size(); i++) {
        aclrtFree(paramsListDeviceAddr[i]);
    }
    aclrtFree(outputDeviceAddr);
    aclrtFree(hyDeviceAddr);
    if (hxDeviceAddr != nullptr) {
        aclrtFree(hxDeviceAddr);
    }
    if (isTraining) {
        for (size_t i = 0; i < rOutDeviceAddr.size(); i++) {
            aclrtFree(rOutDeviceAddr[i]);
            aclrtFree(zOutDeviceAddr[i]);
            aclrtFree(nOutDeviceAddr[i]);
            aclrtFree(hnOutDeviceAddr[i]);
            aclrtFree(hOutDeviceAddr[i]);
        }
    }
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
