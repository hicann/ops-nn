# aclnnChunkLossBackward

## 产品支持情况

<!-- npu="950" id1 -->

- <term>Ascend 950PR&950DT系列产品</term>：支持

<!-- end id1 -->

<!-- npu="A3" id2 -->

- <term>Atlas A3系列产品</term>：支持

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

- 接口功能：计算ChunkLoss中投影权重的梯度。根据隐藏状态hidden、投影权重weight、标签labels及正向保存的Softmax统计量，重算词表得分，输出经归一化和上游梯度缩放后的gradWeightOut。
- 计算公式：

  首先重新计算每个token对应的词表得分：

  $$
  \ell_{ij}=\sum_{k=0}^{H-1}h_{ik}W_{jk}
  $$

  结合正向保存的逐token最大值和减去最大值后的指数和进行Softmax归一化：

  $$
  m_i=\max_{0\le j<V}\ell_{ij},\qquad
  s_i=\sum_{j=0}^{V-1}\exp(\ell_{ij}-m_i),\qquad
  p_{ij}=\frac{\exp(\ell_{ij}-m_i)}{s_i}
  $$

  计算交叉熵对词表得分的梯度，忽略标签对应的整行梯度为0：

  $$
  g_{ij}=\begin{cases}
  0,&y_i=\mathrm{ignoreIndex}\\
  \alpha\left(p_{ij}-\mathbf{1}[j=y_i]\right),&y_i\ne\mathrm{ignoreIndex}
  \end{cases}
  $$

  沿token维度归约，输出经过上游梯度缩放的权重梯度：

  $$
  D_{jk}=q\sum_{i=0}^{N-1}g_{ij}h_{ik}
  $$

  其中，$h$表示hidden，$W$表示weight，$y$表示labels，$\alpha$表示alpha，$m$表示softmaxMax，$s$表示softmaxSum，$q$表示gradOutput中的唯一元素，$D$表示gradWeightOut。

  - $N$、$H$、$V$分别为token数量、隐藏维度和词表大小。
  - $\ell$、$p$、$g$分别为重算得分、概率和得分梯度，不作为接口输出。
  - $\mathbf{1}[j=y_i]$为指示函数，在$j=y_i$时为1，否则为0。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnChunkLossBackwardGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnChunkLossBackward`接口执行计算。

```cpp
aclnnStatus aclnnChunkLossBackwardGetWorkspaceSize(
  const aclTensor *hidden,
  const aclTensor *weight,
  const aclTensor *labels,
  double           alpha,
  const aclTensor *softmaxMax,
  const aclTensor *softmaxSum,
  const aclTensor *gradOutput,
  int64_t          ignoreIndex,
  aclTensor       *gradWeightOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnChunkLossBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnChunkLossBackwardGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1500px">
  <colgroup>
    <col style="width: 220px">
    <col style="width: 80px">
    <col style="width: 200px">
    <col style="width: 350px">
    <col style="width: 180px">
    <col style="width: 70px">
    <col style="width: 150px">
    <col style="width: 150px">
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
      <td>隐藏状态，对应公式中的h。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>数据类型与weight、gradWeightOut一致。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[N,H]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight（const aclTensor*）</td>
      <td>输入</td>
      <td>投影权重，对应公式中的W。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>第二维与hidden的第二维一致。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[V,H]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>labels（const aclTensor*）</td>
      <td>输入</td>
      <td>每个token的目标类别索引，对应公式中的y。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>非ignoreIndex项须满足0≤labels[i]<V。</li>
        </ul>
      </td>
      <td>INT64</td>
      <td>ND</td>
      <td>[N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha（double）</td>
      <td>输入</td>
      <td>损失的归一化系数，对应公式中的α，等于正向损失归一化分母的倒数。</td>
      <td>
        <ul>
          <li>Host侧标量，须为有限正数，且不超过FLOAT32最大有限值。</li>
          <li>转换为FLOAT32后仍须大于0，计算时使用该转换值。</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>softmaxMax（const aclTensor*）</td>
      <td>输入</td>
      <td>正向保存的逐token词表得分最大值，对应公式中的m。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>须为有限值，并与softmaxSum配对使用。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>softmaxSum（const aclTensor*）</td>
      <td>输入</td>
      <td>正向保存的逐token稳定化指数和，对应公式中的s。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>须为有限正值；不接受logsumexp或未减去最大值的指数和。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput（const aclTensor*）</td>
      <td>输入</td>
      <td>损失的上游梯度，对应公式中的q。</td>
      <td>
        <ul>
          <li>不支持空Tensor。</li>
          <li>须为一维单元素Tensor，不支持零维Tensor或按token广播。</li>
          <li>元素须为有限值，可为正、负或零；单独求当前损失的梯度时传入1。</li>
        </ul>
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[1]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>ignoreIndex（int64_t）</td>
      <td>输入</td>
      <td>不参与权重梯度计算的标签值，对应公式中的ignoreIndex。</td>
      <td>
        <ul>
          <li>Host侧标量，调用时须显式传入。</li>
          <li>须与正向一致，取值可位于[0,V)内或范围外；匹配标签的token不贡献权重梯度。</li>
        </ul>
      </td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradWeightOut（aclTensor*）</td>
      <td>输出</td>
      <td>经归一化和上游梯度缩放后的权重梯度，对应公式中的D。</td>
      <td>
        <ul>
          <li>不支持空Tensor，须由调用方预分配。</li>
          <li>shape和数据类型与weight一致。</li>
          <li>输出存储不得与任一输入重叠，且输出元素之间不得共享存储位置。</li>
        </ul>
      </td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>[V,H]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor**）</td>
      <td>输出</td>
      <td>返回包含算子计算流程的执行器。</td>
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
<colgroup><col style="width: 260px"><col style="width: 100px"><col style="width: 640px"></colgroup>
<thead><tr><th>返回值</th><th>错误码</th><th>描述</th></tr></thead>
<tbody>
<tr><td>ACLNN_ERR_PARAM_NULLPTR</td><td>161001</td><td>任一必选Tensor参数、workspaceSize或executor为空指针。</td></tr>
<tr><td rowspan="5">ACLNN_ERR_PARAM_INVALID</td><td rowspan="5">161002</td><td>hidden、weight、labels、softmaxMax、softmaxSum、gradOutput、gradWeightOut的数据类型不在支持的范围内。</td></tr>
<tr><td>hidden、weight、labels、softmaxMax、softmaxSum、gradOutput、gradWeightOut的数据格式不在支持的范围内。</td></tr>
<tr><td>hidden、weight、labels、softmaxMax、softmaxSum、gradOutput、gradWeightOut的shape不在支持的范围内。</td></tr>
<tr><td>hidden、weight、gradWeightOut的数据类型不一致。</td></tr>
<tr><td>alpha非有限、非正、超过FLOAT32最大有限值，或转换为FLOAT32后为0。</td></tr>
</tbody></table>



## aclnnChunkLossBackward

- **参数说明**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnChunkLossBackwardGetWorkspaceSize获取。</td>
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

- 所有Tensor均为必选参数，采用ND格式，位于同一Device，不支持空Tensor。`hidden`、`weight`和`gradWeightOut`的数据类型须相同。
- N、H、V均为正整数且不超过2147483647，N×H和V×H均不超过⌊(2^63−1)/4⌋。
- 非忽略标签须满足0 ≤ labels[i] < V。
- `alpha`为正向损失归一化分母的倒数，须为有限正数且不超过FLOAT32最大有限值，转换为FLOAT32后仍须大于0。跨chunk调用时共用同一全局`alpha`，由调用方累加各次输出。
- `softmaxMax`和`softmaxSum`须为有限正值。
- `gradOutput`的shape为[1]，支持正、负或零值。
- 确定性计算：

  - aclnnChunkLossBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_chunk_loss_backward.h"

#define LOG_PRINT(...) std::printf(__VA_ARGS__)
#define CHECK_RET(cond, action) do { if (!(cond)) { action; } } while (0)

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> allocations;
    ~Resources()
    {
        if (stream != nullptr) { aclrtSynchronizeStream(stream); }
        for (auto tensor : tensors) { aclDestroyTensor(tensor); }
        for (auto address : allocations) { aclrtFree(address); }
        if (stream != nullptr) { aclrtDestroyStream(stream); }
        if (deviceSet) { aclrtResetDevice(0); }
        if (initialized) { aclFinalize(); }
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t size = 1;
    for (auto dim : shape) { size *= dim; }
    return size;
}

bool Init(Resources& resources)
{
    CHECK_RET(aclInit(nullptr) == ACL_SUCCESS, return false);
    resources.initialized = true;
    CHECK_RET(aclrtSetDevice(0) == ACL_SUCCESS, return false);
    resources.deviceSet = true;
    return aclrtCreateStream(&resources.stream) == ACL_SUCCESS;
}

template <typename T>
aclTensor* CreateAclTensor(const std::vector<T>& data, const std::vector<int64_t>& shape,
    aclDataType dtype, Resources& resources, void** deviceAddress = nullptr)
{
    auto bytes = GetShapeSize(shape) * sizeof(T);
    void* address = nullptr;
    CHECK_RET(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return nullptr);
    resources.allocations.push_back(address);
    CHECK_RET(aclrtMemcpy(address, bytes, data.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS,
        return nullptr);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    auto tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
        ACL_FORMAT_ND, shape.data(), shape.size(), address);
    if (tensor != nullptr) { resources.tensors.push_back(tensor); }
    if (deviceAddress != nullptr) { *deviceAddress = address; }
    return tensor;
}

int main()
{
    // 1. 初始化Device和Stream，资源在退出时统一释放。
    Resources resources;
    CHECK_RET(Init(resources), LOG_PRINT("ACL initialization failed\n"); return 1);
    // 2. 构造N=1、H=1、V=2的例子。uint16_t保存FLOAT16的原始位表示。
    auto hidden = CreateAclTensor<uint16_t>({0x3c00}, {1, 1}, ACL_FLOAT16, resources); // 1.0
    auto weight = CreateAclTensor<uint16_t>({0, 0}, {2, 1}, ACL_FLOAT16, resources);
    auto labels = CreateAclTensor<int64_t>({0}, {1}, ACL_INT64, resources);
    const double alpha = 1.0;
    // logits全为0，因此softmaxMax=0、softmaxSum=exp(0)+exp(0)=2。
    auto softmaxMax = CreateAclTensor<float>({0.0f}, {1}, ACL_FLOAT, resources);
    auto softmaxSum = CreateAclTensor<float>({2.0f}, {1}, ACL_FLOAT, resources);
    auto gradOutput = CreateAclTensor<float>({1.0f}, {1}, ACL_FLOAT, resources);
    void* outputAddress = nullptr;
    auto output = CreateAclTensor<uint16_t>({0, 0}, {2, 1}, ACL_FLOAT16, resources, &outputAddress);
    CHECK_RET(hidden && weight && labels && softmaxMax && softmaxSum && gradOutput && output,
        LOG_PRINT("Tensor construction failed\n"); return 1);
    // 3. 查询workspace并创建执行器。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    auto status = aclnnChunkLossBackwardGetWorkspaceSize(hidden, weight, labels, alpha,
        softmaxMax, softmaxSum, gradOutput, -100, output, &workspaceSize, &executor);
    CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed: %d\n", status); return 1);
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_RET(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS, return 1);
        resources.allocations.push_back(workspace);
    }
    // 4. 异步执行并等待结果。
    status = aclnnChunkLossBackward(workspace, workspaceSize, executor, resources.stream);
    CHECK_RET(status == ACL_SUCCESS, LOG_PRINT("Backward failed: %d\n", status); return 1);
    CHECK_RET(aclrtSynchronizeStream(resources.stream) == ACL_SUCCESS, return 1);
    // 5. 拷贝输出，预期FLOAT16位表示为[-0.5, 0.5]，即[0xb800, 0x3800]。
    uint16_t result[2] = {};
    CHECK_RET(aclrtMemcpy(result, sizeof(result), outputAddress, sizeof(result),
        ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS, return 1);
    LOG_PRINT("dW bits: [0x%04x, 0x%04x], expected [0xb800, 0x3800]\n", result[0], result[1]);
    return result[0] == 0xb800 && result[1] == 0x3800 ? 0 : 1;
}
```
