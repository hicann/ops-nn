# aclnnConfidenceHeadFused

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：将hidden与可选的markovEmbOptional沿最后一维逻辑拼接，与weight做矩阵向量乘，加上可选的biasOptional，经temperatureOptional缩放、可选截断和Sigmoid计算后，输出每行的置信度。矩阵乘累加、后处理及输出的数据类型为FLOAT32。
- 计算公式：

  以下以三维输入为例；二维输入省略batch下标$b$。$K_1$为hidden的最后一维长度；markovEmbOptional为nullptr时$K_2=0$，否则$K_2$为其最后一维长度。weight的长度为$K_1+K_2$。

  1. 沿最后一维逻辑拼接后，计算矩阵向量乘并加偏置：

     $$
     z_{b,n}=\sum_{k=0}^{K_1-1}\mathrm{hidden}_{b,n,k}\,\mathrm{weight}_k
     +\sum_{k=0}^{K_2-1}\mathrm{markovEmbOptional}_{b,n,k}\,\mathrm{weight}_{K_1+k}+p_n
     $$

     $K_2=0$时第二项为0。biasOptional为nullptr时$p_n=0$，否则$p_n=\mathrm{biasOptional}_n$，并在各batch共用。

  2. 按温度缩放，得到指数函数的输入：

     $$
     T=\begin{cases}
     1, & \mathrm{temperatureOptional}=\mathrm{nullptr}\text{或}\mathrm{temperatureOptional}[0]=0,\\
     \mathrm{temperatureOptional}[0], & \text{其他情况},
     \end{cases}
     \qquad u_{b,n}=-\frac{z_{b,n}}{T}
     $$

  3. 先处理下限、再处理上限，最后计算Sigmoid：

     $$
     \begin{aligned}
     v_{b,n}&=\begin{cases}u_{b,n},&L=-C,\\\max(u_{b,n},L),&L\ne-C,\end{cases}\\
     w_{b,n}&=\begin{cases}v_{b,n},&H=C,\\\min(v_{b,n},H),&H\ne C,\end{cases}\\
     \mathrm{out}_{b,n}&=\frac{1}{1+\exp(w_{b,n})}
     \end{aligned}
     $$

     $C=\operatorname{FP32}(\mathrm{INT32\_MAX})=2147483648.0$。clampLowOptional为nullptr时$L=-C$，否则$L$为其FLOAT32值；clampHighOptional为nullptr时$H=C$，否则$H$为其FLOAT32值。显式传入约定值也会跳过对应的截断。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnConfidenceHeadFusedGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConfidenceHeadFused”接口执行计算。

```cpp
aclnnStatus aclnnConfidenceHeadFusedGetWorkspaceSize(
  const aclTensor *hidden,
  const aclTensor *weight,
  const aclTensor *markovEmbOptional,
  const aclTensor *temperatureOptional,
  const aclTensor *biasOptional,
  const aclScalar *clampLowOptional,
  const aclScalar *clampHighOptional,
  const aclTensor *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnConfidenceHeadFused(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

接口调用时须显式传入全部参数；不使用某个可选输入时，在对应位置传入nullptr。

## aclnnConfidenceHeadFusedGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
    <col style="width: 120px">
    <col style="width: 300px">
    <col style="width: 330px">
    <col style="width: 212px">
    <col style="width: 100px">
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
        <th>维度</th>
        <th>非连续</th>
      </tr></thead>
    <tbody>
      <tr>
        <td>hidden</td>
        <td>输入</td>
        <td>待投影的特征，对应公式中的hidden。</td>
        <td><li>必选，shape为 [N,K1] 或 [B,N,K1]。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>所有行共用的投影权重，对应公式中的weight。</td>
        <td><li>必选，shape为 [K1+K2]，数据类型与hidden一致。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>markovEmbOptional</td>
        <td>输入</td>
        <td>可选的Markov特征，对应公式中的markovEmbOptional。</td>
        <td><li>可选参数。传入nullptr时不拼接Markov特征，K2=0；非空时shape为[N,K2]或[B,N,K2]，rank、前导维度和数据类型与hidden一致。</li></td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>2、3</td>
        <td>√</td>
      </tr>
      <tr>
        <td>temperatureOptional</td>
        <td>输入</td>
        <td>温度系数，用于确定公式中的有效温度T。</td>
        <td><li>可选参数。非空时为[1] Tensor；传入nullptr或当前值为+0/-0时按温度1计算，其他值（含负值）使用当前值，不修改输入Tensor。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>按N维添加的偏置，用于确定公式中的p_n。</td>
        <td><li>可选参数。传入nullptr时不加偏置；非空时shape为[N]，同一n在各batch使用相同的biasOptional[n]。N&gt;1时不接受[1]广播。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>clampLowOptional</td>
        <td>输入</td>
        <td>指数输入的截断下限，对应公式中的L。</td>
        <td><li>可选参数。传入空指针时不做下限截断；传入数值须可转换为FLOAT32，转换后等于-2147483648.0时也不做下限截断。</li></td>
        <td>aclScalar</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>clampHighOptional</td>
        <td>输入</td>
        <td>指数输入的截断上限，对应公式中的H。</td>
        <td><li>可选参数。传入空指针时不做上限截断；传入数值须可转换为FLOAT32，转换后等于2147483648.0时也不做上限截断。</li></td>
        <td>aclScalar</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>每行的置信度，对应公式中的out。</td>
        <td><li>必选；由调用方创建，shape为 [N] 或 [B,N]，与hidden的前导维度一致。</li></td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>1、2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td>必选输出指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>必选输出指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景报错：

  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
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
        <td>传入的hidden、weight、out、workspaceSize或executor是空指针。</td>
      </tr>
      <tr>
        <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="4">161002</td>
        <td>当前平台不支持该接口。</td>
      </tr>
      <tr>
        <td>输入或输出的数据类型、数据格式、维度不满足参数说明。</td>
      </tr>
      <tr>
        <td>hidden、markovEmbOptional、weight、biasOptional或out的shape不匹配，或维度、展平行数及K1+K2超出支持范围。</td>
      </tr>
      <tr>
        <td>clampLowOptional或clampHighOptional的数据类型不能转换为FLOAT32。</td>
      </tr>
  </tbody></table>

## aclnnConfidenceHeadFused

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
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
        <td>在Device侧申请的workspace内存地址。第一段接口返回的workspace大小为0时，可传入nullptr。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnConfidenceHeadFusedGetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输入</td>
        <td>第一段接口返回的op执行器，包含算子计算流程。</td>
      </tr>
      <tr>
        <td>stream</td>
        <td>输入</td>
        <td>指定执行任务的Stream。</td>
      </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 支持合法基础格式和非连续Tensor，不支持FRACTAL_NZ。
- hidden、weight、markovEmbOptional的数据类型须同为FLOAT16或同为BFLOAT16；temperatureOptional、biasOptional和out为FLOAT32。weight的长度为K1+K2，markovEmbOptional的前导维度与hidden一致，biasOptional的shape为[N]，out的shape与hidden去掉最后一维后的shape一致。
- 各维度、B*N和K1+K2须位于[0,UINT32_MAX]。二维输入按B=1理解。
- K1+K2=0时仍执行biasOptional、temperatureOptional、可选截断和Sigmoid计算；B=0或N=0时输出为空。
- 确定性计算：aclnnConfidenceHeadFused默认支持确定性计算。

## 调用示例

示例使用FP16三维特征、Markov特征、biasOptional[N] 和值为2的temperatureOptional；两个batch的预期输出均约为 [0.5621765,0.6791787,0.7772999]。FP16数据按真实16位编码传入。切换为BF16时，将特征常数1的编码改为0x3F80，权重常数0.5的编码改为0x3F00，并将对应Tensor的dtype改为ACL_BF16。

完整示例请参见 [test_aclnn_confidence_head_fused.cpp](../examples/arch35/test_aclnn_confidence_head_fused.cpp)。编译与执行过程参见[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)，通过`--soc=ascend950`选择目标芯片。

```cpp
#include <cstdint>
#include <cstdio>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_confidence_head_fused.h"

#define CHECK_RET(expr)                          \
    do {                                        \
        auto ret = (expr);                       \
        if (ret != ACL_SUCCESS) {               \
            std::printf("error: %d\n", ret);    \
            return 1;                           \
        }                                       \
    } while (0)

struct Runtime {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;
    ~Runtime()
    {
        if (stream != nullptr) { aclrtDestroyStream(stream); }
        if (deviceSet) { aclrtResetDevice(0); }
        if (initialized) { aclFinalize(); }
    }
};

struct DeviceTensor {
    void* data = nullptr;
    aclTensor* tensor = nullptr;
    ~DeviceTensor()
    {
        if (tensor != nullptr) { aclDestroyTensor(tensor); }
        if (data != nullptr) { aclrtFree(data); }
    }

    template <typename T>
    int Create(const std::vector<T>& values, const std::vector<int64_t>& shape, aclDataType dtype)
    {
        size_t bytes = values.size() * sizeof(T);
        CHECK_RET(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_RET(aclrtMemcpy(data, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
        std::vector<int64_t> strides(shape.size(), 1);
        for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0,
                                 ACL_FORMAT_ND, shape.data(), shape.size(), data);
        return tensor == nullptr ? 1 : 0;
    }
};

struct Workspace {
    void* data = nullptr;
    ~Workspace() { if (data != nullptr) { aclrtFree(data); } }
};

int main()
{
    Runtime runtime;
    CHECK_RET(aclInit(nullptr));
    runtime.initialized = true;
    CHECK_RET(aclrtSetDevice(0));
    runtime.deviceSet = true;
    CHECK_RET(aclrtCreateStream(&runtime.stream));

    DeviceTensor hidden, weight, markov, temperatureOptional, biasOptional, out;
    CHECK_RET(hidden.Create(std::vector<uint16_t>(12, 0x3C00), {2, 3, 2}, ACL_FLOAT16));
    CHECK_RET(weight.Create(std::vector<uint16_t>(3, 0x3800), {3}, ACL_FLOAT16));
    CHECK_RET(markov.Create(std::vector<uint16_t>(6, 0x3C00), {2, 3, 1}, ACL_FLOAT16));
    CHECK_RET(temperatureOptional.Create(std::vector<float>{2.0f}, {1}, ACL_FLOAT));
    CHECK_RET(biasOptional.Create(std::vector<float>{-1.0f, 0.0f, 1.0f}, {3}, ACL_FLOAT));
    CHECK_RET(out.Create(std::vector<float>(6, 0.0f), {2, 3}, ACL_FLOAT));

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_RET(aclnnConfidenceHeadFusedGetWorkspaceSize(hidden.tensor, weight.tensor, markov.tensor,
                 temperatureOptional.tensor, biasOptional.tensor, nullptr, nullptr, out.tensor, &workspaceSize, &executor));
    Workspace workspace;
    if (workspaceSize != 0) {
        CHECK_RET(aclrtMalloc(&workspace.data, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    auto status = aclnnConfidenceHeadFused(workspace.data, workspaceSize, executor, runtime.stream);
    // Synchronize before releasing device buffers, including the execution-error path.
    auto syncStatus = aclrtSynchronizeStream(runtime.stream);
    CHECK_RET(status);
    CHECK_RET(syncStatus);
    std::vector<float> result(6);
    CHECK_RET(aclrtMemcpy(result.data(), result.size() * sizeof(float), out.data,
                         result.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (float value : result) { std::printf("%.7f\n", value); }
    return 0;
}
```
