# aclnnRotateQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/rotate_quant)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    √     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- 接口功能：对张量x进行旋转变换，然后执行可选的clamp操作，最后执行对称动态量化（目的数据类型为int8或者quint4x2）或者MX量化（目的数据类型为FLOAT4类、FLOAT8类）。
- 计算公式：
  1. 旋转变换

    $$
    Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
    $$

    其中：$\mathbf{x} \in \mathbb{R}^{m \times n}$，$\mathbf{Y} \in \mathbb{R}^{m \times n}$，$\mathbf{rotation} \in \mathbb{R}^{k \times k}$。

  2. 当alpha在有效取值范围(0.0, 1.0)内，执行clamp计算

    $$
    groupMaxVal = GroupMax(|Y|) \\
    limit = alpha * groupMaxVal \\
    Y = Y.clamp(min=-limit, max=limit)
    $$

    其中：GroupMax表示每32个为一组，计算组内最大值。

  3. 执行量化
    - <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：对称动态量化（pertoken逐行量化）
      - 缩放因子计算（逐行计算）

        $$
        s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
        $$

        其中：$s_i$ 是第 $i$ 行的缩放因子；$C_{\text{MAX}}$ 是量化范围最大值，int8取127，quint4x2取7。
      - 量化计算

        $$
        y_{i,j} = \frac{Y_{i,j}}{s_i}
        $$

    - <term>Ascend 950PR/Ascend 950DT</term>：MX量化

      - 场景1，当scaleAlg为0时：
        - 将输入x在axis维度上按k = 32个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32

          $$
          shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
          mxscale = 2^{shared\_exp}\\
          P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
          $$

        - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

        - emax: 对应数据类型的最大正则数的指数位。

            |   DataType    | emax |
            | :-----------: | :--: |
            |  FLOAT4_E2M1  |  2   |
            | FLOAT8_E4M3FN |  8   |
            |  FLOAT8_E5M2  |  15  |

      - 场景2，当scaleAlg为1时，只涉及FP8类型：
        - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
        - 找到该块中数值的最大绝对值:

          $$
          Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
          $$

        - 将FP32映射到目标数据类型FP8可表示的范围内，其中$Amax(DType)$是目标精度能表示的最大值

          $$
          S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
          $$

        - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
        - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
        - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：

          $$
          E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
          $$

        - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
        - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
        - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

      - 场景3，当scaleAlg为2时，只涉及FP4_E2M1类型：
        - 当dstTypeMax = 6.0/7.0时：
          - 将输入x在axis维度上按k = 32个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = 32：

            $$
            shared\_exp = \begin{cases} ceil(log_2(max_i(|V_i|))) - emax, & \text{如果} 尾数位的高比特前一/两位 \text{为1，且尾数不全为0} \\ floor(log_2(max_i(|V_i|))) - emax, & \text{其它} \end{cases} \\
            $$

            $$
            P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space 32\\
            $$

          - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
        - 当dstTypeMax != 6.0/7.0时：
          - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
          - 找到该块中数值的最大绝对值:

            $$
            Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
            $$

          - 将FP32映射到目标数据类型可表示的范围内，其中当dst_max_value=0时，$Amax(DType)$是目标精度能表示的最大值；当dst_max_value!=0时，$Amax(DType)$是dst_max_value传入值。

            $$
            S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
            $$

          - 将块缩放因子$S_{fp32}^b$转换为FP4_E2M1格式下可表示的缩放值$S_{ue8m0}^b$。
          - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$。
          - 为保证量化时不溢出，对指数进行向上取整，且在FP4_E2M1可表示的范围内：

            $$
            E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{否则} \end{cases}
            $$

          - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
          - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
          - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。
          - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRotateQuantGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRotateQuant”接口执行计算。

```c++
aclnnStatus aclnnRotateQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rotation,
  const aclTensor   *alpha,
  int64_t            axis,
  char              *roundMode,
  int64_t            scaleAlg,
  double             dstTypeMax,
  bool               trans,
  aclTensor         *yOut,
  aclTensor         *scaleOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnRotateQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnRotateQuantGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1491px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 200px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x(aclTensor*)</td>
      <td>输入</td>
      <td>待旋转量化的输入张量</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotation(aclTensor*)</td>
      <td>输入</td>
      <td>旋转矩阵</td>
      <td>不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2或3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>alpha(aclTensor*)</td>
      <td>输入</td>
      <td>clamp需要限制的范围的缩放系数</td>
      <td>可选参数。shape为(1,)，数据类型为BFLOAT16，有效取值范围(0.0, 1.0)，不在有效取值范围内不做clamp处理。</td>
      <td>BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>-</td>
    </tr>
    <tr>
      <td>axis(int64_t)</td>
      <td>输入</td>
      <td>表示量化发生的轴（reduce轴）</td>
      <td>目前只支持-1或最后一维的正索引。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode(char*)</td>
      <td>输入</td>
      <td>表示数据转换的模式</td>
      <td>支持"rint"、"round"、"floor"，传入空指针时，采用"rint"模式。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg(int64_t)</td>
      <td>输入</td>
      <td>表示scale的计算算法</td>
      <td>支持取值0、1、2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstTypeMax(double)</td>
      <td>输入</td>
      <td>表示量化目标类型的最大值</td>
      <td>支持取值0.0或者[6.0, 12.0]范围内（表示Amax为传入值）。</td>
      <td>DOUBLE</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>trans(bool)</td>
      <td>输入</td>
      <td>表示输出y是否转置</td>
      <td>目前只支持false。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut(aclTensor*)</td>
      <td>输出</td>
      <td>量化后的输出张量</td>
      <td>输出Tensor需预先分配。yOut的数据类型即为量化输出类型（y_dtype），torch单算子调用时通过参数传入，aclnn调用时无需单独传入y_dtype参数，通过yOut获取dtype即可。</td>
      <td>FLOAT4_E2M1、FLOAT8_E4M3FN、FLOAT8_E5M2、INT8、INT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut(aclTensor*)</td>
      <td>输出</td>
      <td>动态量化计算出的缩放系数</td>
      <td>输出Tensor需预先分配。</td>
      <td>FLOAT8_E8M0、FLOAT</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t)</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor*)</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程</td>
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

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>入参存在空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>入参的数据类型、维度、shape或参数取值不符合约束。</td>
    </tr>
  </tbody>
  </table>

## aclnnRotateQuant

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRotateQuantGetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- <term>Ascend 950PR/Ascend 950DT</term>：
  - x的shape为(*, N)，维度范围[1, 7]；rotation的shape为(K, K)或(N/K, K, K)，维度范围[2, 3], K当前版本仅支持取32，64，128。
  - x最后一维的长度(N)必须是K的整数倍。
  - yOut的输出类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2，shape与x相同。
  - x和rotation的数据类型必须相同，必须同时为BFLOAT16或FLOAT16。
  - rotation最后两维的长度必须一致。
  - scaleOut的shape必须是(*, CeilDiv(N,64), 2)，shape的维度支持2-8，数据类型为FLOAT8_E8M0。
  - yOut的数据类型必须在[FLOAT4_E2M1,FLOAT8_E4M3FN,FLOAT8_E5M2],yOut的shape必须和x的shape保持一致。
  - alpha为可选输入，不为空指针时，shape为(1,)，数据类型为BFLOAT16，有效取值范围(0.0, 1.0)。传入空指针或者不在有效取值范围内不做clamp处理。
  - axis目前只支持-1或者D-1，D为x的shape的维数。
  - roundMode支持"rint"、"round"、"floor"，传入空指针时，采用"rint"模式。当yOut的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2时，roundMode仅支持"rint"。
  - scaleAlg支持取值0、1、2，当yOut的数据类型为FLOAT8_E4M3FN或FLOAT8_E5M2时只支持0和1，当yOut的数据类型为FLOAT4_E2M1时只支持0和2。
  - dstTypeMax：当scaleAlg=2时dstTypeMax必须在[6.0, 12.0]范围内，其余场景仅支持0.0。
  - trans目前只支持false。

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当yOut的输出类型为int8时，y的shape必须和x相同(M, N)。
  - 当yOut的输出类型为int32时，y的shape必须为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(M)，数据类型为FLOAT。
  - alpha为可选输入，当前版本仅支持传入空指针。
  - axis目前只支持-1或者1。
  - roundMode支持"rint"、"round"、"floor"，传入空指针时，采用"rint"模式。
  - scaleAlg目前只支持0。
  - dstTypeMax目前只支持0.0。
  - trans目前只支持false。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。
  - scaleOut的shape仅支持1维。

- 确定性计算：
  - aclnnRotateQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant.h"

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

int Init(int32_t deviceId, aclrtStream* stream)
{
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

std::vector<uint16_t> GenerateRandomBf16Data(int64_t size, unsigned int seed = 42)
{
    std::vector<uint16_t> data(size);
    std::mt19937 gen(seed);
    for (int64_t i = 0; i < size; i++) {
        int sign = (gen() % 2) ? 0x8000 : 0;
        int exp = 0x3F00 + (gen() % 2);
        int mant = gen() % 128;
        data[i] = sign | exp | mant;
    }
    return data;
}

std::vector<uint16_t> GenerateIdentityMatrix(int64_t K)
{
    std::vector<uint16_t> matrix(K * K, 0);
    uint16_t bf16One = 0x3F80;
    for (int64_t i = 0; i < K; i++) {
        matrix[i * K + i] = bf16One;
    }
    return matrix;
}

int main()
{
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;

    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    // x: (M, N), rot: (K, K), y: (M, N), scale: (M,)
    // Constraints: rot must be square, N % K == 0, N % 8 == 0
    int64_t M = 1024;
    int64_t N = 256;
    int64_t K = 64;

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> rotShape = {K, K};
    std::vector<int64_t> yShape = {M, N};
    std::vector<int64_t> scaleShape = {M};

    // 创建x aclTensor (BF16)
    auto xHostData = GenerateRandomBf16Data(M * N, 42);
    void* xDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x tensor failed.\n"); return ret);

    // 创建rot aclTensor (BF16,必须为方阵)
    auto rotHostData = GenerateIdentityMatrix(K);
    void* rotDeviceAddr = nullptr;
    aclTensor* rotTensor = nullptr;
    ret = CreateAclTensor(rotHostData, rotShape, &rotDeviceAddr, aclDataType::ACL_BF16, &rotTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rotTensorPtr(rotTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rotAddrPtr(rotDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create rot tensor failed.\n"); return ret);

    // 创建y aclTensor (INT8)
    std::vector<int8_t> yHostData(M * N, 0);
    void* yDeviceAddr = nullptr;
    aclTensor* yTensor = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create y tensor failed.\n"); return ret);

    // 创建scale aclTensor (FLOAT32)
    std::vector<float> scaleHostData(M, 0.0f);
    void* scaleDeviceAddr = nullptr;
    aclTensor* scaleTensor = nullptr;
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Create scale tensor failed.\n"); return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnRotateQuant第一段接口
    int64_t axis = -1;
    char* roundMode = const_cast<char*>("rint");
    int64_t scaleAlg = 0;
    double dstTypeMax = 0.0;
    bool trans = false;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnRotateQuantGetWorkspaceSize(xTensor, rotTensor, nullptr, axis, roundMode, scaleAlg, dstTypeMax, trans,
                                            yTensor, scaleTensor, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnRotateQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnRotateQuant第二段接口
    ret = aclnnRotateQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuant failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto ySize = GetShapeSize(yShape);
    std::vector<int8_t> yResult(ySize, 0);
    ret = aclrtMemcpy(yResult.data(), ySize * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed.\n"); return ret);

    auto scaleSize = GetShapeSize(scaleShape);
    std::vector<float> scaleResult(scaleSize, 0.0f);
    ret = aclrtMemcpy(scaleResult.data(), scaleSize * sizeof(float), scaleDeviceAddr, scaleSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result failed.\n"); return ret);

    // 打印部分结果
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(yResult.size()); i++) {
        LOG_PRINT("y[%ld] = %d\n", i, static_cast<int>(yResult[i]));
    }
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(scaleResult.size()); i++) {
        LOG_PRINT("scale[%ld] = %f\n", i, scaleResult[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(xTensor);
    aclDestroyTensor(rotTensor);
    aclDestroyTensor(yTensor);
    aclDestroyTensor(scaleTensor);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(rotDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
