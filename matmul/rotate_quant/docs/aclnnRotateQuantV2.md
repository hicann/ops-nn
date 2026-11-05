# aclnnRotateQuantV2

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

- **与V1版本的差异**：
  - V2版本新增了`group_list`、`group_list_type`参数，用于指定量化分组情况。

- **用户选择建议**：
  - 若需要指定量化的分组情况，请选择V2版本。

- 接口功能：对张量x进行旋转变换，然后执行可选的clamp操作，最后执行MX量化（目的数据类型为FLOAT4_E2M1、FLOAT8_E4M3FN或FLOAT8_E5M2）。
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

    <!-- npu="A3,910b" id7 -->
    - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：对称动态量化（pertoken逐行量化）
      - 缩放因子计算（逐行计算）

        $$
        s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
        $$

        其中：$s_i$ 是第 $i$ 行的缩放因子；$C_{\text{MAX}}$ 是量化范围最大值，int8取127，quint4x2取7。
      - 量化计算

        $$
        y_{i,j} = \frac{Y_{i,j}}{s_i}
        $$

    <!-- end id7 -->
    <!-- npu="950" id8 -->
    - <term>Ascend 950PR&950DT系列产品</term>：MX量化

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

    <!-- end id8 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)：必须先调用 `aclnnRotateQuantV2GetWorkspaceSize` 接口获取入参并根据流程计算所需workspace大小，再调用 `aclnnRotateQuantV2` 接口执行计算。

```c++
aclnnStatus aclnnRotateQuantV2GetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rotation,
  const aclTensor   *alpha,
  const aclTensor   *group_list,
  int64_t            group_list_type,
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
aclnnStatus aclnnRotateQuantV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnRotateQuantV2GetWorkspaceSize

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
      <td>支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotation(aclTensor*)</td>
      <td>输入</td>
      <td>旋转矩阵</td>
      <td>支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2或3或4</td>
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
      <td>group_list（aclTensor*）</td>
      <td>可选输入</td>
      <td>代表输入`x`的量化分组大小分布。</td>
      <td>不支持空Tensor。</td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>group_list_type(int64_t)</td>
      <td>输入</td>
      <td>代表group_list输入的分组方式。</td>
      <td>当group_list不为nullptr时支持取值0，1。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
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
      <td>workspaceSize(uint64_t*)</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor**)</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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

## aclnnRotateQuantV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRotateQuantV2GetWorkspaceSize获取。</td>
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

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

<!-- npu="950" id9 -->
- <term>Ascend 950PR&950DT系列产品</term>：
  - group_list支持传入nullptr。
  - group_list不为nullptr时：
    - group_listType必须从[0, 1]中取值。
    - group_list的shape为(E,)，E表示分组数，E需要大于等于1且小于等于1024。
    - group_list的数值需要满足以下条件，否则无法保证输出是否符合预期：
      - 当group_list_type为0时，group_list必须为非负单调非递减数列，表示分组后每组大小的cumsum结果（累计和），最后一个值应小于等于x中tensor的第一维。
      - 当group_list_type为1时，group_list必须为非负数列，表示分组后每组大小，数值的总和应小于等于x中tensor的第一维。
    - 不支持rotation为空Tensor。
  - group_list传入nullptr时，x的shape为(*, N)，维度范围[1, 7]；rotation的shape为(K, K)或(N/K, K, K)，维度范围[2, 3], K当前版本仅支持取32，64，128，256。
  - group_list不为nullptr时，x的shape为(M, N)，维度仅限二维；rotation的shape为(E, K, K)或(E, N/K, K, K)，维度范围[3, 4], K当前版本仅支持取32，64，128，256。
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
<!-- end id9 -->
<!-- npu="A3,910b" id10 -->
- <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>：
  - group_list仅支持传入nullptr。
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当yOut的输出类型为int8时，y的shape必须和x相同(M, N)。
  - 当yOut的输出类型为int32时，y的shape必须为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(M)，数据类型为FLOAT。
  - alpha为可选输入，当前版本仅支持传入空指针。
  - axis目前只支持-1或者1。
  <!-- npu="A3,910b" id11 -->
  - <term>Atlas A3系列产品</term>、<term>Atlas A2系列产品</term>的roundMode当前仅支持rint。
  <!-- end id11 -->
  - scaleAlg目前只支持0。
  - dstTypeMax目前只支持0.0。
  - trans目前只支持false。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。
  - scaleOut的shape仅支持1维。

<!-- end id10 -->
- 确定性计算：
  - aclnnRotateQuantV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

本示例使用V2接口执行普通RotateQuant：group_list传入nullptr，group_list_type设为0。根据运行设备选择受支持的输出数据类型和scale布局，不启用分组量化。

```cpp
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_rotate_quant_v2.h"

static void Check(int code, const char* api)
{
    if (code != ACL_SUCCESS) {
        throw std::runtime_error(std::string(api) + " failed, code=" + std::to_string(code));
    }
}

struct Resources {
    int32_t deviceId = 0; // 根据实际设备选择 deviceId。
    bool aclReady = false;
    bool deviceReady = false;
    aclrtStream stream = nullptr;
    void* workspace = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        // 异常退出时也先等待已提交任务结束，再释放其引用的内存。
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (auto tensor : tensors) {
            aclDestroyTensor(tensor);
        }
        for (auto buffer : buffers) {
            aclrtFree(buffer);
        }
        if (workspace != nullptr) {
            aclrtFree(workspace);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceReady) {
            aclrtResetDevice(deviceId);
        }
        if (aclReady) {
            aclFinalize();
        }
    }
};

static aclTensor* CreateDeviceTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype,
                                     const void* hostData, size_t bytes, void** deviceData)
{
    Check(aclrtMalloc(deviceData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    resources.buffers.push_back(*deviceData);
    Check(aclrtMemcpy(*deviceData, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy H2D");

    // stride 按逻辑元素计算；Device 存储字节数由调用方按 dtype 显式提供。
    std::vector<int64_t> strides(shape.size(), 1);
    for (size_t i = shape.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * shape[i - 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), *deviceData);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor failed");
    }
    resources.tensors.push_back(tensor);
    return tensor;
}

int main()
{
    Resources resources;
    try {
        resources.tensors.reserve(4);
        resources.buffers.reserve(4);
        Check(aclInit(nullptr), "aclInit");
        resources.aclReady = true;
        Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice");
        resources.deviceReady = true;
        Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream");

        const char* socName = aclrtGetSocName();
        if (socName == nullptr) {
            throw std::runtime_error("aclrtGetSocName failed");
        }
        std::string soc(socName);
        std::transform(soc.begin(), soc.end(), soc.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        const bool isInt = soc.rfind("ascend910b", 0) == 0 || soc.rfind("ascend910_93", 0) == 0;
        const bool isMx = soc.rfind("ascend950", 0) == 0 || soc.rfind("ascend350", 0) == 0 ||
                          soc.rfind("ascend960dt", 0) == 0 || soc.rfind("ascend960pr", 0) == 0;
        if (!isInt && !isMx) {
            throw std::runtime_error("Unsupported SOC for this example: " + std::string(socName));
        }
        std::printf("Current SOC: %s; mode=ordinary, y=%s, scale=%s\n", socName,
                    isMx ? "FLOAT8_E4M3FN" : "INT8", isMx ? "FLOAT8_E8M0" : "FLOAT");

        const int64_t M = 1024;
        const int64_t N = 256;
        const int64_t K = 64;
        const int64_t scaleSlots = 2 * ((N + 63) / 64);
        const int64_t group_list_type = 0;

        std::vector<int64_t> xShape = {M, N};
        std::vector<int64_t> rotShape = {K, K};
        std::vector<int64_t> yShape = {M, N};
        std::vector<int64_t> scaleShape = isMx ? std::vector<int64_t>{M, (N + 63) / 64, 2} :
                                               std::vector<int64_t>{M};

        // uint16_t 存储 FP16 原始位：0x3800=0.5，0x3c00=1.0。使用二维单位旋转矩阵。
        std::vector<uint16_t> xHost(M * N, 0x3800);
        std::vector<uint16_t> rotHost(K * K, 0);
        for (int64_t j = 0; j < K; ++j) {
            rotHost[j * K + j] = 0x3c00;
        }

        // INT8 和 FP8 每元素均为一字节；FP8 分支仅打印原始编码。
        std::vector<int8_t> yHost(M * N, 0);
        std::vector<float> scaleFloatHost(isMx ? 0 : M, 0.0f);
        std::vector<uint8_t> scaleMxHost(isMx ? M * scaleSlots : 0, 0);
        void* scaleHostData = isMx ? static_cast<void*>(scaleMxHost.data()) :
                                    static_cast<void*>(scaleFloatHost.data());
        const size_t yBytes = yHost.size() * sizeof(int8_t);
        const size_t scaleBytes = isMx ? scaleMxHost.size() * sizeof(uint8_t) :
                                        scaleFloatHost.size() * sizeof(float);
        void* xDevice = nullptr;
        void* rotDevice = nullptr;
        void* yDevice = nullptr;
        void* scaleDevice = nullptr;

        aclTensor* x = CreateDeviceTensor(resources, xShape, ACL_FLOAT16, xHost.data(), xHost.size() * sizeof(uint16_t),
                                          &xDevice);
        aclTensor* rotation = CreateDeviceTensor(resources, rotShape, ACL_FLOAT16, rotHost.data(),
                                                 rotHost.size() * sizeof(uint16_t), &rotDevice);
        aclTensor* yOut = CreateDeviceTensor(resources, yShape, isMx ? ACL_FLOAT8_E4M3FN : ACL_INT8,
                                             yHost.data(), yBytes, &yDevice);
        aclTensor* scaleOut = CreateDeviceTensor(resources, scaleShape, isMx ? ACL_FLOAT8_E8M0 : ACL_FLOAT,
                                                 scaleHostData, scaleBytes, &scaleDevice);

        int64_t axis = -1;
        char roundMode[] = "rint";
        int64_t scaleAlg = 0;
        double dstTypeMax = 0.0;
        bool trans = false;
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;

        // V2 兼容普通模式：所有平台均传入空分组列表，group_list_type 显式采用默认语义值 0。
        Check(
            aclnnRotateQuantV2GetWorkspaceSize(x, rotation, nullptr, nullptr, group_list_type, axis, roundMode,
                                               scaleAlg, dstTypeMax, trans, yOut, scaleOut, &workspaceSize, &executor),
            "aclnnRotateQuantV2GetWorkspaceSize");
        if (workspaceSize != 0) {
            Check(aclrtMalloc(&resources.workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc workspace");
        }
        Check(aclnnRotateQuantV2(resources.workspace, workspaceSize, executor, resources.stream), "aclnnRotateQuantV2");
        Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream");

        Check(aclrtMemcpy(yHost.data(), yBytes, yDevice, yBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy Y D2H");
        Check(aclrtMemcpy(scaleHostData, scaleBytes, scaleDevice, scaleBytes, ACL_MEMCPY_DEVICE_TO_HOST),
              "aclrtMemcpy scale D2H");

        if (isMx) {
            // FP8/E8M0 的原始编码不能直接解释为量化结果的整数数值。
            std::printf("Y first code=0x%02x, scale first code=0x%02x\n",
                        static_cast<unsigned>(static_cast<uint8_t>(yHost[0])),
                        static_cast<unsigned>(scaleMxHost[0]));
        } else {
            std::printf("Y first value=%d, scale first value=%f\n", static_cast<int>(yHost[0]),
                        static_cast<double>(scaleFloatHost[0]));
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
```
