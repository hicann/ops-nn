# aclnnRmsNormDynamicMxQuant

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

- 接口功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicMxQuant算子则是在尾轴上按blocksize分组进行动态MX量化的算子。RmsNormDynamicMxQuant算子将RmsNorm归一化输出给到的DynamicMxQuant算子融合起来，减少搬入搬出操作。在输入尾轴axis上，根据每blocksize=32个数，计算出这组数对应的量化尺度mxscale，然后对这组数每一个除以mxscale，根据roundMode转换到对应的dstType，得到量化结果y。在dstType为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scaleAlg的取值来指定计算mxscale的不同算法。
- 计算公式：

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$
  场景1，当scaleAlg为0时：

   - 将RmsNorm输出y在尾轴维度上按k = 32个数分组，一组k个数 $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale,\{P_i\}_{i=1}^{k}\}$
    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax
    $$

    $$
    mxscale = 2^{shared\_exp}
    $$

    $$
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

  场景2，当scaleAlg为1时，只涉及FP8类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。
    - 找到该块中数值的最大绝对值：

      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$

    - 将FP32映射到目标数据类型FP8可表示的范围内：

      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$

    - 转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整：

      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$

    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnRmsNormDynamicMxQuantGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnRmsNormDynamicMxQuant`接口执行计算。

```Cpp
aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *gamma,
  const aclTensor   *beta,
  double             epsilon,
  int64_t            scaleAlg,
  char              *roundMode,
  int64_t            dstType,
  bool               outputRstd,
  aclTensor         *yOut,
  aclTensor         *mxscaleOut,
  aclTensor         *rstdOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnRmsNormDynamicMxQuant(
    void            *workspace,
    uint64_t         workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnRmsNormDynamicMxQuantGetWorkspaceSize

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
      <td>x（const aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>具体约束详见约束说明。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（const aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的gamma。</td>
      <td><ul><li>支持空Tensor。</li><li>shape只能为一维，需要与x最后一维维度匹配。</li><li>数据类型默认与输入x一致；若不一致，则显式设为FLOAT32。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（const aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的偏置项。对应公式中的beta。</td>
      <td><ul><li>支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>如果提供，shape和数据类型需要与gamma保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>epsilon（double）</td>
      <td>输入</td>
      <td>表示添加到分母中的值，以确保数值稳定。对应公式中的epsilon。</td>
      <td><ul><li>建议传入较小正数，如1e-6。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg（int64_t）</td>
      <td>输入</td>
      <td>表示mxscaleOut的计算方法，对应公式中的scaleAlg。</td>
      <td><ul><li>支持取值0和1，取值为0代表场景1，取值为1代表场景2。</li><li>当dstType为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode（char*）</td>
      <td>输入</td>
      <td>表示数据转换的模式，对应公式中的round_mode。</td>
      <td><ul><li>不支持传入空指针，必须显式传入支持的字符串。</li><li>当dstType为40/41时，支持{"rint", "floor", "round"}。</li><li>当dstType为36/35时，仅支持{"rint"}。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>表示指定数据转换后yOut的类型，对应公式中的DType。</td>
      <td><ul><li>输入范围为{35, 36, 40, 41}，分别对应{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>outputRstd（bool）</td>
      <td>输入</td>
      <td>表示指定是否输出有效的rstdOut。</td>
      <td><ul><li>支持True和False。</li><li>当outputRstd为False时，rstdOut为无效输出。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化并量化后的结果，对应公式中的Pi和di。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与输入x一致。</li><li>数据类型需要与dstType保持一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示每个分组对应的量化尺度，对应公式中的mxscale和Sb。</td>
      <td><ul><li>支持空Tensor。</li><li>沿x的尾轴每32个元素对应一个量化尺度，尺度数量向上取整后补齐为偶数，补齐位置填0。</li><li>shape为x.shape[:-1] + (ceil(ceil(x.shape[-1] / 32) / 2), 2)。</li><li>具体计算过程见约束说明。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化分母的倒数。对应公式中Rms(x)的倒数。</td>
      <td><ul><li>支持空Tensor。</li><li>当outputRstd为True时，维度数与入参x保持一致，除最后一维外各维度与x对应维度保持一致，最后一维为1。rstdOut的shape与x的shape关系举例：若x的shape为(2,3,4,8)，则rstdOut的shape为(2,3,4,1)。</li><li>当outputRstd为False时，该参数的最终输出无效。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会检查必需指针。对于尾轴外存在空维的x，指针检查通过后直接返回成功，不触发后续shape、dtype及属性校验。未提前返回的场景继续执行后续检查，出现以下情况时报错：

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
      <td rowspan="2">ACLNN_ERR_PARAM_NULLPTR</td>
      <td rowspan="2">161001</td>
      <td>传入的x、gamma、roundMode、yOut或mxscaleOut是空指针。</td>
    </tr>
    <tr>
      <td>当outputRstd为True时，传入rstdOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_INNER_NULLPTR</td>
      <td rowspan="8">561103</td>
      <td>输入或输出参数的维度数和数据类型不在范围之内，dstType和yOut的数据类型不匹配。</td>
    </tr>
    <tr>
      <td>输入x和输出yOut的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>gamma、beta(若存在)的shape不是完全相同的shape，或二者的维度数不为1，或轴长不等于x的尾轴大小，或者类型不相同。</td>
    </tr>
    <tr>
      <td>gamma的维度和x的需要做norm的维度不相同，或rstdOut的维度和x的不需要norm的维度不相同，或rstdOut对应x归一化维度的轴长不为1。</td>
    </tr>
    <tr>
      <td>scaleAlg不是0或1，或roundMode不属于{"rint", "floor", "round"}。</td>
    </tr>
    <tr>
      <td>dstType为FP8时，roundMode不是"rint"。</td>
    </tr>
    <tr>
      <td>dstType为fp4时，scaleAlg不是0，或者输入x的尾轴不能被2整除。</td>
    </tr>
    <tr>
      <td>mxscaleOut的维度数不等于输入x的维度数+1，或轴长不符合约束说明。</td>
    </tr>
  </tbody></table>

## aclnnRmsNormDynamicMxQuant

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1042px"><colgroup>
  <col style="width: 141px">
  <col style="width: 110px">
  <col style="width: 791px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRmsNormDynamicMxQuantGetWorkspaceSize获取。</td>
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

- x的rank为1–7；gamma和可选beta为一维，长度等于x的尾轴长度。
- x尾轴为空时，yOut和mxscaleOut为空，启用的rstdOut为NaN；尾轴以外存在空维时无有效输出数据。

- **边界值场景说明**
  - 当输入是Inf时：1、输出yOut为0；2、输出mxscaleOut为255，偶数pad填充值为0；3、输出rstdOut为0。
  - 当输入是NaN时：1、输出yOut为0；2、输出mxscaleOut为255，偶数pad填充值为0；3、输出rstdOut为NaN。

- **数据格式说明**

  所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

- **各产品型号支持数据类型说明**

  <!-- npu="950" id7 -->
  - <term>Ascend 950PR&950DT系列产品</term>：

    | `x`数据类型 | `gamma`数据类型 | `beta`数据类型 | `yOut`数据类型 | `mxscaleOut`数据类型 | `rstdOut`数据类型 |
    | ----------- | -------------- | ------------- | -------------- | ------------------- | ---------------- |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | FLOAT8_E8M0         | FLOAT32          |
  <!-- end id7 -->

- mxscaleOut的shape约束说明如下：
  - rank(mxscaleOut) = rank(x) + 1。
  - mxscaleOut.shape[-2] = ceil(ceil(x.shape[-1] / 32) / 2)。
  - mxscaleOut.shape[-1] = 2。
  - 其他维度与输入x一致。

- x的shape约束说明如下：
  - 当输出yOut的数据类型为FLOAT4_E2M1或FLOAT4_E1M2，x尾轴的值必须为偶数。

- 确定性计算：
  - aclnnRmsNormDynamicMxQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

使用自行构建的算子包时，将下方代码保存为`test_aclnn_rms_norm_dynamic_mx_quant.cpp`。配置CANN环境后，将`VENDOR_OPP_PATH`设为自定义算子包的安装目录，加载该包的环境配置，编译并运行：

```bash
export VENDOR_OPP_PATH="${ASCEND_HOME_PATH}/vendors/custom_nn"
source "${VENDOR_OPP_PATH}/bin/set_env.bash"
g++ -std=c++11 test_aclnn_rms_norm_dynamic_mx_quant.cpp \
    -I"${VENDOR_OPP_PATH}/op_api/include" -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_HOME_PATH}/include/aclnnop" \
    -L"${VENDOR_OPP_PATH}/op_api/lib" -L"${ASCEND_HOME_PATH}/lib64" \
    -lcust_opapi -lascendcl -lnnopbase -o test_aclnn_rms_norm_dynamic_mx_quant
./test_aclnn_rms_norm_dynamic_mx_quant
```

```cpp
#include <iostream>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cmath>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_dynamic_mx_quant.h"

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
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  // 固定写法，资源初始化
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); aclFinalize(); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret);
      aclrtResetDevice(deviceId); aclFinalize(); return ret);
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
  CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed\n"); return 1);
  return 0;
}

bool CheckOutputs(const std::vector<uint8_t>& y, const std::vector<uint8_t>& mxscale,
                  const std::vector<float>& rstd, uint8_t expectedY, double expectedRstd)
{
    // For these constant FP16 inputs, the affine result rounds to 1 or 1.5.
    // OCP uses exponent floor(log2(max(abs(affine)))) - 8 = -8.
    // E8M0 encodes -8 as 119; E4M3FN encodes 256/384 as 0x78/0x7c.
    constexpr uint8_t expectedScale = 119;
    constexpr double rstdRelativeTolerance = 1.0 / 8192.0;
    for (size_t i = 0; i < y.size(); ++i) {
        if (y[i] != expectedY) {
            LOG_PRINT("y[%zu]: got 0x%02x, expected 0x%02x\n", i, y[i], expectedY);
            return false;
        }
    }
    for (size_t i = 0; i < mxscale.size(); ++i) {
        if (mxscale[i] != expectedScale) {
            LOG_PRINT("mxscale[%zu]: got %u, expected %u\n", i, mxscale[i], expectedScale);
            return false;
        }
    }
    for (size_t i = 0; i < rstd.size(); ++i) {
        if (!std::isfinite(rstd[i]) ||
            std::abs(rstd[i] - expectedRstd) > rstdRelativeTolerance * std::abs(expectedRstd)) {
            LOG_PRINT("rstd[%zu]: got %.9g, expected %.9g\n", i, rstd[i], expectedRstd);
            return false;
        }
    }
    LOG_PRINT("Checked y=%zu, mxscale=%zu, rstd=%zu elements\n", y.size(), mxscale.size(), rstd.size());
    return true;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> gammaShape = {64};
  std::vector<int64_t> mxscaleShape = {2, 1, 2};  // CeilDiv(CeilDiv(64, 32), 2) = 1
  std::vector<int64_t> rstdShape = {2, 1};

  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;

  void* yDeviceAddr = nullptr;
  void* mxscaleDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* y = nullptr;
  aclTensor* mxscale = nullptr;
  aclTensor* rstd = nullptr;

  void* workspaceAddr = nullptr;

  int cleanupStatus = 0;
  auto checkCleanup = [&](const char* name, int status) {
      if (status != ACL_SUCCESS) {
          LOG_PRINT("%s failed. ERROR: %d\n", name, status);
          if (cleanupStatus == 0) {
              cleanupStatus = status;
          }
      }
  };
  auto destroyTensor = [&](aclTensor*& tensor) {
      if (tensor != nullptr) {
          checkCleanup("aclDestroyTensor", aclDestroyTensor(tensor));
          tensor = nullptr;
      }
  };
  auto freeDeviceAddr = [&](void*& deviceAddr) {
      if (deviceAddr != nullptr) {
          checkCleanup("aclrtFree", aclrtFree(deviceAddr));
          deviceAddr = nullptr;
      }
  };
  auto releaseResources = [&]() {
      // Finish any submitted work before freeing tensors or device memory.
      checkCleanup("aclrtSynchronizeStream", aclrtSynchronizeStream(stream));
      freeDeviceAddr(workspaceAddr);
      destroyTensor(x);
      destroyTensor(gamma);
      destroyTensor(y);
      destroyTensor(mxscale);
      destroyTensor(rstd);
      freeDeviceAddr(xDeviceAddr);
      freeDeviceAddr(gammaDeviceAddr);
      freeDeviceAddr(yDeviceAddr);
      freeDeviceAddr(mxscaleDeviceAddr);
      freeDeviceAddr(rstdDeviceAddr);
      checkCleanup("aclrtDestroyStream", aclrtDestroyStream(stream));
      checkCleanup("aclrtResetDevice", aclrtResetDevice(deviceId));
      checkCleanup("aclFinalize", aclFinalize());
      return cleanupStatus;
  };

  int64_t xShapeSize = GetShapeSize(xShape);
  int64_t gammaShapeSize = GetShapeSize(gammaShape);
  int64_t mxscaleShapeSize = GetShapeSize(mxscaleShape);
  int64_t rstdShapeSize = GetShapeSize(rstdShape);

  std::vector<uint16_t> xHostData(xShapeSize, 0x3800);
  std::vector<uint16_t> gammaHostData(gammaShapeSize, 0x3e00);

  std::vector<uint8_t> yHostData(xShapeSize, 0);
  std::vector<uint8_t> mxscaleHostData(mxscaleShapeSize, 0);
  std::vector<float> rstdHostData(rstdShapeSize, 0);

  double epsilon = 1e-6;
  char roundMode[] = "rint";
  bool outputRstd = true;  // 设置为true，输出有效的rstdOut

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, releaseResources(); return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, releaseResources(); return ret);

  // 创建y aclTensor (FLOAT8_E4M3FN)
  ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
  CHECK_RET(ret == ACL_SUCCESS, releaseResources(); return ret);
  // 创建mxscale aclTensor (FLOAT8_E8M0)
  ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
  CHECK_RET(ret == ACL_SUCCESS, releaseResources(); return ret);
  // 创建rstd aclTensor
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, releaseResources(); return ret);

  // 3. 调用CANN算子库API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnRmsNormDynamicMxQuant第一段接口
  ret = aclnnRmsNormDynamicMxQuantGetWorkspaceSize(
      x, gamma, nullptr, epsilon, 0, roundMode, 36, outputRstd, y, mxscale, rstd, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRmsNormDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); releaseResources(); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); releaseResources(); return ret);
  }
  // 调用aclnnRmsNormDynamicMxQuant第二段接口
  ret = aclnnRmsNormDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS,
      LOG_PRINT("aclnnRmsNormDynamicMxQuant failed. ERROR: %d\n", ret); releaseResources(); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); releaseResources(); return ret);

  // 5. 获取并检查输出
  std::vector<uint8_t> yRet(xShapeSize, 0);
  std::vector<uint8_t> mxscaleRet(mxscaleShapeSize, 0);
  std::vector<float> rstdRet(rstdShapeSize, 0);
  ret = aclrtMemcpy(yRet.data(), yRet.size(), yDeviceAddr, yRet.size(), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y failed. ERROR: %d\n", ret); releaseResources(); return ret);
  ret = aclrtMemcpy(mxscaleRet.data(), mxscaleRet.size(), mxscaleDeviceAddr, mxscaleRet.size(),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale failed. ERROR: %d\n", ret); releaseResources(); return ret);
  ret = aclrtMemcpy(rstdRet.data(), rstdRet.size() * sizeof(float), rstdDeviceAddr,
                    rstdRet.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy rstd failed. ERROR: %d\n", ret); releaseResources(); return ret);
  // x=0.5, gamma=1.5: FP16 affine=1.5, E4M3FN y=384 (0x7c).
  if (!CheckOutputs(yRet, mxscaleRet, rstdRet, 0x7c, 1.0 / std::sqrt(0.25 + epsilon))) {
    releaseResources();
    return 1;
  }
  ret = releaseResources();
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  LOG_PRINT("RmsNormDynamicMxQuant example PASSED\n");
  return 0;
}
```
