# aclnnAddRmsNormDynamicMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/add_rms_norm_dynamic_mx_quant)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：RmsNorm算子是大模型常用的归一化操作，相比LayerNorm算子，其去掉了减去均值的部分。DynamicMxQuant算子则是在尾轴上按blocksize分组进行动态MX量化的算子。AddRmsNormDynamicMxQuant算子将RmsNorm前的Add算子和RmsNorm归一化输出给到的DynamicMxQuant算子融合起来，减少搬入搬出操作。 在输入尾轴axis上，根据每blocksize=32个数，计算出这组数对应的量化尺度mxscale，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dst_type，得到量化结果y。在dst_type为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scale_alg的取值来指定计算mxscale的不同算法。
- 计算公式：

  $$
  x=x_{1}+x_{2}
  $$

  $$
  y = \operatorname{RmsNorm}(x)=\frac{x}{\operatorname{Rms}(\mathbf{x})}\cdot gamma+beta, \quad \text { where } \operatorname{Rms}(\mathbf{x})=\sqrt{\frac{1}{n} \sum_{i=1}^n x_i^2+epsilon}
  $$

  当scaleAlg为0时：
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

  当scaleAlg为1时，只涉及FP8类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。
    - 找到该块中数值的最大绝对值:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - 将FP32映射到目标数据类型FP8可表示的范围内:
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{Amax(DType)}
      $$
    - 转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整:
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为非正规数，且} M_{fixp}^b > 0.5 \\ E_{int}^b, & \text{否则} \end{cases}
      $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤：$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize`接口获取入参并根据计算流程所需workspace大小，再调用`aclnnAddRmsNormDynamicMxQuant`接口执行计算。

```Cpp
aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *gamma,
  const aclTensor *beta,
  double           epsilon,
  int64_t          scaleAlg,
  char            *roundModeOptional,
  int64_t          dstType,
  bool            outputRstd,
  aclTensor       *yOut,
  aclTensor       *xOut,
  aclTensor       *mxscaleOut,
  aclTensor       *rstdOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddRmsNormDynamicMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize

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
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的x1。</td>
      <td><ul><li>不支持空Tensor。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的源数据张量。对应公式中的x2。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和数据类型需要与x1保持一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的权重张量。对应公式中的gamma。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape只能为一维，需要与x1最后一维维度匹配。</li></ul></td>
      <td>FLOAT16、BFLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>beta（aclTensor*）</td>
      <td>输入</td>
      <td>表示标准化过程中的偏置项。对应公式中的beta。</td>
      <td><ul><li>不支持空Tensor。</li><li>可选参数，支持传入空指针。</li><li>如果提供，shape和数据类型需要与gamma保持一致。</li></ul></td>
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
      <td><ul><li>支持取值0和1，取值为0表示Open Compute Project(OCP)实现，取值为1表示cuBLAS实现。</li><li>当dstType为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundMode（char*）</td>
      <td>输入</td>
      <td>表示数据转换的模式，对应公式中的round_mode。</td>
      <td><ul><li>当dstType为40/41时，支持{"rint", "floor", "round"}。</li><li>当dstType为36/35时，仅支持{"rint"}。</li></ul></td>
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
      <td><ul><li>支持True和False。</li><li>当outputRstd为False时，rstdOut最终输出为空Tensor，此时rstdOut为无效占位输出。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化并量化后的结果，对应公式中的Pi和di。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape需要与输入x1一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>xOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示x1和x2的和，对应公式中的x。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape和数据类型需要与输入x1一致。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示每个分组对应的量化尺度，对应公式中的mxscale和Sb。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape在尾轴上为x对应值除以blocksize=32向上取整，并对其进行偶数pad，pad填充值为0，具体计算过程见约束说明。其余维度与入参x1的shape前几维保持一致，前几维指x1的维度减去gamma的维度，表示不需要norm的维度。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstdOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示归一化后的标准差的倒数。对应公式中Rms(x)的倒数。</td>
      <td><ul><li>支持空Tensor。</li><li>当outputRstd为True时，shape与入参x1的shape前几维保持一致，前几维指x1的维度减去gamma的维度，表示不需要norm的维度。</li><li>当outputRstd为False时，rstdOut输出空Tensor。</li></ul></td>
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
      <td>传入的x1、x2、gamma、yOut、xOut、mxscaleOut和rstdOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>输入或输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_INNER_TILING_ERROR</td>
      <td rowspan="9">561002</td>
    </tr>
    <tr>
      <td>scaleAlg不是0或1，roundModeOptional(非空时)不是 {rint, floor, round}。</td>
    </tr>
    <tr>
      <td>dstType为 fp8 时，roundModeOptional不是 rint。</td>
    </tr>
    <tr>
      <td>输入x1、输出yOut的shape不是完全相同的shape。</td>
    </tr>
    <tr>
      <td>mxscaleOut的维度数不等于输入x1的维度数+1。</td>
    </tr>
    <tr>
      <td>gamma、beta(若存在)的shape不是完全相同的shape，或者类型不相同。</td>
    </tr>
    <tr>
      <td>gamma的维度和x1的需要作norm的维度不相同，或rstdOut的维度和x1的不需要norm的维度不相同，或x1的需要norm的维度数不为1。</td>
    </tr>
  </tbody></table>

## aclnnAddRmsNormDynamicMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize获取。</td>
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

- **边界值场景说明**
  - 当输入是Inf时：1、输出yOut为0；2、输出xOut为Inf；3、输出mxscaleOut为255，偶数pad填充值为0；4、输出rstdOut为0。
  - 当输入是NaN时：1、输出yOut为0；2、输出xOut为Nan；3、输出mxscaleOut为255，偶数pad填充值为0；4、输出rstdOut为NaN。

- **数据格式说明**

  所有输入输出Tensor的数据格式推荐使用ND格式，其他数据格式会由框架默认转换成ND格式进行处理。

- **各产品型号支持数据类型说明**

  - <term>Ascend 950PR/Ascend 950DT</term>：
    
    | `x1`数据类型 | `x2`数据类型 | `gamma`数据类型 | `beta`数据类型 | `yOut`数据类型 | `xOut`数据类型 | `mxscaleOut`数据类型 | `rstdOut`数据类型 |
    | ----------- | ----------- | -------------- | ------------- | -------------- | ------------- | ------------------- | ---------------- |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E2M1    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT4_E1M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E4M3FN  | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT16        | FLOAT16       | FLOAT8_E5M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E2M1    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT4_E1M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E4M3FN  | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | BFLOAT16       | BFLOAT16      | FLOAT8_E5M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | FLOAT16     | FLOAT16     | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | FLOAT16       | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E2M1    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT4_E1M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E4M3FN  | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |
    | BFLOAT16    | BFLOAT16    | FLOAT32        | FLOAT32       | FLOAT8_E5M2    | BFLOAT16      | FLOAT8_E8M0         | FLOAT32          |

- mxscaleOut的shape约束说明如下：
  - rank(mxscaleOut) = rank(x1) + 1。
  - mxscaleOut.shape[-2] = (ceil(x1.shape[-1] / 32) + 2 - 1) / 2。
  - mxscaleOut.shape[-1] = 2。
  - 其他维度与输入x1一致。

- x1的shape约束说明如下：
  - 当输出yOut的数据类型为FLOAT4_E2M1或FLOAT4_E1M2，x1尾轴的值必须为偶数。

- 确定性计算：
  - aclnnAddRmsNormDynamicMxQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_rms_norm_dynamic_mx_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
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
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {2, 64};
    std::vector<int64_t> gammaShape = {64};
    std::vector<int64_t> mxscaleShape = {2, 1, 2};
    std::vector<int64_t> rstdShape = {2, 1};

    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* betaDeviceAddr = nullptr;

    void* yDeviceAddr = nullptr;
    void* mxscaleDeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* xDeviceAddr = nullptr;

    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* beta = nullptr;
    aclTensor* y = nullptr;
    aclTensor* mxscale = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* x = nullptr;

    int64_t xShapeSize = GetShapeSize(xShape);
    int64_t gammaShapeSize = GetShapeSize(gammaShape);
    int64_t mxscaleShapeSize = GetShapeSize(mxscaleShape);
    int64_t rstdShapeSize = GetShapeSize(rstdShape);

    std::vector<short> x1HostData(xShapeSize, 0x3800);
    std::vector<short> x2HostData(xShapeSize, 0x3800);
    std::vector<short> gammaHostData(gammaShapeSize, 0x3e00);
    std::vector<short> betaHostData(gammaShapeSize, 0);

    std::vector<int8_t> yHostData(xShapeSize, 0);
    std::vector<int8_t> mxscaleHostData(mxscaleShapeSize, 0);
    std::vector<float> rstdHostData(rstdShapeSize, 0);
    std::vector<short> xHostData(xShapeSize, 0);

    float epsilon = 1e-6;

    // 创建x1 aclTensor
    ret = CreateAclTensor(x1HostData, xShape, &x1DeviceAddr, aclDataType::ACL_FLOAT16, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x2 aclTensor
    ret = CreateAclTensor(x2HostData, xShape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gamma aclTensor
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT16, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建y aclTensor (FLOAT8_E4M3FN)
    ret = CreateAclTensor(yHostData, xShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建mxscale aclTensor (FLOAT8_E8M0)
    ret = CreateAclTensor(mxscaleHostData, mxscaleShape, &mxscaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAddRmsNormDynamicMxQuant第一段接口
    ret = aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(
        x1, x2, gamma, nullptr, epsilon, 0, nullptr, 36, false, y, x, mxscale, nullptr, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddRmsNormDynamicMxQuant第二段接口
    ret = aclnnAddRmsNormDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
        LOG_PRINT("aclnnAddRmsNormDynamicMxQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto size = GetShapeSize(xShape);
    std::vector<int8_t> yRet(size, 0);
    ret = aclrtMemcpy(
        yRet.data(), yRet.size() * sizeof(yRet[0]), yDeviceAddr, size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %d\n", i, yRet[i]);
    }

    // 6. 释放aclTensor
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(gamma);
    aclDestroyTensor(y);
    aclDestroyTensor(mxscale);
    aclDestroyTensor(x);

    // 7. 释放device资源
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(mxscaleDeviceAddr);
    aclrtFree(xDeviceAddr);

    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
