# aclnnSwigluMxQuant

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     x    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     x    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：在Swish门控线性单元激活函数后添加DynamicMxQuant操作，实现x的SwigluMxQuant计算。
- swigluMode为0时的计算公式：  

  $$
  swigluOut_i = Swiglu(x_i)=Swish(A_i)*B_i
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示x<sub>i</sub>的前半部分，B<sub>i</sub>表示x<sub>i</sub>的后半部分。

- swigluMode为1时的计算公式：  

  $$
  x\_glu = x\_glu.clamp(min=None, max=clampLimit)
  $$
  
  $$
  x\_linear = x\_linear.clamp(min=-clampLimit, max=clampLimit)
  $$

  $$
  out\_glu = x\_glu * sigmoid(gluAlpha * x\_glu)
  $$

  $$
  swigluOut_i = out\_glu * (x\_linear + gluBias)
  $$

  $$
  y, mxscale = DynamicMxQuant(swigluOut_i)
  $$

  其中，x\_glu表示x<sub>i</sub>的偶数索引部分，x\_linear表示x<sub>i</sub>的奇数索引部分。

- 场景1，当scaleAlg为0时，DynamicMxQuant计算逻辑：
    - 将输入x在axis维度上按k = blocksize = 32个数分组，一组k个数  $\{\{V_i\}_{i=1}^{k}\}$ 动态量化为 $\{mxscale1, \{P_i\}_{i=1}^{k}\}$, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$

    - ​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。

    - emax: 对应数据类型的最大正则数的指数位。

        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
        | FLOAT8_E4M3FN |  8   |
        |  FLOAT8_E5M2  |  15  |

- 场景2，当scaleAlg为1时，只涉及FP8类型，DynamicMxQuant计算逻辑：
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
  - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnSwigluMxQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnSwigluMxQuant”接口执行计算。

```Cpp
aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *groupIndexOptional,
  int64_t          activateDim,
  bool             activateLeft,
  int64_t          swigluMode,
  double           clampLimit,
  double           gluAlpha,
  double           gluBias,
  int64_t          groupMode,
  int64_t          axis,
  int64_t          dstType,
  char            *roundModeOptional,
  int64_t          scaleAlg,
  double           maxDtypeValue,
  const aclTensor *yOut,
  const aclTensor *mxscaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnSwigluMxQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnSwigluMxQuantGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1480px"><colgroup>
  <col style="width: 301px">
  <col style="width: 115px">
  <col style="width: 150px">
  <col style="width: 350px">
  <col style="width: 177px">
  <col style="width: 104px">
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
    </tr></thead>
   <tbody>
       <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>输入待处理的数据，公式中的x。</td>
      <td><ul><li>shape为[X1,X2,...Xn,2H]，shape不超过7维，不小于2维。</li><li>输入x对应activateDim的维度需要是2的倍数。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td><ul><li>shape支持1维的Tensor，shape为[groupNum]，groupNum大于等于1且小于等于256。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateDim（int64_t）</td>
      <td>输入</td>
      <td>表示进行swish计算时，选择的指定切分轴。</td>
      <td><ul><li>activateDim的取值范围是：[-1, -2, xDim - 2, xDim - 1]（其中xDim指输入x的维度）。</li><li>当前仅支持-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateLeft（bool）</td>
      <td>输入</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td><ul><li>当值为false时，对输入的右半部分做激活。如果swigluMode为1，默认对x的偶数块做激活。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>swigluMode（int64_t）</td>
      <td>输入</td>
      <td>表示swiglu的计算模式。</td>
      <td><ul><li>取值范围为：[0, 1]。</li><li>0：表示传统swiglu计算方式。</li><li>1：表示swiglu的变种，使用奇偶分块方式，并支持clampLimit、激活系数以及偏差。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>clampLimit（double）</td>
      <td>输入</td>
      <td>表示变体swiglu使用的门限值。</td>
      <td><ul><li>该参数为可选参数。</li><li>用于对输入进行裁剪，其值需大于0且小于无穷，避免过大数值影响swiglu计算稳定性。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluAlpha（double）</td>
      <td>输入</td>
      <td>表示变体swiglu使用的缩放参数。</td>
      <td><ul><li>该参数为可选参数。</li><li>用于调整glu激活函数中线性部分的缩放。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>gluBias（double）</td>
      <td>输入</td>
      <td>表示变体swiglu使用的偏差参数。</td>
      <td><ul><li>该参数为可选参数。</li><li>用于在swiglu的线性计算中增加偏移。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>groupMode（int64_t）</td>
      <td>输入</td>
      <td>表示groupIndexOptional对应的模式</td>
      <td><ul><li>取值范围为：[0, 1]。</li><li>0：表示count模式。</li><li>1：表示cumsum模式。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>axis（int64_t）</td>
      <td>输入</td>
      <td>表示DynamicMxQuant量化发生的轴</td>
      <td><ul><li>取值范围为：[-1, -2, xDim - 2, xDim - 1]（其中xDim指输入x的维度）。当前仅支持-1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>dstType（int64_t）</td>
      <td>输入</td>
      <td>表示指定输出y的数据类型。</td>
      <td><ul><li>dstType的取值范围是:[35, 36, 40, 41]，分别对应FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>roundModeOptional（char*）</td>
      <td>输入</td>
      <td>表示对输出y结果的舍入模式。</td>
      <td><ul><li>取值范围是：["rint", "round", "floor"]。</li><li>当输出y的数据类型为FLOAT8_E5M2、FLOAT8_E4M3FN时，仅支持"rint"模式。</li><li>支持传入空指针，传入空指针时，则表示使用“rint”。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>scaleAlg（int64_t）</td>
      <td>输入</td>
      <td>表示mxscaleOut的计算方法，对应公式中的scaleAlg。</td>
      <td><ul><li>scaleAlg的取值范围是:[0, 1, 2]，取值为0代表场景1，为1代表场景2。</li><li>当dstType为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li><li>当前仅支持取值为0, 1。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>maxDtypeValue（double）</td>
      <td>输入</td>
      <td>表示DynamicMxQuant过程中指定的目标数据类型最大值。</td>
      <td><ul><li>maxDtypeValue取值不小于0，仅当scaleAlg=2且y的数据类型为FLOAT4_E2M1/FLOAT4_E1M2才生效。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>yOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示输入x量化后的对应结果，对应公式中的Pi和di。</td>
      <td><ul><li>当activateDim对应的x的尾轴时，shape为[X1,X2,...Xn,H]。</li><li>当activateDim对应的不是x的尾轴时，shape为[X1,X2,...,XactivateDim / 2,...,2H]。</li><li>当yOut的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，yOut的最后一维需要是2的倍数。</li></ul></td>
      <td>FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscaleOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示每个分组对应的量化尺度，对应公式中的mxscale和Sb</td>
      <td><ul><li>shape在axis轴上为x对应轴的值除以blocksize=32向上取整，并对其进行偶数pad，pad填充值为0。</li><li>当axis为非尾轴时，mxscaleOut输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>3-8</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的x、yOut或mxscaleOut是空指针。</td>
    </tr>
    <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的参数维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>activateDim不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>swigluMode不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>clampLimit不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>axis不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>dstType不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>roundModeOptional不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>scaleAlg不在指定的取值范围内。</td>
    </tr>
  </tbody></table>

## aclnnSwigluMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluMxQuantGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算：
  - aclnnSwigluMxQuant默认确定性实现。

- 输入x对应activateDim的维度需要是2的倍数，且x的维数必须大于1维。
- 当输出yOut的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，yOut的最后一维需要是2的倍数，x的最后一维需要是4的倍数。
- groupIndexOptional所有元素之和不能大于输入x除尾轴之外的剩余轴的乘积，groupIndexOptional的每个元素需要大于0。
- 输出yOut和mxscaleOut超出groupIndexOptional所有元素之和的部分未进行清理，该部分内存为垃圾数据。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_mx_quant.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {2, 64};
  std::vector<int64_t> groupIndexShape = {1};
  std::vector<int64_t> outShape = {2, 32};
  std::vector<int64_t> scaleOutShape = {2, 1, 2};

  void* xDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* groupIndex = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;

  std::vector<uint16_t> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
                                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<uint8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<uint8_t> scaleOutHostData = {0, 0, 0, 0};

  bool activateLeft = true;
  int64_t dstType = 36;
  int64_t activateDim = -1;
  int64_t swigluMode = 1;
  float clampLimit = 7.0;
  float gluAlpha = 1.0;
  float gluBias = 1.702;
  int64_t groupMode = 0;
  int64_t axis = -1;
  int64_t scaleAlg = 0;
  float maxDtypeValue = 0.0;

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scaleOut aclTensor
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnSwigluMxQuant第一段接口
  ret = aclnnSwigluMxQuantGetWorkspaceSize(x, nullptr, activateDim, activateLeft, swigluMode, clampLimit, gluAlpha, gluBias,groupMode, axis, dstType, "rint", scaleAlg, maxDtypeValue, out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnSwigluMxQuant第二段接口
  ret = aclnnSwigluMxQuant(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluMxQuant failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<uint8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(scaleOutDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
