# aclnnDynamicMxQuantWithDualAxis

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_mx_quant_with_dual_axis)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：在-1轴和-2轴上同时进行目的数据类型为FLOAT4类、FLOAT8类的MX量化。在给定的-1轴和-2轴上，每32个数，计算出这两组数对应的量化尺度mxscale1、mxscale2作为输出mxscale1Out、mxscale2Out的对应部分，然后分别对两组数所有元素除以对应的mxscale1或mxscale2，根据round_mode转换到对应的dstType，得到量化结果y1和y2分别作为输出y1Out和y2Out的对应部分。在dstType为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scaleAlg的取值来指定计算mxscale的不同算法。

- 合轴说明：算子实现时，会对-2轴（不包含）之前的所有轴进行合轴处理。即对于输入shape为$(d_0, d_1, ..., d_{n-3}, d_{n-2}, d_{n-1})$的张量，-2轴之前的维度$(d_0, d_1, ..., d_{n-3})$会被合并为一个维度，等效于将输入reshape为$(d_0 \times d_1 \times ... \times d_{n-3}, d_{n-2}, d_{n-1})$后再进行量化计算。

- 计算公式：
  - 场景1，当scaleAlg为0时，即OCP Microscaling Formats (Mx) Specification实现：
    - 将输入x在-1轴上按照32个数进行分组，一组32个数 $\{\{V_i\}_{i=1}^{32}\}$ 量化为 $\{mxscale1, \{P_i\}_{i=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      mxscale1 = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/mxscale1, round\_mode), \space i\space from\space 1\space to\space 32
      $$

    - 同时，将输入x在-2轴上按照32个数进行分组，一组32个数 $\{\{V_j\}_{j=1}^{32}\}$ 量化为 $\{mxscale2, \{P_j\}_{j=1}^{32}\}$

      $$
      shared\_exp = floor(log_2(max_j(|V_j|))) - emax
      $$

      $$
      mxscale2 = 2^{shared\_exp}
      $$

      $$
      P_j = cast\_to\_dst\_type(V_j/mxscale2, round\_mode), \space j\space from\space 1\space to\space 32
      $$

    - -1轴​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y1Out，mxscale1按对应的-1轴维度上的分组组成输出mxscale1Out。-2轴​量化后的 $P_{j}$ 按对应的 $V_{j}$ 的位置组成输出y2Out，mxscale2按对应的-2轴维度上的分组组成输出mxscale2Out。

    - emax: 对应数据类型的最大正则数的指数位。

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - 场景2，当scaleAlg为1时，只涉及FP8类型（CuBALS Scale计算算法）：
    - **-1轴量化**：将输入x在-1轴上按照32个数进行分组，每组长度为32，对每组单独计算一个块缩放因子$S_{fp32}^b$，再把组内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FP8。如果最后一组不足32个元素，把缺失值视为0，按照完整组处理。
      - 找到该组中数值的最大绝对值:
        $$
        Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{32})
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
      - 应用到量化的最终步骤，对于每个组内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终-1轴输出的量化结果是$\left(S^b, [d^i]_{i=1}^{32}\right)$，其中$S^b$代表块的缩放因子，即$S_{ue8m0}^b$，$[d^i]_{i=1}^{32}$代表组内量化后的数据。
    - **-2轴量化**：同时，将输入x在-2轴上按照32个数进行分组，采用与-1轴相同的CuBALS Scale计算算法，对每组独立计算块缩放因子并量化。-2轴输出的量化结果是$\left(S^b, [d^j]_{j=1}^{32}\right)$。
    - -1轴量化结果组成输出y1Out，对应的块缩放因子组成输出mxscale1Out。-2轴量化结果组成输出y2Out，对应的块缩放因子组成输出mxscale2Out。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicMxQuantWithDualAxis”接口执行计算。

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(
  const aclTensor *x, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          scaleAlg, 
  const aclTensor *y1Out, 
  const aclTensor *mxscale1Out, 
  const aclTensor *y2Out, 
  const aclTensor *mxscale2Out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuantWithDualAxis(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 120px">
  <col style="width: 130px">
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
      <th>维度（shape）</th>
      <th>非连续Tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>x (aclTensor*)</td>
      <td>输入</td>
      <td>表示输入x，对应公式中V<sub>i</sub>。</td>
      <td><ul><li>目的类型为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。</li></ul></td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>roundModeOptional (char*)</td>
      <td>输入</td>
      <td>表示数据转换的模式，对应公式中的round_mode。</td>
      <td><ul><li>当dstType为40/41，对应输出y1Out和y2Out的数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}。</li><li>当dstType为35/36，对应输出y1Out和y2Out数据类型为FLOAT8_E5M2/FLOAT8_E4M3FN时，仅支持{"rint"}。</li><li>传入空指针时，采用"rint"模式。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType (int64_t)</td>
      <td>输入</td>
      <td>表示指定数据转换后y1Out和y2Out的类型。</td>
      <td><ul><li>输入范围为{35, 36, 40, 41}，分别对应输出y1Out和y2Out的数据类型为{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2}。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>scaleAlg (int64_t)</td>
      <td>输入</td>
      <td>表示mxscale1Out和mxscale2Out的计算方法，对应公式中的scaleAlg。</td>
      <td><ul><li>支持取值0和1，取值为0代表场景1，为1代表场景2。</li><li>当dstType为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>y1Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示输入x量化-1轴后的对应结果，对应公式中的P<sub>i</sub>。</td>
      <td><ul><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale1Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示-1轴每个分组对应的量化尺度，对应公式中的mxscale1。</td>
      <td><ul><li>shape为x的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>y2Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示输入x量化-2轴后的对应结果，对应公式中的P<sub>j</sub>。</td>
      <td><ul><li>shape和输入x一致。</li></ul></td>
      <td>FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mxscale2Out (aclTensor*)</td>
      <td>输出</td>
      <td>表示-2轴每个分组对应的量化尺度，对应公式中的mxscale2。</td>
      <td><ul><li>shape为x的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。</li><li>mxscale2Out输出需要对每两行数据进行交织处理。</li></ul></td>
      <td>FLOAT8_E8M0</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
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
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1000px"><colgroup>
  <col style="width: 253px">
  <col style="width: 130px">
  <col style="width: 720px">
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
      <td>x存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td> x、roundModeOptional、dstType、scaleAlg、y1Out、mxscale1Out、y2Out、mxscale2Out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、y1Out、y2Out、mxscale1Out或mxscale2Out的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>roundModeOptional、dstType、scaleAlg不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuantWithDualAxis

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 200px">
  <col style="width: 130px">
  <col style="width: 770px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize获取。</td>
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

 - 关于x、mxscale1Out、mxscale2Out的shape约束说明如下：
    - rank(mxscale1Out) = rank(x) + 1。
    - rank(mxscale2Out) = rank(x) + 1。
    - mxscale1Out.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
    - mxscale2Out.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2。
    - mxscale1Out.shape[-1] = 2。
    - mxscale2Out.shape[-1] = 2。
    - 其他维度与输入x一致。
    - 举例：输入x的shape为[B, M, N]，目的数据类型为FP8类时，对应的y1和y2的shape为[B, M, N]，mxscale1的shape为[B, M, (ceil(N/32)+2-1)/2, 2]，mxscale2的shape为[B, (ceil(M/32)+2-1)/2, N, 2]。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_mx_quant_with_dual_axis.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)
  
  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

      int64_t
      GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnDynamicMxQuantWithDualAxisTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> y1OutShape = {1, 4};
      std::vector<int64_t> y2OutShape = {1, 4};
      std::vector<int64_t> mxscale1OutShape = {1, 1, 2};
      std::vector<int64_t> mxscale2OutShape = {1, 4, 2};
      void* xDeviceAddr = nullptr;
      void* y1OutDeviceAddr = nullptr;
      void* mxscale1OutDeviceAddr = nullptr;
      void* y2OutDeviceAddr = nullptr;
      void* mxscale2OutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* y1Out = nullptr;
      aclTensor* mxscale1Out = nullptr;
      aclTensor* y2Out = nullptr;
      aclTensor* mxscale2Out = nullptr;
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      std::vector<uint8_t> y1OutHostData = {0, 72, 96, 120};
      std::vector<uint8_t> y2OutHostData = {0, 0, 0, 0};
      std::vector<uint8_t> mxscale1OutHostData = {128, 0};
      std::vector<uint8_t> mxscale2OutHostData = {0, 0, 122, 0, 125, 0, 128, 0};
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t scaleAlg = 0;
      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y1Out aclTensor
      ret = CreateAclTensor(y1OutHostData, y1OutShape, &y1OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y1OutTensorPtr(y1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y1OutDeviceAddrPtr(y1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建mxscale1Out aclTensor
      ret = CreateAclTensor(mxscale1OutHostData, mxscale1OutShape, &mxscale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale1OutTensorPtr(mxscale1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale1OutDeviceAddrPtr(mxscale1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y2Out aclTensor
      ret = CreateAclTensor(y2OutHostData, y2OutShape, &y2OutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &y2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> y2OutTensorPtr(y2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> y2OutDeviceAddrPtr(y2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建mxscale2Out aclTensor
      ret = CreateAclTensor(mxscale2OutHostData, mxscale2OutShape, &mxscale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscale2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscale2OutTensorPtr(mxscale2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscale2OutDeviceAddrPtr(mxscale2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 调用aclnnDynamicMxQuantWithDualAxis第一段接口
      ret = aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, y1Out, mxscale1Out, y2Out, mxscale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnDynamicMxQuantWithDualAxis第二段接口
      ret = aclnnDynamicMxQuantWithDualAxis(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxis failed. ERROR: %d\n", ret); return ret);

      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size1 = GetShapeSize(y1OutShape);
      auto size2 = GetShapeSize(y2OutShape);
      std::vector<uint8_t> y1OutData(
          size1, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
      std::vector<uint8_t> y2OutData(
          size2, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
      ret = aclrtMemcpy(y1OutData.data(), y1OutData.size() * sizeof(y1OutData[0]), y1OutDeviceAddr,
                        size1 * sizeof(y1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST) && 
                        aclrtMemcpy(y2OutData.data(), y2OutData.size() * sizeof(y2OutData[0]), y2OutDeviceAddr,
                        size2 * sizeof(y2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y1Out and y2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("y1Out[%ld] is: %d\n", i, y1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("y2Out[%ld] is: %d\n", i, y2OutData[i]);
      }
      size1 = GetShapeSize(mxscale1OutShape);
      size2 = GetShapeSize(mxscale2OutShape);
      std::vector<uint8_t> mxscale1OutData(
          size1, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      std::vector<uint8_t> mxscale2OutData(
          size2, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      ret = aclrtMemcpy(mxscale1OutData.data(), mxscale1OutData.size() * sizeof(mxscale1OutData[0]), mxscale1OutDeviceAddr,
                        size1 * sizeof(mxscale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(mxscale2OutData.data(), mxscale2OutData.size() * sizeof(mxscale2OutData[0]), mxscale2OutDeviceAddr,
                        size2 * sizeof(mxscale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscale2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("mxscale1Out[%ld] is: %d\n", i, mxscale1OutData[i]);
      }
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("mxscale2Out[%ld] is: %d\n", i, mxscale2OutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicMxQuantWithDualAxisTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantWithDualAxisTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
