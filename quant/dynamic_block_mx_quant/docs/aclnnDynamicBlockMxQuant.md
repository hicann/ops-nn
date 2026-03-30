# aclnnDynamicBlockMxQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/dynamic_block_mx_quant)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：对输入变量，以数据块（32\*32）为基本块进行MX量化转换为目的数据类型。在每个基本块中，根据scaleAlg的取值采取不同的scale算法计算出当前块对应的量化参数scale（1\*1），将其广播为scale1（32\*1）和scale2（1\*32）输出。同时对基本块中的每一个数除以scale，根据round_mode转换到对应的dst_type，得到量化结果y。

- 计算公式：
  - 场景1，当scaleAlg为0时：
    - 将输入x以数据块（32\*32）为基本块进行分组，一个数据块的数 $\{\{V_i\}_{i=1}^{32*32}\}$ 量化为 $\{scale, \{P_i\}_{i=1}^{32*32}\}$

      $$
      shared\_exp = floor(log_2(max_i(|V_i|))) - emax
      $$

      $$
      scale = 2^{shared\_exp}
      $$

      $$
      P_i = cast\_to\_dst\_type(V_i/scale, round\_mode), \space i\space from\space 1\space to\space 32*32
      $$

    - 同时将 scale（1\*1）广播为scale1（32\*1）和scale2（1\*32）作为输出scale1和scale2，​量化后的 $P_{i}$ 按对应的 $V_{i}$ 的位置组成输出y。

    - emax: 对应数据类型的最大正规数的指数位。

      |   DataType    | emax |
      | :-----------: | :--: |
      |  FLOAT4_E2M1  |  2   |
      |  FLOAT4_E1M2  |  0   |
      | FLOAT8_E4M3FN |  8   |
      |  FLOAT8_E5M2  |  15  |

  - 场景2，当scaleAlg为2时，只涉及FLOAT4_E2M1类型：
    - 将长向量按块分，每块长度为k，对每块单独计算一个块缩放因子$S_{fp32}^b$，再把块内所有元素用同一个$S_{fp32}^b$映射到目标低精度类型FLOAT4_E2M1，scale存储类型为FLOAT8_E8M0。如果最后一块不足k个元素，把缺失值视为0，按照完整块处理。
    - 找到该块中数值的最大绝对值:
      $$
      Amax(D_{fp32}^b)=max(\{|d_{i}|\}_{i=1}^{k})
      $$
    - 引入新的属性 dst_type_max。dst_type_max类型为float，默认值为0。默认输出时代表max_type为目标数据类型的最大值，如果传入其他数值，则需要按照传入的数值计算scale，有效值当前支持0.0、6.0和7.0，只支持在FLOAT4_E2M1场景设置该值。
    - 将FP32映射到目标数据类型FLOAT4_E2M1可表示的范围内。
      $$
      S_{fp32}^b = \frac{Amax(D_{fp32}^b)}{dst\_type\_max}
      $$
    - 将块缩放因子$S_{fp32}^b$转换为FP8格式下可表示的缩放值$S_{ue8m0}^b$
    - 从块的浮点缩放因子$S_{fp32}^b$中提取无偏指数$E_{int}^b$和尾数$M_{fixp}^b$
    - 为保证量化时不溢出，对指数进行向上取整，且在FP8可表示的范围内：
      $$
      E_{int}^b = \begin{cases} E_{int}^b + 1, & \text{如果} S_{fp32}^b \text{为正规数，且} E_{int}^b < 254 \text{且} M_{fixp}^b > 0 \\ E_{int}^b, & \text{其余情况} \end{cases}
      $$
    - 计算块缩放因子：$S_{ue8m0}^b=2^{E_{int}^b}$
    - 计算块转换因子：$R_{fp32}^b=\frac{1}{fp32(S_{ue8m0}^b)}$
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^b)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDynamicBlockMxQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicBlockMxQuant”接口执行计算。

```cpp
aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          scaleAlg,
  float            dstTypeMax,
  aclTensor       *yOut, 
  aclTensor       *scale1Out, 
  aclTensor       *scale2Out, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicBlockMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicBlockMxQuantGetWorkspaceSize

- **参数说明：**

  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape）| 非连续Tensor |  
  | ----- | ----- |----- |----- |----- |----- |----- |----- |
  | x (aclTensor\*) | 输入 | 表示输入x，对应公式中$V_i$。 | 当dstType为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。不支持空Tensor。 | FLOAT16、BFLOAT16 | ND | 2-3 | √ |
  | roundModeOptional (char\*)  | 输入 | 表示数据转换的模式，对应公式中的round_mode。 | 当dstType为40/41，对应输出yOut的数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}；<br> 当dstType为35/36，对应输出yOut数据类型为FLOAT8_E5M2/FLOAT8_E4M3FN时，仅支持{"rint"}；<br> 传入空指针时，采用"rint"模式。 | STRING | - | - | - |
  | dstType (int64_t) | 输入 | 表示指定数据转换后yOut的类型。 | 输入范围为{35, 36, 40, 41}，分别对应输出yOut的数据类型为{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2} | INT64 | - | - | - |
  | scaleAlg (int64_t) | 输入 | 表示scale1Out和scale2Out的计算方法。 | 当前支持取值0和2，分别代表OCP Microscaling Formats (Mx) Specification实现和Dynamic Dtype Range实现。 | INT64 | - | - | - |
  | dstTypeMax (float) | 输入 | 表示目标数据类型的最大值。| 在scaleAlg=2，dstType为FLOAT4_E2M1时生效，需要按照传入的数值计算scale。<br>当前仅支持dstTypeMax取值为0.0/6.0/7.0。 | FLOAT | - | - | - |
  | yOut (aclTensor\*) | 输出 | 表示输入x量化后的对应结果，对应公式中的$P_i$。 | shape和输入x一致。 | FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND | 2-3 | √ |
  | scale1Out (aclTensor*) | 输出 | 表示-1轴每个分组对应的量化尺度，对应公式中的scale1。 | shape为x的-1轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0。 | FLOAT8_E8M0 | ND | 3-4 | √ |
  | scale2Out (aclTensor*) | 输出 | 表示-2轴每个分组对应的量化尺度，对应公式中的scale2。 | shape为x的-2轴的值除以32向上取整，并对其进行偶数pad，pad填充值为0； <br>  scale2Out输出需要对每两行数据进行交织处理。 | FLOAT8_E8M0 | ND | 3-4 | √ |
  | workspaceSize (uint64_t\*)  | 输出 | 返回需要在Device侧申请的workspace大小。 | - | - | - | - | - |
  | executor (aclOpExecutor\*\*)  | 输出 | 返回op执行器，包含了算子计算流程。 | - | - | - | - | - |

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 126px">
  <col style="width: 677px">
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
      <td> x、roundModeOptional、dstType、scaleAlg、dstTypeMax、yOut、scale1Out、scale2Out的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、yOut、scale1Out或scale2Out的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>roundModeOptional、dstType、scaleAlg、dstTypeMax不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnDynamicBlockMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDynamicBlockMxQuantGetWorkspaceSize获取。</td>
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

 - 关于x、scale1Out、scale2Out的shape约束说明如下：
    - x的维度应该为2~3。
    - rank(scale1) = rank(x) + 1。
    - rank(scale2) = rank(x) + 1。
    - scale1.shape[-2] = (ceil(x.shape[-1] / 32) + 2 - 1) / 2。
    - scale2.shape[-3] = (ceil(x.shape[-2] / 32) + 2 - 1) / 2。
    - scale1.shape[-1] = 2。
    - scale2.shape[-1] = 2。
    - 其他维度与输入x一致。
    - 举例：输入x的shape为[B, M, N]，目的数据类型为FP8类时，对应的y的shape为[B, M, N]，scale1的shape为[B, M, (ceil(N/32) + 2 - 1) / 2, 2]，scale2的shape为[B, (ceil(M / 32) + 2 - 1) / 2, N, 2]。
 - 确定性说明：aclnnDynamicBlockMxQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include <cstdio>
  #include "aclnnop/aclnn_dynamic_block_mx_quant.h"
  

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

  int aclnnDynamicBlockMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> yOutShape = {1, 4};
      std::vector<int64_t> scale1OutShape = {1, 1, 2};
      std::vector<int64_t> scale2OutShape = {1, 4, 2};
      void* xDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* scale1OutDeviceAddr = nullptr;
      void* scale2OutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* scale1Out = nullptr;
      aclTensor* scale2Out = nullptr;
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
      std::vector<uint8_t> scale1OutHostData = {128, 0};
      std::vector<uint8_t> scale2OutHostData = {128, 0, 128, 0, 128, 0, 128, 0};
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t scaleAlg = 0;
      float dstTypeMax = 0;
      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yOut aclTensor
      ret = CreateAclTensor(yOutHostData, yOutShape, &yOutDeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &yOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yOutTensorPtr(yOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yOutDeviceAddrPtr(yOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale1Out aclTensor
      ret = CreateAclTensor(scale1OutHostData, scale1OutShape, &scale1OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale1Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale1OutTensorPtr(scale1Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> scale1OutDeviceAddrPtr(scale1OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建scale2Out aclTensor
      ret = CreateAclTensor(scale2OutHostData, scale2OutShape, &scale2OutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &scale2Out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scale2OutTensorPtr(scale2Out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> scale2OutDeviceAddrPtr(scale2OutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;

      // 调用aclnnDynamicBlockMxQuant第一段接口
      ret = aclnnDynamicBlockMxQuantGetWorkspaceSize(x, roundModeOptional, dstType, scaleAlg, dstTypeMax, yOut, scale1Out, scale2Out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnDynamicBlockMxQuant第二段接口
      ret = aclnnDynamicBlockMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuant failed. ERROR: %d\n", ret); return ret);

      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size1 = GetShapeSize(yOutShape);
      std::vector<uint8_t> yOutData(
          size1, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size1 * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size1; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      auto size2 = GetShapeSize(scale1OutShape);
      auto size3 = GetShapeSize(scale2OutShape);
      std::vector<uint8_t> scale1OutData(
          size2, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      std::vector<uint8_t> scale2OutData(
          size3, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      ret = aclrtMemcpy(scale1OutData.data(), scale1OutData.size() * sizeof(scale1OutData[0]), scale1OutDeviceAddr,
                        size2 * sizeof(scale1OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale1Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      ret = aclrtMemcpy(scale2OutData.data(), scale2OutData.size() * sizeof(scale2OutData[0]), scale2OutDeviceAddr,
                        size3 * sizeof(scale2OutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale2Out from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size2; i++) {
          LOG_PRINT("scale1Out[%ld] is: %d\n", i, scale1OutData[i]);
      }
      for (int64_t i = 0; i < size3; i++) {
          LOG_PRINT("scale2Out[%ld] is: %d\n", i, scale2OutData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicBlockMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicBlockMxQuantTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
