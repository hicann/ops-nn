# aclnnDynamicMxQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

- 接口功能：目的数据类型为FLOAT4类、FLOAT8类的动态对称量化。在给定的轴axis上，根据每blocksize个数，计算出这组数对应的量化尺度mxscale作为输出mxscaleOut的对应部分，然后对这组数每一个除以mxscale，根据round_mode转换到对应的dstType，得到量化结果y作为输出yOut的对应部分。在dstType为FLOAT8_E4M3FN、FLOAT8_E5M2时，根据scaleAlg的取值来指定计算mxscale的不同算法。

- 计算公式：
  - 场景1，当scaleAlg为0时：
    - 将输入x在axis维度上按k = blocksize个数分组，一组k个数 {{V<sub>i</sub>}<sub>i=1</sub><sup>k</sup>} 动态量化为 {mxscale, {P<sub>i</sub>}}<sub>i=1</sub><sup>k</sup>}, k = blocksize

    $$
    shared\_exp = floor(log_2(max_i(|V_i|))) - emax \\
    mxscale = 2^{shared\_exp}\\
    P_i = cast\_to\_dst\_type(V_i/mxscale, round\_mode), \space i\space from\space 1\space to\space blocksize\\
    $$
    
    - ​量化后的P<sub>i</sub>按对应的V<sub>i</sub>的位置组成输出yOut，mxscale按对应的axis维度上的分组组成输出mxscaleOut。
    
    - emax: 对应数据类型的最大正则数的指数位。
    
        |   DataType    | emax |
        | :-----------: | :--: |
        |  FLOAT4_E2M1  |  2   |
        |  FLOAT4_E1M2  |  0   |
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
    - 应用到量化的最终步骤，对于每个块内元素，$d^i = DType(d_{fp32}^i \cdot R_{fp32}^n)$，最终输出的量化结果是$\left(S^b, [d^i]_{i=1}^k\right)$，其中$S^b$代表块的缩放因子，这里指$S_{ue8m0}^b$，$[d^i]_{i=1}^k$代表块内量化后的数据。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDynamicMxQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicMxQuant”接口执行计算。

```cpp
aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(
  const aclTensor *x, 
  int64_t          axis, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          blocksize, 
  int64_t          scaleAlg, 
  aclTensor       *yOut, 
  aclTensor       *mxscaleOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnDynamicMxQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnDynamicMxQuantGetWorkspaceSize

- **参数说明：**
  | 参数名 | 输入/输出 | 描述 | 使用说明 | 数据类型 | 数据格式 | 维度（shape）| 非连续Tensor |  
  | ----- | ----- |----- |----- |----- |----- |----- |----- |
  | x (aclTensor\*) | 输入 | 表示输入x，对应公式中$V_i$和$d_i$。 | 目的类型为FLOAT4_E2M1、FLOAT4_E1M2时，x的最后一维必须是偶数。 | FLOAT16、BFLOAT16 | ND | 1-7 | √ |
  | axis (int64_t) | 输入 | 表示量化发生的轴，对应公式中的axis。 | 取值范围为[-D, D-1]，D为x的shape的维数。 | INT64 | - | - | - |
  | roundModeOptional (char\*)  | 输入 | 表示数据转换的模式，对应公式中的round_mode。 | 当dstType为40/41，对应输出yOut数据类型为FLOAT4_E2M1/FLOAT4_E1M2时，支持{"rint", "floor", "round"}；<br> 当dstType为36/35，对应输出yOut数据类型为FLOAT8_E4M3FN/FLOAT8_E5M2时，仅支持{"rint"}；<br> 传入空指针时，采用"rint"模式。 | INT64 | - | - | - |
  | dstType (int64_t) | 输入 | 表示指定数据转换后yOut的类型，对应公式中的DType。 | 输入范围为{35, 36, 40, 41}，分别对应输出yOut的数据类型为{35:FLOAT8_E5M2, 36:FLOAT8_E4M3FN, 40:FLOAT4_E2M1, 41:FLOAT4_E1M2} | INT64 | - | - | - |
  | blocksize (int64_t)  | 输入 | 表示指定每次量化的元素个数，对应公式中的blocksize。 | 仅支持32的倍数，不能为0，且不能超过1024。 | INT64 | - | - | - |
  | scaleAlg (int64_t) | 输入 | 表示mxscaleOut的计算方法，对应公式中的scaleAlg。 | 支持取值0和1，取值为0代表场景1，为1代表场景2。当dstType为FLOAT4_E2M1/FLOAT4_E1M2时仅支持取值为0。 | INT64 | - | - | √ |
  | yOut (aclTensor\*) | 输出 | 表示输入x量化后的对应结果，对应公式中的$P_i$和$[d^i]_{i=1}^{k}$。 | shape和输入x一致。 | FLOAT4_E2M1、FLOAT4_E1M2、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND | 1-7 | - |
  | mxscaleOut (aclTensor*) | 输出 | 表示每个分组对应的量化尺度，对应公式中的mxscale和$S^b$。 | shape在axis轴上为x对应轴的值除以blocksize向上取整，并对其进行偶数pad，pad填充值为0； <br>  当axis为非尾轴时，mxscaleOut输出需要对每两行数据进行交织处理。 | FLOAT8_E8M0 | ND | 1-8 | - |
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
      <td> x、axis、roundModeOptional、dstType、blocksize、scaleAlg、yOut、mxscaleOut的数据类型和数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、yOut或mxscaleOut的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>axis、roundModeOptional、dstType、blocksize、scaleAlg不符合当前支持的值。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnDynamicMxQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDynamicMxQuantGetWorkspaceSize获取。</td>
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
  - aclnnDynamicMxQuant默认确定性实现。

 - 关于x、mxscaleOut的shape约束说明如下：
    - rank(mxscaleOut) = rank(x) + 1。
    - axis_change = axis if axis >= 0 else axis + rank(x)。
    - mxscaleOut.shape[axis_change] = (ceil(x.shape[axis] / blocksize) + 2 - 1) / 2。
    - mxscaleOut.shape[-1] = 2。
    - 其他维度与输入x一致。
 - 确定性说明：aclnnDynamicMxQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  
  #include "acl/acl.h"
  #include "aclnnop/aclnn_dynamic_mx_quant.h"
  
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
  
  int aclnnDynamicMxQuantTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
  
      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> xShape = {1, 4};
      std::vector<int64_t> yOutShape = {1, 4};
      std::vector<int64_t> mxscaleOutShape = {1, 1, 2};
      void* xDeviceAddr = nullptr;
      void* yOutDeviceAddr = nullptr;
      void* mxscaleOutDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* yOut = nullptr;
      aclTensor* mxscaleOut = nullptr;
      //对应BF16的值(0, 8, 64, 512)
      std::vector<uint16_t> xHostData = {0, 16640, 17024, 17408};
      //对应float8_e4m3的值(0, 4, 32, 256)
      std::vector<uint8_t> yOutHostData = {0, 72, 96, 120};
      //对应float8_e8m0的值(2)
      std::vector<uint8_t> mxscaleOutHostData = {{128, 0}};
      int64_t axis = -1;
      char* roundModeOptional = const_cast<char*>("rint");
      int64_t dstType = 36;
      int64_t blocksize = 32;
      int64_t scaleAlg = 0;
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
      // 创建mxscaleOut aclTensor
      ret = CreateAclTensor(mxscaleOutHostData, mxscaleOutShape, &mxscaleOutDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &mxscaleOut);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> mxscaleOutTensorPtr(mxscaleOut, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> mxscaleOutDeviceAddrPtr(mxscaleOutDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
     
      // 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
   
      // 调用aclnnDynamicMxQuant第一段接口
      ret = aclnnDynamicMxQuantGetWorkspaceSize(x, axis, roundModeOptional, dstType, blocksize, scaleAlg, yOut, mxscaleOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnDynamicMxQuant第二段接口
      ret = aclnnDynamicMxQuant(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuant failed. ERROR: %d\n", ret); return ret);
  
      //（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  
      // 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yOutShape);
      std::vector<uint8_t> yOutData(
          size, 0);  // C语言中无法直接打印fp4的数据，需要用uint8读出来，自行通过二进制转成fp4
      ret = aclrtMemcpy(yOutData.data(), yOutData.size() * sizeof(yOutData[0]), yOutDeviceAddr,
                        size * sizeof(yOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy yOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("yOut[%ld] is: %d\n", i, yOutData[i]);
      }
      size = GetShapeSize(mxscaleOutShape);
      std::vector<uint8_t> mxscaleOutData(
          size, 0);  // C语言中无法直接打印fp8的数据，需要用uint8读出来，自行通过二进制转成fp8
      ret = aclrtMemcpy(mxscaleOutData.data(), mxscaleOutData.size() * sizeof(mxscaleOutData[0]), mxscaleOutDeviceAddr,
                        size * sizeof(mxscaleOutData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy mxscaleOut from device to host failed. ERROR: %d\n", ret);
                return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("mxscaleOut[%ld] is: %d\n", i, mxscaleOutData[i]);
      }
      return ACL_SUCCESS;
  }
  
  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = aclnnDynamicMxQuantTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicMxQuantTest failed. ERROR: %d\n", ret); return ret);
  
      Finalize(deviceId, stream);
      return 0;
  }
  ```