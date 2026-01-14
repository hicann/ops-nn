# aclnnConvTbc

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## 产品支持情况

<table>
<tr>
<th style="text-align:left">产品</th>
<th style="text-align:center; width:100px">是否支持</th>
</tr>
<tr>
<td><term>昇腾910_95 AI处理器</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 200I/500 A2 推理产品</term></td>
<td style="text-align:center">×</td>
</tr>
<tr>
<td><term>Atlas 推理系列产品 </term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 训练系列产品</term></td>
<td style="text-align:center">√</td>
</tr>
<tr>
<td><term>Atlas 200/300/500 推理产品</term></td>
<td style="text-align:center">×</td>
</tr>
</table>

## 功能说明

- 接口功能：实现输入输出维度为 **T**（时间或空间维度）、**B**（批次）、**C**（通道）的一维卷积。
- 计算公式：
  假定输入（self）的 shape 是($H_{\text{in}},N,C_{\text{in}}$)，输出（out）的 shape 是($H_{\text{out}},N,C_{\text{out}}$)，那输出将被表示为：

  $$
  out_{N_i,C_{out j}} = bias(C_{out j}) + \sum_{k = 0}^{C_{in} - 1} weight(k, C_{out j}) \cdot self(N_i, k)
  $$

  其中$N$表示批次大小（batch size），$C$表示通道数，$H$代表时间或空间维度。

## 函数原型

每个算子分为<a href="../../../docs/zh/context/两段式接口.md">两段式接口</a>，必须先调用  aclnnConvTbcGetWorkspaceSize 接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用 aclnnConvTbc 接口执行计算。

```cpp
aclnnStatus aclnnConvTbcGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclTensor       *bias,
    int64_t                pad,
    aclTensor             *out,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvTbc(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream     stream)
```

## aclnnConvTbcGetWorkspaceSize

- **参数说明：**

  <table>
  <tr>
  <th style="width:170px">参数名</th>
  <th style="width:120px">输入/输出</th>
  <th style="width:300px">描述</th>
  <th style="width:400px">使用说明</th>
  <th style="width:212px">数据类型</th>
  <th style="width:100px">数据格式</th>
  <th style="width:100px">维度（shape）</th>
  <th style="width:145px">非连续 Tensor</th>
  </tr>
  <tr>
  <td>self</td>
  <td>输入</td>
  <td>公式中的 self，表示卷积输入。</td>
  <td><ul><li>shape 为 (N,C<sub>in</sub>,H<sub>in</sub>)。</li><li>支持空 Tensor。</li><li>数据类型与 weight 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>N≥0，C≥1，H≥0。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>ND、NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight</td>
  <td>输入</td>
  <td>公式中的 weight，表示卷积权重。</td>
  <td><ul><li>shape 为 (C<sub>out</sub>,C<sub>in</sub>,K)。</li><li>支持空 Tensor。</li><li>数据类型与 self 的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>所有维度≥1。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>ND、NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>bias</td>
  <td>输入</td>
  <td>公式中的 bias，表示卷积偏置。</td>
  <td><ul><li>一维且与 weight 第一维相等，不允许传入空指针。</li><li>数据类型与 self、weight 一致。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>pad</td>
  <td>输入</td>
  <td>表示 H 维度上左右的填充个数。</td>
  <td>大小应在 [0,255] 的范围内。</td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>out</td>
  <td>输出</td>
  <td>公式中的 out，表示卷积输出。</td>
  <td><ul><li>shape 为 (N,C<sub>out</sub>,H<sub>out</sub>)。</li><li>数据类型与 self 一致。</li><li>支持空 Tensor。</li><li>通道数等于 weight 第一维，其他维度≥0。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>ND、NCL</td>
  <td>3</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType</td>
  <td>输入</td>
  <td>用于判断 Cube 单元应该使用哪种计算逻辑进行运算。</td>
  <td><ul><li> 0 (KEEP_DTYPE): 保持输入数据类型进行计算。</li></ul><ul><li> 1 (ALLOW_FP32_DOWN_PRECISION): 允许 FLOAT 降低精度计算，提升性能。</li></ul><ul><li> 2 (USE_FP16): 使用 FLOAT16 精度进行计算。</li></ul><ul><li> 3 (USE_HF32): 使用 HIFLOAT32（混合精度）进行计算。</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输出</td>
  <td>返回需要在 Device 侧申请的 workspace 大小。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>输出</td>
  <td>返回 op 执行器，包含算子计算流程。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>


- **返回值：**

  aclnnStatus：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn 返回码</a>。

  一段接口完成入参校验，出现以下场景时报错：

  <table>
  <tr>
  <td align="center">返回值</td>
  <td align="center">错误码</td>
  <td align="center">描述</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的指针类型入参是空指针。</td>
  </tr>
  <tr>
  <td rowspan="9" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="9" align="left">161002</td>
  <td align="left">self、weight、bias、out 的数据类型和格式不在支持的范围内。</td>
  </tr>
  <tr><td align="left">self、weight、output 数据类型不一致。</td></tr>
  <tr><td align="left">out 的 shape 不满足 infershape 结果。</td></tr>
  <tr><td align="left">out 的 shape 中存在小于 0 的情况。</td></tr>
  <tr><td align="left">self 的 dim 不为 3。</td></tr>
  <tr><td align="left">weight 的 dim 不为 3。</td></tr>
  <tr><td align="left">bias 的 dim 不为 1。</td></tr>
  <tr><td align="left">self 的第二个维度值不等于 weight 的第二个维度值。</td></tr>
  <tr><td align="left">bias 的值不等于 weight 的第一个维度。</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API 内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">API 调用 npu runtime 的接口异常，如 SocVersion 不支持。</td>
  </tr>
  </table>

## aclnnConvTbc

- **参数说明：**

  <table>
  <tr>
  <th style="width:120px">参数名</th>
  <th style="width:80px">输入/输出</th>
  <th>描述</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 内存地址。</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输入</td>
  <td>在 Device 侧申请的 workspace 大小，由第一段接口 aclnnConvTbcGetWorkspaceSize 获取。</td>
  </tr>
  <tr>
  <td>executor</td>
  <td>输入</td>
  <td>op 执行器，包含了算子计算流程。</td>
  </tr>
  <tr>
  <td>stream</td>
  <td>输入</td>
  <td>指定执行任务的 Stream。</td>
  </tr>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn 返回码</a>。

## 约束说明

- 确定性计算
  - aclnnConvTbc默认确定性实现。

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
    <col style="width:70px">
    <col style="width:200px">
    <col style="width:200px">
    <col style="width:200px">
    <col style="width:200px">
    </colgroup>
   <thead>
    <tr>
     <th><term>约束类型</term></th>
     <th><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
     <th><term>Atlas 推理系列产品</term></th>
     <th><term>Atlas 训练系列产品</term></th>
     <th><term>昇腾 910_95 AI 处理器</term></th>
   </tr>
   </thead>
   <tbody>
   <tr>
     <th scope="row">self、weight</th>
     <td>
        <ul>
          <li>self、weight 数据类型不支持 HIFLOAT8。支持 N、C、L 维度大于等于 0。</li>
          <li>weight 支持 N、C 维度大于等于 0。</li>
        </ul>
     </td>
     <td colspan="2">
        <ul>
          <li>self、weight 数据类型不支持 BFLOAT16、HIFLOAT8。支持 N、C、L 维度大于等于 0。</li>
          <li>weight 支持 N、C 维度大于等于 0。</li>
        </ul>
     </td>
     <td>
        <ul>
           <li>self、weight 支持 N、C 维度大于等于 0，支持 L 维度大于等于 0（等于 0 的场景仅在 out 推导的 L 维度也等于 0 时支持）。</li>
           <li>weight 支持 N 维度大于等于 0（等于 0 的场景仅在 bias 的 N 维度和 out 的 C 维度也等于 0 时支持），C 维度大小的支持情况与 self 的 C 维度一致，L 维度的大小应该在 [1,255] 的范围内。</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">bias</th>
     <td>-</td>
     <td>
        <ul>
          bias 数据类型不支持 BFLOAT16。
        </ul>
     </td>
     <td>-</td>
     <td>
        <ul>
          当 self 数据类型为 HIFLOAT8 时，bias 数据类型最终会转成 FLOAT 参与计算。
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">out</th>
     <td colspan="3">
        <ul>
          out 支持 N、C、L 维度大于等于 0（等于 0 的场景仅在 self 的 N 或 C 或 L 维度等于 0 时支持）。
        </ul>
     </td>
     <td>
        <ul>
          <li>支持 N 维度大于等于 0，支持 C 维度大于等于 0（等于 0 的场景仅在 weight 的 N 维度等于 0 时支持）。</li>
          <li>支持 L 维度大于等于 0（等于 0 的场景仅在 self 的 L 维度等于 0 时支持）。</li>
        </ul>
     </td>
   </tr>
   <tr>
     <th scope="row">cubeMathType</th>
     <td>
        <ul>
          <li>为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。</li>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。</li>
        <ul>
     </td>
     <td colspan="2">
        <ul>
          <li>为 0(KEEP_DTYPE) 时，当输入是 FLOAT 暂不支持。</li>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 FLOAT16 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时暂不支持。</li>
        <ul>
     </td>
     <td>
        <ul>
          <li>为 1(ALLOW_FP32_DOWN_PRECISION) 时，当输入是 FLOAT 允许转换为 HFLOAT32 计算。</li>
          <li>为 2(USE_FP16) 时，当输入是 BFLOAT16 不支持该选项。</li>
          <li>为 3(USE_HF32) 时，当输入是 FLOAT 转换为 HFLOAT32 计算。</li>
        </ul>
     </td>
   </tr>
   </tbody>
  </table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考 <a href="../../../docs/zh/context/编译与运行样例.md">编译与运行样例</a>。
```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
  do {                                    \
    if (!(cond)) {                        \
      Finalize(deviceId, stream);         \
      return_expr;                        \
    }                                     \
  } while (0)

#define LOG_PRINT(message, ...)      \
  do {                               \
    printf(message, ##__VA_ARGS__);  \
  } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shape_size = 1;
  for (auto i: shape) {
    shape_size *= i;
  }
  return shape_size;
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
                    aclDataType dataType, aclTensor** tensor, aclFormat dataFormat) {
  auto size = GetShapeSize(shape) * sizeof(T);
  // 调用 aclrtMalloc 申请 device 侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用 aclrtMemcpy 将 host 侧数据拷贝到 device 侧内存上
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  // 计算连续 tensor 的 strides
  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  // 调用 aclCreateTensor 接口创建 aclTensor
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvTbcTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // check 根据自己的需要处理
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据 API 的接口自定义构造
  std::vector<int64_t> shapeInput = {2, 2, 3};
  std::vector<int64_t> shapeWeight = {3, 3, 4};
  std::vector<int64_t> shapeBias = {4};
  std::vector<int64_t> shapeResult = {2, 2, 4};

  void* deviceDataA = nullptr;
  void* deviceDataB = nullptr;
  void* deviceDataC = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* input = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  std::vector<float> inputData(GetShapeSize(shapeInput), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outputData(GetShapeSize(shapeResult), 1);

  // 创建 input aclTensor
  ret = CreateAclTensor(inputData, shapeInput, &deviceDataA, aclDataType::ACL_FLOAT, &input, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> inputTensorPtr(input, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataAPtr(deviceDataA, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建 weight aclTensor
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataB, aclDataType::ACL_FLOAT, &weight, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBPtr(deviceDataB, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建 bias aclTensor
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataC, aclDataType::ACL_FLOAT, &bias, aclFormat::ACL_FORMAT_ND);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataCPtr(deviceDataC, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建 result aclTensor
  ret = CreateAclTensor(outputData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result, aclFormat::ACL_FORMAT_NCL);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用 CANN 算子库 API，需要修改为具体的 Api 名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用 aclnnConvTbc 第一段接口
  ret = aclnnConvTbcGetWorkspaceSize(input, weight, bias, 1, result, 1, &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的 workspaceSize 申请 device 内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用 aclnnConvTbc 第二段接口
  ret = aclnnConvTbc(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbc failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧，需要根据具体 API 的接口定义修改
  auto size = GetShapeSize(shapeResult);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), deviceDataResult,
                    size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  return ACL_SUCCESS;
}

int main() {
  // 1. （固定写法）device/stream 初始化，参考 acl API 手册
  // 根据自己的实际 device 填写 deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvTbcTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvTbcTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
