# aclnnConvDepthwise2d

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/conv/convolution_forward)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>       |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>       |    √     |
|  <term>Atlas 200I/500 A2 推理产品</term>                      |     ×    |
|  <term>Atlas 推理系列产品</term>                              |     ×    |
|  <term>Atlas 训练系列产品</term>                              |     ×    |

## 功能说明

- 接口功能：DepthwiseConv2D是一种二维深度卷积运算。在该运算中，每个输入通道都会与一个独立的卷积核（称为深度卷积核）进行卷积。

- 计算公式：

  假定输入self的shape是 $(N, C_{\text{in}}, H, W)$，输出out的shape是 $(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})$，那么每个卷积核的输出将被表示为：

  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \text{weight}(C_{\text{out}_j}, C_{\text{in}_j}) \star \text{self}(N_i, C_{\text{in}_j})
  $$

  其中，$\star$ 表示卷积计算，$N$ 代表批次大小（batch size），$C$ 代表通道数，$W$ 和 $H$ 分别代表宽和高。

## 函数原型

每个算子分为<a href="../../../docs/zh/context/两段式接口.md">两段式接口</a>，必须先调用aclnnConvDepthwise2dGetWorkspaceSize接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用aclnnConvDepthwise2d接口执行计算。

```cpp
aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(
    const aclTensor       *self,
    const aclTensor       *weight,
    const aclIntArray     *kernelSize,
    const aclTensor       *bias,
    const aclIntArray     *stride,
    const aclIntArray     *padding,
    const aclIntArray     *dilation,
    aclTensor             *out,
    int8_t                 cubeMathType,
    uint64_t              *workspaceSize,
    aclOpExecutor         **executor)
```

```cpp
aclnnStatus aclnnConvDepthwise2d(
    void            *workspace,
    const uint64_t   workspaceSize,
    aclOpExecutor   *executor,
    aclrtStream      stream)
```

## aclnnConvDepthwise2dGetWorkspaceSize

- **参数说明**

  <table>
  <tr>
  <th style="width:220px">参数名</th>
  <th style="width:120px">输入/输出</th>
  <th style="width:280px">描述</th>
  <th style="width:350px">使用说明</th>
  <th style="width:200px">数据类型</th>
  <th style="width:120px">数据格式</th>
  <th style="width:130px">维度（shape）</th>
  <th style="width:145px">非连续Tensor</th>
  </tr>
  <tr>
  <td>self（aclTensor*）</td>
  <td>输入</td>
  <td>公式中的self，表示卷积输入。</td>
  <td><ul><li>支持空Tensor。</li><li>数据类型与weight的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>shape为（N,C<sub>in</sub>,H<sub>in</sub>,W<sub>in</sub>）。</li><li>N≥0，C≥1，H≥0，W≥0。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>weight（aclTensor*）</td>
  <td>输入</td>
  <td>公式中的weight，表示卷积权重。</td>
  <td><ul><li>支持空Tensor。</li><li>数据类型与self的数据类型需满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>）。</li><li>shape为（C<sub>out</sub>,C<sub>in</sub>/groups,K<sub>H</sub>,K<sub>W</sub>）。</li><li>weight第一维的数值应等于self通道数的整数倍，第二维仅能为1。</li><li>所有维度≥1，H、W维度应小于self的H、W维度。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>kernelSize（aclIntArray*）</td>
  <td>输入</td>
  <td>卷积核尺寸。</td>
  <td><ul><li>（INT64, INT64）型元组。</li><li>数值为weight的H、W两维的数值。</li></ul></td>
  <td>INT64</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>bias（aclTensor*）</td>
  <td>输入</td>
  <td>公式中的bias，表示卷积偏置。</td>
  <td><ul><li>shape为（C<sub>out</sub>）。</li><li>一维且数值与weight第一维相等。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16</td>
  <td>ND</td>
  <td>1</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>stride（aclIntArray*）</td>
  <td>输入</td>
  <td>卷积扫描步长。</td>
  <td><ul><li>数组长度需等于self维度-2。</li><li>strideH和strideW∈[1,63]。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>padding（aclIntArray*）</td>
  <td>输入</td>
  <td>对self的填充。</td>
  <td><ul><li>数组长度需等于self维度-2。</li><li>paddingH、paddingW∈[0,255]。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>dilation（aclIntArray*）</td>
  <td>输入</td>
  <td>卷积核中元素的间隔。</td>
  <td><ul><li>数组长度需等于self维度-2。</li><li>dilationH、dilationW∈[1,255]。</li></ul></td>
  <td>INT32</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>out（aclTensor*）</td>
  <td>输出</td>
  <td>公式中的out，表示卷积输出。</td>
  <td><ul><li>支持空Tensor。</li><li>shape为（N,C<sub>out</sub>,H<sub>out</sub>,W<sub>out</sub>）。</li><li>通道数等于weight第一维，H≥0，W≥0，其他维度≥1。</li></ul></td>
  <td>FLOAT、FLOAT16、BFLOAT16、HIFLOAT8</td>
  <td>NCHW</td>
  <td>4</td>
  <td style="text-align:center">√</td>
  </tr>
  <tr>
  <td>cubeMathType（int8_t）</td>
  <td>输入</td>
  <td>用于判断Cube单元应该使用哪种计算逻辑进行运算。</td>
  <td><ul><li> 0（KEEP_DTYPE）：保持输入数据类型进行计算。</li></ul><ul><li> 1（ALLOW_FP32_DOWN_PRECISION）：允许FLOAT降低精度计算，提升性能。</li></ul><ul><li> 2（USE_FP16）：使用FLOAT16精度进行计算。</li></ul><ul><li> 3（USE_HF32）：使用HFLOAT32（混合精度）进行计算。</li></ul></td>
  <td>INT8</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>workspaceSize（uint64_t*）</td>
  <td>输出</td>
  <td>返回需要在Device侧申请的workspace大小。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  <tr>
  <td>executor（aclOpExecutor**）</td>
  <td>输出</td>
  <td>返回op执行器，包含算子计算流程。</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td>-</td>
  <td style="text-align:center">-</td>
  </tr>
  </table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1430px"><colgroup>
    <col style="width:250px">
    <col style="width:130px">
    <col style="width:1050px">
    </colgroup>
   <thead>
  <tr>
  <td>返回值</td>
  <td>错误码</td>
  <td>描述</td>
  </tr> </thead>
  <tr>
  <td align="left">ACLNN_ERR_PARAM_NULLPTR</td>
  <td align="left">161001</td>
  <td align="left">传入的指针类型入参是空指针。</td>
  </tr>
  <tr>
  <td rowspan="6" align="left">ACLNN_ERR_PARAM_INVALID</td>
  <td rowspan="6" align="left">161002</td>
  <td align="left">self，weight，bias，out数据类型和数据格式不在支持的范围之内。</td>
  </tr>
  <tr><td align="left">self，weight，out数据类型不一致。</td></tr>
  <tr><td align="left">stride, padding, dilation输入shape不对。</td></tr>
  <tr><td align="left">weight和self通道数不满足要求。</td></tr>
  <tr><td align="left">out的shape不满足infer_shape结果。</td></tr>
  <tr><td align="left">self, weight，bias，out为不支持的空Tensor输入或输出。</td></tr>
  <tr>
  <td align="left">ACLNN_ERR_INNER_NULLPTR</td>
  <td align="left">561103</td>
  <td align="left">API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
  </tr>
  <tr>
  <td align="left">ACLNN_ERR_RUNTIME_ERROR</td>
  <td align="left">361001</td>
  <td align="left">API调用npu runtime的接口异常，如soc_version不支持。</td>
  </tr>
  </table>

## aclnnConvDepthwise2d

- **参数说明**

  <table>
  <tr>
  <th style="width:120px">参数名</th>
  <th style="width:80px">输入/输出</th>
  <th>描述</th>
  </tr>
  <tr>
  <td>workspace</td>
  <td>输入</td>
  <td>在Device侧申请的workspace内存地址。</td>
  </tr>
  <tr>
  <td>workspaceSize</td>
  <td>输入</td>
  <td>在Device侧申请的workspace大小，由第一段接口aclnnConvDepthwise2dGetWorkspaceSize获取。</td>
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
  </table>

- **返回值**

  `aclnnStatus`：返回状态码，具体参见 <a href="../../../docs/zh/context/aclnn返回码.md">aclnn返回码</a>。

## 约束说明

- 确定性计算
  - aclnnConvDepthwise2d默认确定性实现。

<table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width:150px">
  <col style="width:500px">
  <col style="width:500px">
  </colgroup>
  <thead>
  <tr>
    <th>约束类型</th>
    <th><term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></th>
    <th><term>Ascend 950PR/Ascend 950DT</term></th>
  </tr>
  </thead>
  <tbody>
  <tr>
    <th scope="row">self、weight</th>
    <td>
      <ul>
        <li>self、weight数据类型不支持HIFLOAT8。</li>
        <li>self通道数应小于等于65535。</li>
      </ul>
    </td>
    <td> - </td>
  </tr>
  <tr>
    <th scope="row">bias</th>
    <td>
      bias数据类型不支持HIFLOAT8、FLOAT8_E4M3FN。数据类型与input、weight一致。
    </td>
    <td>
      当self数据类型为HIFLOAT8时，bias数据类型最终会转成FLOAT参与计算。
    </td>
  </tr>
  <tr>
    <th scope="row">cubeMathType</th>
    <td>
      <ul>
        <li>为1（ALLOW_FP32_DOWN_PRECISION）时，当输入是FLOAT允许转换为HFLOAT32计算。</li>
        <li>为2（USE_FP16）时，当输入是BFLOAT16不支持该选项。</li>
        <li>为3（USE_HF32）时，当输入是FLOAT转换为HFLOAT32计算。</li>
      </ul>
    </td>
    <td>
      <ul>
        <li>为1（ALLOW_FP32_DOWN_PRECISION）时，当输入是FLOAT允许转换为HFLOAT32计算。</li>
        <li>为2（USE_FP16）时，当输入是BFLOAT16不支持该选项。</li>
        <li>为3（USE_HF32）时，当输入是FLOAT转换为HFLOAT32计算。</li>
      </ul>
    </td>
  </tr>
  <tr>
    <th scope="row">kernelSize约束</th>
    <td colspan="2">
      kernelSize数值为weight的H、W两维的大小。
    </td>
  </tr>
  <tr>
    <th scope="row">其他约束</th>
    <td colspan="2">
      self, weight, bias中每一组tensor的每一维大小都应不大于1000000。
    </td>
  </tr>
  </tbody>
</table>

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_NCHW,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

void Finalize(int32_t deviceId, aclrtStream& stream)
{
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
}

int aclnnConvDepthwise2dTest(int32_t deviceId, aclrtStream& stream)
{
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> shapeSelf = {2, 2, 2, 2};
  std::vector<int64_t> shapeWeight = {2, 1, 1, 1};
  std::vector<int64_t> shapeBias = {2};
  std::vector<int64_t> shapeResult = {2, 2, 2, 2};

  void* deviceDataSelf = nullptr;
  void* deviceDataWeight = nullptr;
  void* deviceDataBias = nullptr;
  void* deviceDataResult = nullptr;

  aclTensor* self = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* bias = nullptr;
  aclTensor* result = nullptr;
  aclIntArray* kernelSize = nullptr;
  aclIntArray* stride = nullptr;
  aclIntArray* padding = nullptr;
  aclIntArray* dilation = nullptr;

  std::vector<float> selfData(GetShapeSize(shapeSelf), 1);
  std::vector<float> weightData(GetShapeSize(shapeWeight), 1);
  std::vector<float> biasData(GetShapeSize(shapeBias), 1);
  std::vector<float> outData(GetShapeSize(shapeResult), 1);
  std::vector<int64_t> kernelSizeData = {1, 1};
  std::vector<int64_t> strideData = {1, 1};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};

  // 创建self aclTensor
  ret = CreateAclTensor(selfData, shapeSelf, &deviceDataSelf, aclDataType::ACL_FLOAT, &self);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> selfTensorPtr(self, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataSelfPtr(deviceDataSelf, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建weight aclTensor
  ret = CreateAclTensor(weightData, shapeWeight, &deviceDataWeight, aclDataType::ACL_FLOAT, &weight);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> weightTensorPtr(weight, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataWeightPtr(deviceDataWeight, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建bias aclTensor
  ret = CreateAclTensor(biasData, shapeBias, &deviceDataBias, aclDataType::ACL_FLOAT, &bias);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> biasTensorPtr(bias, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataBiasPtr(deviceDataBias, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outData, shapeResult, &deviceDataResult, aclDataType::ACL_FLOAT, &result);
  std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outputTensorPtr(result, aclDestroyTensor);
  std::unique_ptr<void, aclError (*)(void *)> deviceDataResultPtr(deviceDataResult, aclrtFree);
  CHECK_FREE_RET(ret == ACL_SUCCESS, return ret);

  kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> kernelSizePtr(kernelSize, aclDestroyIntArray);
  CHECK_FREE_RET(kernelSize != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  stride = aclCreateIntArray(strideData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> stridePtr(stride, aclDestroyIntArray);
  CHECK_FREE_RET(stride != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  padding = aclCreateIntArray(paddingData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> paddingPtr(padding, aclDestroyIntArray);
  CHECK_FREE_RET(padding != nullptr, return ACL_ERROR_INTERNAL_ERROR);
  dilation = aclCreateIntArray(dilationData.data(), 2);
  std::unique_ptr<aclIntArray, aclnnStatus (*)(const aclIntArray *)> dilationPtr(dilation, aclDestroyIntArray);
  CHECK_FREE_RET(dilation != nullptr, return ACL_ERROR_INTERNAL_ERROR);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnConvDepthwise2d第一段接口
  ret = aclnnConvDepthwise2dGetWorkspaceSize(self, weight, kernelSize, bias, stride, padding, dilation, result, 1,
                                             &workspaceSize, &executor);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    workspaceAddrPtr.reset(workspaceAddr);
  }
  // 调用aclnnConvDepthwise2d第二段接口
  ret = aclnnConvDepthwise2d(workspaceAddr, workspaceSize, executor, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2d failed. ERROR: %d\n", ret); return ret);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
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
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = aclnnConvDepthwise2dTest(deviceId, stream);
  CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvDepthwise2dTest failed. ERROR: %d\n", ret); return ret);

  Finalize(deviceId, stream);
  return 0;
}
```
