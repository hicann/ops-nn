# aclnnDequantSwigluQuantV2

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>昇腾910_95 AI处理器</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     ×    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品 </term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas 200/300/500 推理产品</term>       |     ×    |

## 功能说明
- 算子功能：在Swish门控线性单元激活函数前后添加dequant和quant操作，实现x的DequantSwigluQuant计算。本接口相较于[aclnnDequantSwigluQuant](aclnnDequantSwigluQuant.md)，新增了三个输入参数：dstType、roundModeOptional、activateDim，请根据实际情况选择合适的接口。
- 计算公式：

  $$
  dequantOut_i = Dequant(x_i)
  $$

  $$
  swigluOut_i = Swiglu(dequantOut_i)=Swish(A_i)*B_i
  $$

  $$
  out_i = Quant(swigluOut_i)
  $$

  其中，A<sub>i</sub>表示dequantOut<sub>i</sub>的前半部分，B<sub>i</sub>表示dequantOut<sub>i</sub>的后半部分。

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnDequantSwigluQuantV2GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDequantSwigluQuantV2”接口执行计算。

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weightScaleOptional,
  const aclTensor *activationScaleOptional,
  const aclTensor *biasOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *groupIndexOptional,
  bool             activateLeft,
  char            *quantModeOptional,
  int64_t          dstType,
  char            *roundModeOptional,
  int64_t          activateDim,
  const aclTensor *yOut,
  const aclTensor *scaleOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnDequantSwigluQuantV2(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```


## aclnnDequantSwigluQuantV2GetWorkspaceSize


- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 101px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 238px">
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
      <td>x</td>
      <td>输入</td>
      <td>输入待处理的数据，公式中的x。</td>
      <td><ul><li>shape为[X1,X2,...Xn,2H]，shape不超过8维，不小于2维。</li><li>输入x对应activateDim的维度需要是2的倍数。</li></ul></td>
      <td>FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
     <tr>
      <td>weightScaleOptional</td>
      <td>输入</td>
      <td><ul><li>shape支持1维或2维，shape表示为[2H]或[groupNum, 2H]，且取值2H和x最后一维保持一致。</li><li>可选参数，支持传空指针。当groupIndexOptional为空指针时，shape为[2H]；当groupIndexOptional不为空指针时，shape为[groupNum, 2H]。</li></ul></td>
      <td>-</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1或2</td>
      <td>√</td>
    </tr>
      <td>activationScaleOptional</td>
      <td>输入</td>
      <td>激活函数的反量化scale。</td>
      <td><ul><li>激活函数的反量化scale。</li><li>shape为[X1,X2,...Xn]，shape不超过7维不小于1维，维度比x的维度少一维，且shape与对应维度的x的shape一致。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
      <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>Matmul的bias，公式中的biasOptional。</td>
      <td><ul><li>shape支持1维，shape表示为[2H]，且取值2H和x最后一维保持一致。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantScaleOptional</td>
      <td>输入</td>
      <td>量化的scale，公式中的quantScaleOptional。</td>
      <td><ul><li>当quantModeOptional为static时，shape为1维，值为1，shape表示为shape[1]。</li><li>当quantModeOptional为dynamic时，shape为1维或2维，shape表示为[H], [2H]或[groupNum, H]。</li><li>当groupIndexOptional为空指针且activateDim为尾轴时，shape为[H]。</li><li>当groupIndexOptional不为空指针且activateDim为尾轴时，shape为[groupNum, H]。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
       <tr>
      <td>quantOffsetOptional</td>
      <td>输入</td>
      <td>量化的offset。</td>
      <td>暂时不支持此参数。</td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>-</td>
      <td>√</td>
    </tr>
      <tr>
      <td>groupIndexOptional</td>
      <td>输入</td>
      <td>MoE分组需要的group_index。</td>
      <td><ul><li>shape支持1维Tensor，shape为[groupNum]，groupNum大于等于1。</li><li>可选参数，支持传空指针。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
      <tr>
      <td>activateLeft</td>
      <td>输入</td>
      <td>表示是否对输入的左半部分做swiglu激活。</td>
      <td>当值为false时，对输入的右半部分做激活。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>quantModeOptional</td>
      <td>输入</td>
      <td>表示使用动态量化。</td>
      <td><ul><li>仅支持“dynamic”。</li><li>支持传入空指针，传入空指针时，则默认使用“static”。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>dstType</td>
      <td>输入</td>
      <td>表示指定输出y的数据类型。</td>
      <td>dstType的取值范围是:[2, 35, 36, 40, 41]，分别对应INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>roundModeOptional</td>
      <td>输入</td>
      <td>表示对输出y结果的舍入模式。</td>
      <td><ul><li>取值范围是：["rint", "round", "floor", "ceil", "trunc"]。</li><li>当输出y的数据类型为INT8、FLOAT8_E5M2、FLOAT8_E4M3FN时，仅支持"rint"模式。支持传入空指针，传入空指针时，则默认使用“rint”。</li></ul></td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>activateDim</td>
      <td>输入</td>
      <td>表示进行swish计算时，选择的指定切分轴。</td>
      <td><ul><li>activateDim的取值范围是：[-xDim, xDim - 1]（其中xDim指输入x的维度）。</li><li>当activateDim对应的不是x的尾轴时，不允许输入groupIndexOptional。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
       <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>-</td>
      <td><ul><li>当activateDim对应的x的尾轴时，shape为[X1,X2,...Xn,H]。</li><li>当activateDim对应的不是x的尾轴时，shape为[X1,X2,...,XactivateDim / 2,...,2H]。</li><li>当yOut的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，yOut的最后一维需要是2的倍数。</li><li>yOut的尾轴需要小于5120。</li></ul></td>
      <td>INT8、FLOAT8_E5M2、FLOAT8_E4M3FN、FLOAT4_E2M1、FLOAT4_E1M2</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>-</td>
      <td><ul><li>当activateDim对应的x的尾轴时，shape为[X1,X2,...,Xn]。</li><li>当activateDim对应的不是x的尾轴时，shape为[X1,X2,...,XactivateDim / 2,...,Xn]。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
       <tr>
      <td>workspaceSize</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>executor</td>
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
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
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
      <td><ul><li>传入的x、yOut或scaleOut是空指针。</li><li>当x的数据类型为int32时，weightScaleOptional是空指针。</li></ul></td>
    </tr>
    <tr>
      <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="8">161002</td>
      <td>输入或输出的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>输入或输出的参数维度不在支持的范围内。</td>
    </tr>
    <tr>
      <td>quantModeOptional不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>dstType不在指定的取值范围内。</td>
    </tr>
     <tr>
      <td>roundModeOptional不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>activateDim不在指定的取值范围内。</td>
    </tr>
    <tr>
      <td>weightScaleOptional、activationScaleOptional、biasOptional、quantScaleOptional、
                                           groupIndexOptional的shape与x不满足约束。</td>
    </tr>
  </tbody></table>


## aclnnDequantSwigluQuantV2
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDequantSwigluQuantV2GetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。


## 约束说明

- 输入x对应activateDim的维度需要是2的倍数，且x的维数必须大于1维。
- 当输入x的数据类型为INT32时，weightScaleOptional不能为空；当输入x的数据类型不为INT32时，weightScaleOptional不允许输入，传入空指针。
- 当输入x的数据类型不为INT32时，activationScaleOptional不允许输入，传入空指针。
- 当输入x的数据类型不为INT32时，biasOptional不允许输入，传入空指针。
- 当输出yOut的数据类型为FLOAT4_E2M1、FLOAT4_E1M2时，yOut的最后一维需要是2的倍数。
- 输出yOut的尾轴不超过5120.


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```C++
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dequant_swiglu_quant_v2.h"

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
  std::vector<int64_t> xShape = {2, 32};
  std::vector<int64_t> scaleShape = {16};
  std::vector<int64_t> offsetShape = {1};
  std::vector<int64_t> outShape = {2, 16};
  std::vector<int64_t> scaleOutShape = {2};
  void* xDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* scaleOutDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;
  aclTensor* out = nullptr;
  aclTensor* scaleOut = nullptr;
  std::vector<float> xHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                                    43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
  std::vector<float> scaleHostData = {1};
  std::vector<float> offsetHostData = {1};
  std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> scaleOutHostData = {0, 0};
  int64_t dstType = 2;
  int64_t activateDim = -1;

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建scale aclTensor
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
   // 创建offset aclTensor
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT8, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scaleOut aclTensor
  ret = CreateAclTensor(scaleOutHostData, scaleOutShape, &scaleOutDeviceAddr, aclDataType::ACL_FLOAT, &scaleOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnDequantSwigluQuantV2第一段接口
  ret = aclnnDequantSwigluQuantV2GetWorkspaceSize(x, nullptr, nullptr, nullptr, scale, nullptr, nullptr, false, "dynamic", dstType, "rint", activateDim, out, scaleOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnDequantSwigluQuantV2第二段接口
  ret = aclnnDequantSwigluQuantV2(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDequantSwigluQuantV2 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<int8_t> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
  }
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);
  aclDestroyTensor(out);
  aclDestroyTensor(scaleOut);
  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
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