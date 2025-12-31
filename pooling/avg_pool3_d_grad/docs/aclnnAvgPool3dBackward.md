# aclnnAvgPool3dBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

- 计算公式：
  反向时的输出数据input($N,C,D_{in},H_{in},W_{in}$)、梯度gradOutput($N,C,D_{out},H_{out},W_{out}$)和池化步长($stride$)、池化窗口大小kernelSize($kD,kH,kW$)的关系是

  $$
  D_{in} = (D_{out} - 1) * {stride[0]} + kernel\_size[0] - 2 * padding[0]
  $$

  $$
  H_{in} = (H_{out} - 1) * {stride[1]} + kernel\_size[1] - 2 * padding[1]
  $$

  $$
  W_{in} = (W_{out} - 1) * {stride[2]} + kernel\_size[2] - 2 * padding[2]
  $$

  若ceil_mode为true，且满足

  $$
  (D_{out} - 1) * stride[0] >= D_{in} + padding[0]
  $$
  
  则D_{out}的shape需减1。H_{out},W_{out}同理。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAvgPool3dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAvgPool3dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *stride,
  const aclIntArray *padding,
  bool               ceilMode,
  bool               countIncludePad,
  int64_t            divisorOverride,
  aclTensor         *output,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnAvgPool3dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```
## aclnnAvgPool3dBackwardGetWorkspaceSize

- **参数说明**：
  <table style="undefined;table-layout: fixed; width: 1478px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 264px">
  <col style="width: 253px">
  <col style="width: 262px">
  <col style="width: 148px">
  <col style="width: 135px">
  <col style="width: 146px">
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
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示输入梯度。</td>
      <td>-</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>正向过程中的输入，对应公式中的input。</td>
      <td>-</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>池化窗口大小，公式中的k。</td>
      <td>长度为1(KD = KH = KW)或3(KD, KH, KW)，数值必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>池化操作的步长，公式中的strides。</td>
      <td>长度为0（数值与kernelSize数值保持一致）或者1(SD = SH = SW)或者3(SD, SH, SW)，数值必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>在输入的D、H、W方向上padding补0的层数，公式中的paddings。</td>
      <td>长度为1(PD = PH = PW)或3(PD, PH, PW)，数值在[0, kernelSize/2]的范围内。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>ceilMode</td>
      <td>输入</td>
      <td>推导的输出out的shape是否向上取整。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>countIncludePad</td>
      <td>输入</td>
      <td>计算平均池化时是否包括padding填充的0。</td>
      <td>-</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divisorOverride</td>
      <td>输入</td>
      <td>取平均的除数。</td>
      <td>当值为0时，该属性不生效。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>output</td>
      <td>输出</td>
      <td>输出的tensor。</td>
      <td>shape与入参self相同</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4-5</td>
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
  </tbody></table>

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 267px">
  <col style="width: 124px">
  <col style="width: 775px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的gradOutput、self、kernelSize、padding或output是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>传入的gradOutput、self和output的数据类型/维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>传入kernelSize，stride, padding的维度不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>传入的kernelSize、stride某个维度值小于等于0，padding某个维度值小于0。</td>
    </tr>
    <tr>
      <td>属性padding超过kernelSize对应位置的1/2，例如paddingH=2，kernelSizeH=2，paddingH>kernelSizeH*1/2。</td>
    </tr>
    <tr>
      <td>output的shape与self的shape不一致。</td>
    </tr>
    <tr>
      <td>根据平均池化语义计算得到的gradOutput的shape与接口传入的gradOutput的shape不一致。</td>
    </tr>
  </tbody>
  </table>
## aclnnAvgPool3dBackward

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAvgPool3dBackwardGetWorkspaceSize获取。</td>
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
-  **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnAvgPool3dBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool3d_backward.h"

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
                    aclDataType dataType, aclTensor** tensor, aclFormat Format = aclFormat::ACL_FORMAT_ND) {
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
  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, Format, shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  // 1. （固定写法）device/stream初始化，参考acl对外接口列表
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> gradOutputShape = {1, 16, 1, 1, 1};
  std::vector<int64_t> selfShape = {1, 16, 4, 4, 4};
  std::vector<int64_t> kernelDims = {4, 4, 4};
  std::vector<int64_t> strideDims = {1, 1, 1};
  std::vector<int64_t> paddingDims = {0, 0, 0};
  std::vector<int64_t> outputShape = {1, 16, 4, 4, 4};
  bool ceilMode = false;
  int64_t divisorOverride = 0;
  bool countIncludePad = false;

  void* gradOutputDeviceAddr = nullptr;
  void* selfAddr = nullptr;
  void* outputAddr = nullptr;

  aclTensor* gradOutput = nullptr;
  aclTensor* selfInput = nullptr;
  aclTensor* output = nullptr;

  std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);
  std::vector<float> selfHostData(GetShapeSize(selfShape), 1);
  std::vector<float> outputHostData(GetShapeSize(outputShape), 1);
  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建inputshape aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfAddr, aclDataType::ACL_FLOAT,
  &selfInput);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建output
  ret = CreateAclTensor(outputHostData, outputShape, &outputAddr, aclDataType::ACL_FLOAT, &output);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建kernel aclIntArray
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 3);

  // 创建stride aclIntArray
  aclIntArray *stride = aclCreateIntArray(strideDims.data(), 3);

  // 创建paddings aclIntArray
  aclIntArray *padding = aclCreateIntArray(paddingDims.data(), 3);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAvgPool3dBackward第一段接口
  ret = aclnnAvgPool3dBackwardGetWorkspaceSize(gradOutput,
                                        selfInput,
                                        kernelSize,
                                        stride,
                                        padding,
                                        ceilMode,
                                        countIncludePad,
                                        divisorOverride,
                                        output,
                                        &workspaceSize,
                                        &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool3dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAvgPool3dBackward第二段接口
  ret = aclnnAvgPool3dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool3dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outputAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(selfInput);
  aclDestroyTensor(output);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfAddr);
  aclrtFree(outputAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```