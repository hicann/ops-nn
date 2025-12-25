# aclnnAvgPool3dBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：三维平均池化的反向传播，计算三维平均池化正向传播的输入梯度。

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

- `aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor* gradOutput, const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, bool ceilMode, bool countIncludePad, int64_t divisorOverride, aclTensor* output, uint64_t* workspaceSize, aclOpExecutor** executor)`

- `aclnnStatus aclnnAvgPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream)`

## aclnnAvgPool3dBackwardGetWorkspaceSize

- **参数说明**：
  - gradOutput(aclTensor*, 计算输入)：输入梯度，npu device侧的aclTensor，支持4维（$C,D_{out},H_{out},W_{out}$）或者5维（$N,C,D_{out},H_{out},W_{out}$）, N代表batch size，C代表通道数，D、H和W分别代表深度、高度和宽度, 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md),[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持BFLOAT16、FLOAT16、FLOAT32, 且需要与self一致。
  - self(aclTensor*, 计算输入)：输入数据，npu device侧的aclTensor，正向过程中的输入，对应公式中的$input$，支持4维（$C,D_{in},H_{in},W_{in}$）或者5维（$N,C,D_{in},H_{in},W_{in}$）, 支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md),[数据格式](../../../docs/zh/context/数据格式.md)支持ND。数据类型支持BFLOAT16、FLOAT16、FLOAT32，且需要与gradOutput一致。
  - kernelSize（aclIntArray*,计算输入）: Host侧的aclIntArray，表示池化窗口大小，INT64类型数组，长度为1(KD = KH = KW)或3(KD, KH, KW)。数值必须大于0。
  - stride（aclIntArray*,计算输入）: Host侧的aclIntArray，表示池化操作的步长，INT64类型数组，长度为0（数值与kernelSize数值保持一致）或者1(SD = SH = SW)或者3(SD, SH, SW)。数值必须大于0。
  - padding（aclIntArray*,计算输入）: Host侧的aclIntArray，表示在输入的D、H、W方向上padding补0的层数，INT64类型数组，长度为1(PD = PH = PW)或3(PD, PH, PW)。数值在[0, kernelSize/2]的范围内。
  - ceilMode（bool，计算输入）: 表示正向平均池化过程中推导的输出的shape是否向上取整（True表示向上取整）。数据类型支持BOOL。
  - countIncludePad（bool，计算输入）: 计算正向平均池化时是否包括padding填充的0（True表示包括填充的0）。数据类型支持BOOL。
  - divisorOverride（int64_t，计算输入）: 表示取平均的除数。如果指定，它将用作平均计算中的除数，当值为0时，该属性不生效，数据类型支持INT64。
  - output（aclTensor *，计算输出）: Device侧的aclTensor，shape与入参self相同。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)。 支持[数据格式](../../../docs/zh/context/数据格式.md)为ND。数据类型、[数据格式](../../../docs/zh/context/数据格式.md)需要与gradOutput一致, 数据类型支持BFLOAT16、FLOAT16、FLOAT32。
  - workspaceSize（uint64_t \*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001 (ACLNN_ERR_PARAM_NULLPTR): 1. 传入的gradOutput、self、kernelSize、padding或output是空指针。
  161002 (ACLNN_ERR_PARAM_INVALID): 1. 传入的gradOutput、self和output的数据类型/维度不在支持的范 围之内。
                                    2. 传入kernelSize，stride, padding的维度不在支持的范围之内。
                                    3. 传入的kernelSize、stride某个维度值小于等于0, padding某个维度值小于0。
                                    4. 属性padding超过kernelSize对应位置的1/2,例如paddingH=2,kernelSizeH=2,paddingH>kernelSizeH*1/2。
                                    5. output的shape与self的shape不一致。
                                    6. 根据平均池化语义计算得到的gradOutput shape与接口传入的gradOutput shape不一致。
  ```

## aclnnAvgPool3dBackward

- **参数说明**：

  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnAvgPool3dBackwardGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值**：

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