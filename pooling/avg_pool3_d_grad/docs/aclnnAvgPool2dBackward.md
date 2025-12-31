# aclnnAvgPool2dBackward

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    ×     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

- 接口功能：二维平均池化的反向传播，计算二维平均池化正向传播的输入梯度。

- 计算公式：假设二位平均池化正向的输入张量为$X$，输出张量为$Y$，池化窗口大小为$k*k$,步长为$s$,则$X$的梯度$\frac{\partial L}{\partial X}$计算公式为：

  $$
  \frac{\partial L}{\partial X_{i,j}}=\frac{1}{k^2}\sum_{n=0}^{k-1}\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}
  $$

  各参数意义如下：

  * $L$为损失函数，$\lfloor\cdot\rfloor$表示向上取整。
  * $X_{i,j}$表示输入特征图中第$i$行，第$j$列。
  * $Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}$表示输出特征图中第$\lfloor\frac{i*s+m}{k}\rfloor$行，第$\lfloor\frac{j*s+n}{k}\rfloor$列的像素值。
  * $k$表示池化窗口的大小。
  * $s$表示步长大小。
  * $\frac{\partial L}{\partial X_{i,j}}$表示损失函数L对输入特征图中第i行，第j列的像素值的偏导数。
  * $\frac{\partial L}{\partial Y_{\lfloor\frac{i*s+m}{k}\rfloor,\lfloor\frac{j*s+n}{k}\rfloor}}$表示损失函数$L$对图中第$\lfloor\frac{i*s+m}{k}\rfloor$行，$\lfloor\frac{j*s+n}{k}\rfloor$列的像素值偏导数。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnAvgPool2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAvgPool2dBackward”接口执行计算。

```Cpp
aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(
  const aclTensor   *gradOutput,
  const aclTensor   *self,
  const aclIntArray *kernelSize,
  const aclIntArray *stride,
  const aclIntArray *padding,
  bool               ceilMode,
  bool               countIncludePad,
  int64_t            divisorOverride,
  int8_t             cubeMathType,
  aclTensor         *gradInput,
  uint64_t          *workspaceSize,
  aclOpExecutor     **executor)
```
```Cpp
aclnnStatus aclnnAvgPool2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```
## aclnnAvgPool2dBackwardGetWorkspaceSize

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
      <td>不支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW、NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>正向过程中的输入。</td>
      <td>不支持空Tensor。</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW、NCL</td>
      <td>3-4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>池化窗口大小，公式中的k。</td>
      <td>长度为1(kH=kW)或2(kH, kW)，数值大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>池化操作的步长，公式中的strides。</td>
      <td>长度为1（sH=sW）或2（sH, sW），数值必须大于0。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>在输入的H、W方向上padding补0的层数，公式中的paddings。</td>
      <td>长度为1（padH=padW）或2（padH, padW），数值不能小于0。</td>
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
      <td>-</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>out</td>
      <td>指定Cube单元的计算逻辑</td>
      <td>-</td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>输出的tensor，公式中的out。</td>
      <td>-</td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW、NCL</td>
      <td>3-4</td>
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

  <term>cubeMathType</term>：支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不做处理。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为FLOAT32时，会转换为HFLOAT32计算。当输入为其他数据类型时不支持该选项。
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
      <td>传入的gradOutput、self、kernelSize、padding或gradInput是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>传入的gradOutput或gradInput的数据类型/数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>传入的self和gradInput的数据类型或数据格式不一致。</td>
    </tr>
    <tr>
      <td>传入的gradOutput、self、kernelSize、stride或gradInput维度不在支持范围内。</td>
    </tr>
    <tr>
      <td>传入的gradOutput、self、kernelSize、stride或gradInput某个维度值小于0。</td>
    </tr>
    <tr>
      <td>传入的cubeMathType不在支持范围内。</td>
    </tr>
    <tr>
      <td>属性padding超过kernelSize对应位置的1/2，例如paddingH=2，kernelSizeH=2，paddingH>kernelSizeH*1/2。</td>
    </tr>
    <tr>
      <td>传入的divisorOverride不在支持范围内。</td>
    </tr>
  </tbody>
  </table>
## aclnnAvgPool2dBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAvgPool2dBackwardGetWorkspaceSize获取。</td>
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
  - aclnnAvgPool2dBackward默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_avgpool2d_backward.h"

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
  std::vector<int64_t> gradOutputShape = {1, 16, 1, 1};
  std::vector<int64_t> selfShape = {1, 16, 4, 4};
  std::vector<int64_t> kernelDims = {4, 4};
  std::vector<int64_t> strideDims = {1, 1};
  std::vector<int64_t> paddingDims = {0, 0};
  bool ceilMode = false;
  int64_t divisorOverride = 0;
  bool countIncludePad = true;
  int8_t cubeMathType = 1;
  std::vector<int64_t> gradInputShape = {1, 16, 4, 4};

  void* gradOutputDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* gradInputDeviceAddr = nullptr;

  aclTensor* gradOutput = nullptr;
  aclTensor* self = nullptr;
  aclTensor* gradInput = nullptr;

  std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape) * 2, 1);
  std::vector<float> selfHostData(GetShapeSize(selfShape) * 2, 1);
  std::vector<float> gradInputHostData(GetShapeSize(gradInputShape) * 2, 1);

  // 创建gradOutput aclTensor
  ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建gradInput aclTensor
  ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput, aclFormat::ACL_FORMAT_NCHW);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建kernel aclIntArray
  aclIntArray *kernelSize = aclCreateIntArray(kernelDims.data(), 2);

  // 创建stride aclIntArray
  aclIntArray *stride = aclCreateIntArray(strideDims.data(), 2);

  // 创建paddings aclIntArray
  aclIntArray *padding = aclCreateIntArray(paddingDims.data(), 2);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnAvgPool2dBackward第一段接口
  ret = aclnnAvgPool2dBackwardGetWorkspaceSize(gradOutput,
                                       self,
                                       kernelSize,
                                       stride,
                                       padding,
                                       ceilMode,
                                       countIncludePad,
                                       divisorOverride,
                                       cubeMathType,
                                       gradInput,
                                       &workspaceSize,
                                       &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnAvgPool2dBackward第二段接口
  ret = aclnnAvgPool2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAvgPool2dBackward failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(gradInputShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), gradInputDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("out result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(gradOutput);
  aclDestroyTensor(self);
  aclDestroyTensor(gradInput);
  aclDestroyIntArray(kernelSize);
  aclDestroyIntArray(stride);
  aclDestroyIntArray(padding);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(gradOutputDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(gradInputDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```