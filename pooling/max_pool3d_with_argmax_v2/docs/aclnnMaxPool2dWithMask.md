# aclnnMaxPool2dWithMask

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
对于输入信号的输入通道，提供2维最大池化（max pooling）操作，输出池化后的值out和索引indices（采用mask语义计算得出）。

- 计算公式：

  - output tensor中每个元素的计算公式：

    $$
    out(N_j, C_j, h, w) = \max\limits_{{m\in[0,k_{H}-1],n\in[0,k_{W}-1]}}input(N_i,C_j,stride[0]\times h + m, stride[1]\times w + n)
    $$

  - out tensor的shape推导公式：

    $$
    [N, C, H_{out}, W_{out}]=[N,C,\lfloor{\frac{H_{in}+2 \times {padding[0] - dilation[0] \times(kernelSize[0] - 1) - 1}}{stride[0]}}\rfloor + 1,\lfloor{\frac{W_{in}+2 \times {padding[1] - dilation[1] \times(kernelSize[1] - 1) - 1}}{stride[1]}}\rfloor + 1]
    $$

  - indices tensor的shape推导公式：

    $$
    [N, C, H_{indices}, W_{indices}]=[N,C,k_h \times k_w,   (\lceil{\frac{H_{out} \times W_{out}}{16}}\rceil+1) \times 2 \times 16]
    $$

## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnMaxPool2dWithMaskGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxPool2dWithMask”接口执行计算。

- `aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor* self, const aclIntArray* kernelSize, const aclIntArray* stride, const aclIntArray* padding, const aclIntArray* dilation, bool ceilMode, aclTensor* out, aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)`
- `aclnnStatus aclnnMaxPool2dWithMask(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)`

## aclnnMaxPool2dWithMaskGetWorkspaceSize

- **参数说明：**
  - self（aclTensor*，计算输入）: 输入Tensor，公式中的input，Device侧aclTensor。shape支持3D或者4D，不支持其他shape。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)在3维时支持ND，在4维时支持NCHW。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16和BFLOAT16。
  - kernelSize（aclIntArray*，计算输入）: 表示最大池化的窗口大小，公式中的k，Host侧的aclIntArray，数据类型支持INT32、INT64，数组长度必须为1或2，且数组元素必须都大于0。
  - stride（aclIntArray*，计算输入）: 窗口移动的步长，公式中的stride，Host侧的aclIntArray，数据类型支持INT32、INT64。stride的长度为0时，stride的数值等于kernelSize的值。
  - padding（aclIntArray*，计算输入）: 每一条边补充的层数，公式中的padding_size，Host侧的aclIntArray，补充的位置填写“负无穷”，数据类型支持INT32、INT64，数组长度必须为1或2，且数组元素必须都大于等于0或者小于等于kernelSize/2。
  - dilation（aclIntArray*，计算输入）: 控制窗口中元素的步幅，公式中的dilation_size，Host侧的aclIntArray，数据类型支持INT32、INT64，值仅支持1。
  - ceilMode（bool，计算输入）: 控制计算输出out的shape推导时的取值模式，Host侧的Bool型。仅支持取值true或false。为true时表示推导公式中Hout和Wout的shape时用向上取整的方法，为false时即表示向下取整。
  - out（aclTensor*，计算输出）: 输出Tensor，公式中的out，Device侧的aclTensor。池化后的结果。shape需要按照功能说明中out的shape推导公式进行计算。[数据格式](../../../docs/context/数据格式.md)在3维时支持ND，在4维时支持NCHW，与self保持一致。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT、FLOAT16和BFLOAT16。
  - indices（aclTensor*，计算输出）: 输出Tensor，Device侧的aclTensor。最大值的索引位置组成的Tensor（采用mask语义）。数据类型仅支持INT8。shape需要按照功能说明中indices的shape推导公式进行计算，不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)在3维时支持ND，在4维时支持NCHW，与self保持一致，为自定义的mask值。
  - workspaceSize（uint64_t*，出参）: 返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）: 返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001(ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self、out、kernelSize、stride、padding、dilation是空指针。
  返回161002(ACLNN_ERR_PARAM_INVALID)：1. self的数据类型不在上述参数说明支持的范围内。
                                      2. self的数据格式不在上述参数说明支持的范围内。
                                      3. self的shape不是3维或者4维。
                                      4. self和out的数据类型、数据格式不一致。
                                      5. self的shape在C、H、W的某个轴为0。
                                      6. 通过公式推导出的输出out的shape的某个轴小于0。
                                      7. 通过公式推导出的输出out的shape与实际out的shape不一致。
                                      8. 通过公式推导出的输出indices的shape与实际indices的shape不一致。
                                      9. kernelSize的长度不等于1或者2。
                                      10. kernelSize中的数值中存在小于等于0的数值。
                                      11. stride的长度不等于0，1或2。
                                      12. stride的数值中存在小于等于0的值。
                                      13. padding的长度不等于1或2.
                                      14. padding的数值中存在小于0或者大于kernelSize/2的值。
                                      15. dilation的长度不等于1或2.
                                      16. dilation的数值不等于1。
```

## aclnnMaxPool2dWithMask

- **参数说明：**
  * workspace（void*，入参）: 在Device侧申请的workspace内存地址。
  * workspaceSize（uint64_t，入参）: 在Device侧申请的workspace大小，由第一段接口aclnnMaxPool2dWithMaskGetWorkspaceSize获取。
  * executor（aclOpExecutor*，入参）: op执行器，包含了算子计算流程。
  * stream（aclrtStream，入参）: 指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
- 输入数据暂不支持nan、-inf。
$$s_h >= (H_{in} + padding\_size) / (H_{out} - 1)$$
$$s_w >= (W_{in} + padding\_size) / (W_{out} - 1)$$

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
```Cpp
#include <unistd.h>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool2d_with_indices.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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

int main() {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 1, 4, 3};
  std::vector<int64_t> outShape = {1, 1, 2, 1};
  std::vector<int64_t> indicesShape = {1, 1, 4, 64};
  std::vector<int64_t> kernelSizeData = {2, 2};
  std::vector<int64_t> strideData = {2, 2};
  std::vector<int64_t> paddingData = {0, 0};
  std::vector<int64_t> dilationData = {1, 1};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclTensor* indices = nullptr;
  std::vector<float> selfHostData = {0.0850, -0.5147, -0.0212, -0.5654, -0.3222, 0.5847, 1.7510, 0.9954, 0.1842, 0.8392, 0.4835, 0.9213};
  std::vector<float> outHostData = {0, 0};
  std::vector<int8_t> indicesHostData(256, 0);

  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT8, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建输入数组
  aclIntArray* kernelSize = aclCreateIntArray(kernelSizeData.data(), 2);
  aclIntArray* stride = aclCreateIntArray(strideData.data(), 2);
  aclIntArray* padding = aclCreateIntArray(paddingData.data(), 2);
  aclIntArray* dilation = aclCreateIntArray(dilationData.data(), 2);
  const bool ceilMode = false;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  // aclnnMaxPool2dWithMask接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnMaxPool2dWithMask第一段接口
  ret = aclnnMaxPool2dWithMaskGetWorkspaceSize(self, kernelSize, stride, padding, dilation, ceilMode, out, indices, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithMaskGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxPool2dWithMask第二段接口
  ret = aclnnMaxPool2dWithMask(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool2dWithMask failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy out result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  size = GetShapeSize(indicesShape);
  std::vector<int8_t> indicesResultData(size, 0);
  ret = aclrtMemcpy(indicesResultData.data(), indicesResultData.size() * sizeof(indicesResultData[0]), indicesDeviceAddr,
                    size * sizeof(indicesResultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy indices result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %d\n", i, indicesResultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);
  aclDestroyTensor(indices);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  _exit(0);
}
```
