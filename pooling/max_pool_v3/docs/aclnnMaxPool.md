# aclnnMaxPool

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                       |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：
对于dim=3 或4维的输入张量，进行最大池化（max pooling）操作。
- 计算公式：
  - 当ceilMode=False时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lfloor{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rfloor + 1,\lfloor{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rfloor + 1]
    $$

  - 当ceilMode=True时，out tensor的shape中H和W维度推导公式：

    $$
    [H_{out}, W_{out}]=[\lceil{\frac{H_{in}+  padding\_size_{Htop} + padding\_size_{Hbottom} - {dilation\_size \times(k_h - 1) - 1}}{s_h}}\rceil + 1,\lceil{\frac{W_{in}+ padding\_size_{Wleft} + padding\_size_{Wright} - {dilation\_size \times(k_w - 1) - 1}}{s_w}}\rceil + 1]
    $$

    - 滑窗左上角起始位处在下或右侧pad填充位上或者界外（无法取到有效值）时，舍弃该滑窗结果，在上述推导公式基础上对应空间轴shape需减去1：

      $$
      \begin{cases}
      H_{out}=H_{out} - 1& \text{if } (H_{out}-1)*s_h>=H_{in}+padding\_size_{Htop} \\
      W_{out}=W_{out} - 1& \text{if } (W_{out}-1)*s_w>=W_{in}+padding\_size_{Wleft}  \\
      \end{cases}\\
      $$

## 函数原型
每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnMaxPoolGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMaxPool”接口执行计算。

 * `aclnnStatus aclnnMaxPoolGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelShape, const aclIntArray *strides, const int64_t autoPad, const aclIntArray *pads, const aclIntArray *dilations, const int64_t ceilMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`
 * `aclnnStatus aclnnMaxPool(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMaxPoolGetWorkspaceSize

- **参数说明：**
  - self（aclTensor\*，计算输入）：Device侧的aclTensor输入，对应公式中的H_in和W_in。维度需要是3D（维度分别代表C，H，W）或4D（维度分别代表N，C，H，W），N为Batchsize大小，C为Tensor通道大小，H为Tensor高度，W为Tensor宽度，不支持其他维度。支持[非连续的Tensor](../../../docs/zh/context/非连续的Tensor.md)，[数据格式](../../../docs/zh/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT、BF16、INT32、INT64、UINT8、INT16、INT8、UINT16。
  - kernelShape（aclIntArray\*，计算输入）：表示最大池化的窗口大小，对应公式中的k_h和k_w，长度为1或2，且数组元素必须都大于0。
  - strides（aclIntArray\*，计算输入）：窗口移动的步长，对应公式中的s_h和s_w，数组长度为0、1或2，且数组元素必须都大于0。当数组长度为0时，strides取默认值为1。
  - autoPad（int64_t，计算输入）：指定padding的方式。其中0代表"NOTSET"，并且只支持数值0。
  - pads（aclIntArray\*，计算输入）：沿着空间轴方向开始和结束的位置填充，对应公式中的padding_size。长度为0、1、2或4。当数组长度为0时，不进行填充。当数组长度为1时，H_top、H_bottom、W_left、W_right填充同一个值。当数组长度为2时，H_top、H_bottom分别填充数组第1个值，W_left、W_right分别填充数组第2个值。当数组长度为4时，按[H_top、W_left、H_bottom、W_right]位置关系进行填充。单个空间轴方向填充量之和需小于等于对应方向kernelShape。
  - dilations（aclIntArray\*，计算输入）：沿着核空间轴方向的膨胀值，对应公式中的dilation_size，并且只支持数值为1的输入场景。长度为0、1、2或4。
  - ceilMode（int64_t，计算输入）：计算输出形状的取整模式，为0时，代表False，向下取整；非0值时，代表True，向上取整。
  - out（aclTensor\*，计算输出）：数据类型和self一致。shape由上述公式推导出。数据格式和维度与输入self一致。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT16、FLOAT32。
    - <term>Ascend 950PR/Ascend 950DT</term>：数据类型支持FLOAT16、FLOAT、BF16、INT32、INT64、UINT8、INT16、INT8、UINT16。
  - workspaceSize（uint64_t\*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：1. 传入的self或out是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：1. self的数据类型和数据格式不在支持的范围之内。
                                        2. self的维度不是3D或4D。
                                        3. 根据最大池化语义计算的output shape与指定shape不一致。
                                        4. kernelShape的长度不等于1或2。
                                        5. kernelShape的数值中存在小于等于0。
                                        6. strides的长度不等于0、1或2。
                                        7. strides的数值中存在小于等于0。
                                        8. pads的长度不等于0、1、2或4。
                                        9. 单个空间轴方向pad填充量之和需小于等于对应方向kernelShape。
                                        10. dilation的长度不等于0、1、2或4。
                                        11. dilation的数值不等于1。
  ```

## aclnnMaxPool
- **参数说明：**
  - workspace(void\*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnMaxPoolGetWorkspaceSize获取。
  - executor(aclOpExecutor\*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算：
  - aclnnMaxPool默认确定性实现。

## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_max_pool.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {2, 2, 3, 3};
  std::vector<int64_t> outShape = {2, 2, 2, 2};
  std::vector<int64_t> kernel_size = {2, 2};
  std::vector<int64_t> strides_size = {2, 2};
  std::int64_t autoPads = 0;
  std::vector<int64_t> padding_size = {0, 0, 0, 0};
  std::vector<int64_t> dilation_size = {1, 1};
  std::int64_t ceilMode = 1;

  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* out = nullptr;
  aclIntArray* kernel_shape = aclCreateIntArray(kernel_size.data(), 2);
  aclIntArray* strides = aclCreateIntArray(strides_size.data(), 2);
  aclIntArray* padding = aclCreateIntArray(padding_size.data(), 4);
  aclIntArray* dilations = aclCreateIntArray(dilation_size.data(), 2);
  std::vector<float> selfHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                     10, 11, 12, 13, 14, 15, 16, 17, 18,
                                     19, 20, 21, 22, 23, 24, 25, 26, 27,
                                     28, 29, 30, 31, 32, 33, 34, 35, 36};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0};
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnMaxPool第一段接口
  ret = aclnnMaxPoolGetWorkspaceSize(self, kernel_shape, strides, autoPads, padding, dilations, ceilMode, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPoolGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnMaxPool第二段接口
  ret = aclnnMaxPool(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMaxPool failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> outData(size, 0);
  ret = aclrtMemcpy(outData.data(), outData.size() * sizeof(outData[0]), outDeviceAddr,
                    size * sizeof(outData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, outData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
