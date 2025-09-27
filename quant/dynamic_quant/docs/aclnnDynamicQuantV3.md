# aclnnDynamicQuantV3

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    ×     |

## 功能说明

- 算子功能：为输入张量进行动态量化。在MOE场景下，每个专家的smoothScalesOptional是不同的，根据输入的groupIndexOptional进行区分。支持对称/非对称量化。支持pertoken/pertensor量化模式。相较aclnnDynamicQuantV2，新增了pertensor量化模式，通过quantMode参数指定。

- 计算公式：
  - 对称量化：
    - 若不输入smoothScalesOptional，则
      $$
        scaleOut=row\_max(abs(x))/DTYPE_MAX
      $$
      $$
        yOut=round(x/scaleOut)
      $$
    - 若输入smoothScalesOptional，则
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=row\_max(abs(input))/DTYPE_MAX
      $$
      $$
        yOut=round(input/scaleOut)
      $$
  - 非对称量化：
    - 若不输入smoothScalesOptional，则
      $$
        scaleOut=(row\_max(x) - row\_min(x))/(DTYPE_MAX - DTYPE_MIN)
      $$
      $$
        offset=DTYPE_MAX-row\_max(x)/scaleOut
      $$
      $$
        yOut=round(x/scaleOut+offset)
      $$
    - 若输入smoothScalesOptional，则
      $$
        input = x\cdot smoothScalesOptional
      $$
      $$
        scaleOut=(row\_max(input) - row\_min(input))/(DTYPE_MAX - DTYPE_MIN)
      $$
      $$
        offset=DTYPE_MAX-row\_max(input)/scaleOut
      $$
      $$
        yOut=round(input/scaleOut+offset)
      $$
  其中row_max/row_min代表每行求最大/最小值；如果quantMode为“pertensor”，表示求整个tensor的最大/最小值。DTYPE_MAX是输出类型的最大值，DTYPE_MIN是输出类型的最小值。


## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnDynamicQuantV3GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDynamicQuantV3”接口执行计算。

* `aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(const aclTensor* x, const aclTensor* smoothScalesOptional, const aclTensor* groupIndexOptional, int64_t dstType, bool isSymmetrical, const char* quantMode, const aclTensor* yOut, const aclTensor* scaleOut, const aclTensor* offsetOut, uint64_t* workspaceSize, aclOpExecutor** executor)`
* `aclnnStatus aclnnDynamicQuantV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnDynamicQuantV3GetWorkspaceSize

- **参数说明：**

  - x（aclTensor*, 计算输入）：算子输入的Tensor，shape维度要大于1，不超过8。Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - smoothScalesOptional（aclTensor*, 计算输入）：算子输入的smoothScales，当没有groupIndexOptional时，shape维度是x的最后一维；当有groupIndexOptional时，shape是两维，第一维大小是专家数，不能超过1024。第二维大小是x的最后一维。Device侧的aclTensor，数据类型支持FLOAT16、BFLOAT16，并且数据类型要和x保持一致，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - groupIndexOptional（aclTensor*, 计算输入）：算子输入的groupIndex，Device侧的aclTensor，数据类型支持INT32，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。 shape只支持一维，且维度大小等于smoothScalesOptional的第一维。groupIndexOptional非nullptr时，smoothScalesOptional必须非nullptr。
  - dstType (int64_t, 计算输入)：输出yOut的数据类型对应的枚举值，Host侧的int。
  - isSymmetrical (bool, 计算输入)：指定是否对称量化，Host侧的bool。
  - quantMode (char*, 计算输入)：用于指定量化的模式，Host侧的char*。当前支持"pertoken"和"pertensor"。当quantMode取"pertensor"时，groupIndexOptional必须是nullptr。
  - yOut（aclTensor*, 计算输出）：量化后的输出Tensor，类型由dstType指定。数据类型为INT32时，最后一维是x最后一维的1/8，其他数据类型时shape和x保持一致。Device侧的aclTensor，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
  - scaleOut（aclTensor*, 计算输出）：量化使用的scale，Device侧的aclTensor，数据类型支持FLOAT，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - quantMode是pertoken时，shape为x的shape剔除最后一维。quantMode是pertensor时，shape为(1,)。
  - offsetOut（aclTensor*, 计算输出）：非对称量化使用的offset，shape和scale一致，Device侧的aclTensor，数据类型支持FLOAT，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，[数据格式](../../../docs/context/数据格式.md)支持ND。仅在isSymmetrical是false时支持，如果isSymmetrical是true，offsetOut需要传nullptr。
  - workspaceSize（uint64_t\*, 出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor\*\*, 出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  返回161001 (ACLNN_ERR_PARAM_NULLPTR)：传入的x或out参数是空指针。
  返回161002 (ACLNN_ERR_PARAM_INVALID)：参数的数据类型、数据格式、维度等不在支持的范围内。
  返回561001 (ACLNN_ERR_INNER_CREATE_EXECUTOR)：内部创建aclOpExecutor失败。
  ```

## aclnnDynamicQuantV3

- **参数说明：**
  - workspace(void \*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnDynamicQuantV3GetWorkspaceSize获取。
  - executor(aclOpExecutor \*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
yOut的数据类型为INT4时，需满足x和yOut的最后一维能被2整除。
yOut的数据类型为INT32时，需满足x的最后一维能被8整除。
当有groupIndexOptional时，专家数不超过x剔除最后一维的各个维度乘积。groupIndexOptional的值需要是一组不小于零且非递减的数组，且最后一个值和x剔除最后一维的各个维度乘积相等。若不满足该条件，结果无实际意义。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dynamic_quant_v3.h"

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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
  auto size = GetShapeSize(shape);
  std::vector<float> resultData(size, 0);
  auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                         *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("mean result[%ld] is: %f\n", i, resultData[i]);
  }
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
  int rowNum = 4;
  int rowLen = 2;
  int groupNum = 2;
  std::vector<int64_t> xShape = {4, 2};
  std::vector<int64_t> smoothShape = {groupNum, rowLen};
  std::vector<int64_t> groupShape = {groupNum};
  std::vector<int64_t> yShape = {4, 2};
  std::vector<int64_t> scaleShape = {4};
  std::vector<int64_t> offsetShape = {4};

  void* xDeviceAddr = nullptr;
  void* smoothDeviceAddr = nullptr;
  void* groupDeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  void* scaleDeviceAddr = nullptr;
  void* offsetDeviceAddr = nullptr;

  aclTensor* x = nullptr;
  aclTensor* smooth = nullptr;
  aclTensor* group = nullptr;
  aclTensor* y = nullptr;
  aclTensor* scale = nullptr;
  aclTensor* offset = nullptr;

  std::vector<aclFloat16> xHostData;
  std::vector<aclFloat16> smoothHostData;
  std::vector<int32_t> groupHostData = {2, rowNum};
  std::vector<int8_t> yHostData;
  std::vector<float> scaleHostData;
  std::vector<float> offsetHostData;
  for (int i = 0; i < rowNum; ++i) {
    for (int j = 0; j < rowLen; ++j) {
      float value1 = i * rowLen + j;
      xHostData.push_back(aclFloatToFloat16(value1));
      yHostData.push_back(0);
    }
    scaleHostData.push_back(0);
    offsetHostData.push_back(0);
  }

  for (int m = 0; m < groupNum; ++m) {
    for (int n = 0; n < rowLen; ++n) {
      float value2 = m * rowLen + n;
      smoothHostData.push_back(aclFloatToFloat16(value2));
    }
  }

  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建smooth aclTensor
  ret = CreateAclTensor(smoothHostData, smoothShape, &smoothDeviceAddr, aclDataType::ACL_FLOAT16, &smooth);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建group aclTensor
  ret = CreateAclTensor(groupHostData, groupShape, &groupDeviceAddr, aclDataType::ACL_INT32, &group);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建scale aclTensor
  ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
  ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  const char* quantMode = "pertoken";

  // 调用aclnnDynamicQuantV3第一段接口
  ret = aclnnDynamicQuantV3GetWorkspaceSize(x, smooth, group, 2, false, quantMode, y, scale, offset, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV3GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  // 调用aclnnDynamicQuantV3第二段接口
  ret = aclnnDynamicQuantV3(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDynamicQuantV3 failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  PrintOutResult(yShape, &yDeviceAddr);

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(x);
  aclDestroyTensor(smooth);
  aclDestroyTensor(y);
  aclDestroyTensor(scale);
  aclDestroyTensor(offset);

  // 7. 释放device资源
  aclrtFree(xDeviceAddr);
  aclrtFree(smoothDeviceAddr);
  aclrtFree(yDeviceAddr);
  aclrtFree(scaleDeviceAddr);
  aclrtFree(offsetDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
