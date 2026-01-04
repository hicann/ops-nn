# aclnnAscendQuant

  ## 产品支持情况

  | 产品                                                         | 是否支持 |
  | :----------------------------------------------------------- | :------: |
  | <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
  | <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
  | <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

  ## 功能说明

  - 算子功能：对输入x进行量化操作，且scale和offset的size需要是x的最后一维或1。
  - 计算公式：
    - sqrtMode为false时，计算公式为:

      $$
      y = round((x * scale) + offset)
      $$

    - sqrtMode为true时，计算公式为:

      $$
      y = round((x * scale * scale) + offset)
      $$

  ## 函数原型

  每个算子分为[两段式接口](common/两段式接口.md)，必须先调用“aclnnAscendQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAscendQuant”接口执行计算。

  - `aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, bool sqrtMode, const char *roundMode, int32_t dstType, const aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor)`
  - `aclnnStatus aclnnAscendQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

  ## aclnnAscendQuantGetWorkspaceSize

  - **参数说明：**

    - x（aclTensor*，计算输入）：Device侧的aclTensor，对应公式中的`x`，需要做量化的输入。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如果dstType为3，Shape的最后一维需要能被8整除；如果dstType为29，Shape的最后一维需要能被2整除。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - scale（aclTensor*，计算输入）：Device侧的aclTensor，对应公式中的`scale`，量化中的scale值。scale支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。如果x的dtype不是FLOAT32，需要和x的dtype一致。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - offset（aclTensor*，计算输入）：可选参数，对应公式中的`offset`，Device侧的aclTensor，反向量化中的offset值。offset支持1维张量或多维张量（当shape为1维张量时，scale的第0维需要等于x的最后一维或等于1；当shape为多维张量时，scale的维度需要和x保持一致，最后一个维度的值需要和x保持一致，其他维度为1）。数据类型和shape需要与scale保持一致。支持[非连续的Tensor](common/非连续的Tensor.md)，[数据格式](common/数据格式.md)支持ND。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
    - sqrtMode（bool，计算输入）：host侧的bool，对应公式中的`sqrtMode`，指定scale参与计算的逻辑，数据类型支持BOOL。当取值为true时，公式为y = round((x * scale * scale) + offset)；当取值为false时，公式为y = round((x * scale) + offset)。
    - roundMode（char\*，计算输入）：host侧的string，指定cast到int8输出的转换方式，支持取值round/ceil/trunc/floor。
    - dstType（int32_t，计算输入）：host侧的int32_t，指定输出的数据类型，该属性数据类型支持INT。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持取值2、3、29，分别表示INT8、INT32、INT4。
    - y（aclTensor\*，计算输出）：Device侧的aclTensor。对应公式中的`y`，类型为INT32时，Shape的最后一维是x最后一维的1/8，其余维度和x一致；其他类型时，Shape与x一致。支持空Tensor，支持[非连续的Tensor](common/非连续的Tensor.md)。[数据格式](common/数据格式.md)支持ND。
      - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持INT8、INT32、INT4。
    - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
    - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

    ```
    第一段接口完成入参校验，出现以下场景时报错：
    161001 (ACLNN_ERR_PARAM_NULLPTR):  1. 传入的x、scale、y是空指针。
    161002 (ACLNN_ERR_PARAM_INVALID):  1. x、scale、offset、y的数据类型或数据格式不在支持的范围之内。
                                      2. x、scale、offset、y的shape不满足限制条件。
                                      3. roundMode不在有效取值范围。
                                      4. dstType不在有效取值范围。
                                      5. y的数据类型为INT4时，x的shape尾轴大小不是偶数。
                                      6. y的数据类型为INT32时，y的shape尾轴不是x的shape尾轴大小的8倍，或者x与y的shape的非尾轴的大小不一致。
    ```

  ## aclnnAscendQuant

  - **参数说明：**

    - workspace（void*，入参）：在Device侧申请的workspace内存地址。
    - workspaceSize（uint64_t，入参）：在Device侧申请的workspace大小，由第一段接口aclnnAscendQuantGetWorkspaceSize获取。
    - executor（aclOpExecutor*，入参）：op执行器，包含了算子计算流程。
    - stream（aclrtStream，入参）：指定执行任务的Stream。

  - **返回值：**

    aclnnStatus：返回状态码，具体参见[aclnn返回码](common/aclnn返回码.md)。

  ## 约束说明

  无。

  ## 调用示例

  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](common/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_ascend_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1. （固定写法）device/stream初始化, 参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> selfShape = {4, 2};
    std::vector<int64_t> scaleShape = {2};
    std::vector<int64_t> offsetShape = {2};
    std::vector<int64_t> outShape = {4, 2};
    void* selfDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* scale = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* out = nullptr;
    std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<float> scaleHostData = {1, 2};
    std::vector<float> offsetHostData = {1, 2};
    std::vector<int8_t> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
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

    const int32_t dstType = 2;
    bool sqrtMode = false;
    const char* roundMode = "round";

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnAscendQuant第一段接口
    ret = aclnnAscendQuantGetWorkspaceSize(self, scale, offset, sqrtMode, roundMode, dstType, out, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAscendQuant第二段接口
    ret = aclnnAscendQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAscendQuant failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改，查看resultData中数据
    auto size = GetShapeSize(outShape);
    std::vector<int8_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(scale);
    aclDestroyTensor(offset);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(outDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }