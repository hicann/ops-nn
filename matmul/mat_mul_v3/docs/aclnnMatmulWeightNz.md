# aclnnMatmulWeightNz

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |

## 功能说明

- 算子功能：完成张量self与张量mat2的矩阵乘计算，mat2仅支持昇腾亲和数据排布格式，只支持self为2维, mat2为4维。
  相似接口有aclnnMatmul(mat2仅支持ND) aclnnMm（支持2维Tensor作为输入的矩阵乘）和aclnnBatchMatmul（仅支持3维的矩阵乘，其中第1维为batch）。
- 计算公式：

  $$
  result=self @ mat2
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnMatmulWeightNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMatmulWeightNz”接口执行计算。

- `aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnMatmulWeightNzGetWorkspaceSize
- **参数说明：**

  - self（aclTensor*，计算输入）：表示矩阵乘的第一个矩阵，公式中的self，Device侧aclTensor。数据类型需要与mat2满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/context/数据格式.md)支持ND。shape维度只支持2维，支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16。
    - 在self不转置的情况下各个维度表示：（m，k）。
    - 在self转置的情况下各个维度表示：（k，m）。
  - mat2（aclTensor*，计算输入）：表示矩阵乘的第二个矩阵，公式中的mat2，Device侧的aclTensor，数据类型需要与self满足数据类型推导规则（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/context/数据格式.md)只支持昇腾亲和数据排布格式(NZ)，shape维度支持4维。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16。
    - 当B矩阵不转置时， 昇腾亲和数据排布格式各个维度表示：（n1，k1，k0，n0），其中k0 = 16， n0为16。self shape中的k和mat2 shape中的k1需要满足以下关系：ceil（k，k0） = k1, mat2 shape中的n1与out的n满足以下关系: ceil(n, n0) = n1。
    - 当B矩阵转置时， 昇腾亲和数据排布格式各个维度表示：（k1，n1，n0，k0），其中n0 = 16， k0为16。self shape中的k和mat2 shape中的k1需要满足以下关系：ceil（k，k0） = k1, mat2 shape中的n1与out的n满足以下关系: ceil(n, n0) = n1。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：调用此接口之前，必须使用aclnnTransMatmulWeight接口完成mat2的原始输入Format从ND到昇腾亲和数据排布格式的转换。
  - out（aclTensor*，计算输出）：表示矩阵乘的输出矩阵，公式中的out，Device侧aclTensor。数据类型需要与self与mat2推导之后的数据类型保持一致（参见[互推导关系](../../../docs/context/互推导关系.md)和[约束说明](#约束说明)）。[数据格式](../../../docs/context/数据格式.md)支持ND。shape维度只支持2维。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：数据类型支持BFLOAT16、FLOAT16。
    - 各个维度表示：（m，n），m与self的m一致，n与mat2的n1以及n0满足ceil(n / n0) = n1的关系。
  - cubeMathType（int8_t，计算输入）：用于指定Cube单元的计算逻辑，Host侧的整型。数据类型支持INT8。注意：如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：
    * 0：KEEP_DTYPE，保持输入的数据类型进行计算。
    * 1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持该选项。
    * 2：USE_FP16，支持将输入降精度至FLOAT16计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当输入数据类型为BFLOAT16时不支持该选项。
    * 3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。
      * <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持该选项。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  161001(ACLNN_ERR_PARAM_NULLPTR): 1. 传入的self、mat2或out是空指针。
  161002(ACLNN_ERR_PARAM_INVALID): 1. self和mat2的数据类型和数据格式不在支持的范围之内。
                                   2. self和mat2无法做数据类型推导。
                                   3. 推导出的数据类型无法转换为指定输出out的类型。
  ```

## aclnnMatmulWeightNz

- **参数说明：**

  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnMatmulWeightNzGetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
- 不支持两个输入分别为BFLOAT16和FLOAT16的数据类型推导。
- self只支持2维, mat2只支持昇腾亲和数据排布格式(NZ)，调用此接口之前，必须完成mat2从ND到昇腾亲和数据排布格式的转换。
- 当mat2任意一个维度为1，且mat2为非连续的NZ格式时, 不保证精度和功能, 即不支持k=1或者n=1时, mat2先转NZ后再对tensor的shape做任何操作处理, 如transpose操作。

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  self和mat2数据类型为float16，mat2为昇腾亲和数据排布格式场景下的示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。
  ```Cpp
  #include <iostream>
  #include <vector>
  #include <cmath>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_matmul.h"
  #include "aclnnop/aclnn_trans_matmul_weight.h"
  #include "aclnnop/aclnn_cast.h"

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

  // 将FP16的uint16_t表示转换为float表示
  float Fp16ToFloat(uint16_t h) {
    int s = (h >> 15) & 0x1;              // sign
    int e = (h >> 10) & 0x1F;             // exponent
    int f =  h        & 0x3FF;            // fraction
    if (e == 0) {
      // Zero or Denormal
      if (f == 0) {
        return s ? -0.0f : 0.0f;
      }
      // Denormals
      float sig = f / 1024.0f;
      float result = sig * pow(2, -24);
      return s ? -result : result;
    } else if (e == 31) {
        // Infinity or NaN
        return f == 0 ? (s ? -INFINITY : INFINITY) : NAN;
    }
    // Normalized
    float result = (1.0f + f / 1024.0f) * pow(2, e - 15);
    return s ? -result : result;
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

  template <typename T>
  int CreateAclTensorWeight(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                            aclDataType dataType, aclTensor** tensor) {
    auto size = static_cast<uint64_t>(GetShapeSize(shape));

    const aclIntArray* mat2Size = aclCreateIntArray(shape.data(), shape.size());
    auto ret = aclnnCalculateMatmulWeightSize(mat2Size, &size);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCalculateMatmulWeightSize failed. ERROR: %d\n", ret); return ret);
    size *= sizeof(T);

    // 调用aclrtMalloc申请device侧内存
    ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    // 调用aclrtMemcpy将host侧数据拷贝到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    std::vector<int64_t> storageShape;
    storageShape.push_back(GetShapeSize(shape));

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              storageShape.data(), storageShape.size(), *deviceAddr);
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
    std::vector<int64_t> selfShape = {16, 32};
    std::vector<int64_t> mat2Shape = {32, 16};
    std::vector<int64_t> outShape = {16, 16};
    void* selfDeviceAddr = nullptr;
    void* mat2DeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    aclTensor* self = nullptr;
    aclTensor* mat2 = nullptr;
    aclTensor* out = nullptr;
    std::vector<uint16_t> selfHostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1
    std::vector<uint16_t> mat2HostData(512, 0x3C00); // float16_t 用0x3C00表示int_16的1
    std::vector<uint16_t> outHostData(256, 0);
    // 创建self aclTensor
    ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT16, &self);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建other aclTensor
    ret = CreateAclTensorWeight(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT16, &mat2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    int8_t cubeMathType = 1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用TransWeight
    ret = aclnnTransMatmulWeightGetWorkspaceSize(mat2, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeightGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnTransMatmulWeight第二段接口
    ret = aclnnTransMatmulWeight(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTransMatmulWeight failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnMatmulWeightNz第一段接口
    uint64_t workspaceSizeMm = 0;
    ret = aclnnMatmulWeightNzGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSizeMm, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddrMm = nullptr;
    if (workspaceSizeMm > 0) {
      ret = aclrtMalloc(&workspaceAddrMm, workspaceSizeMm, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnMatmulWeightNz第二段接口
    ret = aclnnMatmulWeightNz(workspaceAddrMm, workspaceSizeMm, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMatmulWeightNz failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<uint16_t> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成float表示的fp16
    for (int64_t i = 0; i < size; i++) {
      float fp16Float = Fp16ToFloat(resultData[i]);
      LOG_PRINT("result[%ld] is: %f\n", i, fp16Float);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(self);
    aclDestroyTensor(mat2);
    aclDestroyTensor(out);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(selfDeviceAddr);
    aclrtFree(mat2DeviceAddr);
    aclrtFree(outDeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    if (workspaceSizeMm > 0) {
      aclrtFree(workspaceAddrMm);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
  }
  ```
