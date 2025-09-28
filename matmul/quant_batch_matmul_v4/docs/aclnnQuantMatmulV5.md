# aclnnQuantMatmulV5

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √    |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √    |

## 功能说明

- 算子功能：完成量化的矩阵乘计算。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：x1支持数据类型为INT8，x2支持数据类型为INT32。完成量化的矩阵乘计算，最小支持输入维度为1维，最大支持输入维度为2维。相似接口有aclnnMm（仅支持2维Tensor作为输入的矩阵乘）。

- 计算公式：
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
    - x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64，yOffset为FLOAT32，out为FLOAT16/BFLOAT16：
      $$
      out = ((x1 @ (x2*x2scale)) + yoffset) * x1scale
      $$


## 函数原型
每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnQuantMatmulV5GetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnQuantMatmulV5”接口执行计算。

- `aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *x1Scale, const aclTensor *x2Scale, const aclTensor *yScale, const aclTensor *x1Offset, const aclTensor *x2Offset, const aclTensor *yOffset, const aclTensor *bias, bool transposeX1, bool transposeX2, int64_t groupSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor)`

- `aclnnStatus aclnnQuantMatmulV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)`

## aclnnQuantMatmulV5GetWorkspaceSize

- **参数说明：**
  - x1（aclTensor*，计算输入）：公式中的输入x1，device侧的aclTensor，[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，shape支持2维，形状为（m, k）。数据类型支持INT8，仅支持k是256的倍数，transposeX1为false，不支持batch轴。

  - x2（aclTensor*，计算输入）：公式中的输入x2，device侧的aclTensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：shape支持2维，形状为（k, n）。数据类型支持INT32。[数据格式](../../../docs/context/数据格式.md)支持ND，不支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)。仅支持k是256的倍数，transposeX2为false，不支持batch轴。
  - x1Scale（aclTensor*，计算输入）：公式中的输入x1Scale，device侧的aclTensor。[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：shape支持2维，形状为（m，1）。数据类型支持FLOAT32。
  - x2Scale（aclTensor*，计算输入）：表示量化参数，公式中的输入x2Scale，device侧的aclTensor。[数据格式](../../../docs/context/数据格式.md)支持ND。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：shape支持2维，形状为（k / groupSize，n）其中n与x2的n一致。数据类型支持UINT64。当原始输入类型不满足[约束说明](#约束说明)中类型组合时，由于TransQuantParamV2只支持1维，需要将x2_scale view成一维(k / groupSize * n)，再调用TransQuantParamV2算子的aclnn接口来将x2Scale转成UINT64数据类型，再将输出view成二维（k / groupSize, n）。
  - yScale（aclTensor*，计算输入）：输出y的反量化scale参数。不支持非连续的Tensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前版本不支持，需要传入nullptr。

  - x1Offset（aclTensor*，计算输入）：预留参数，当前版本不支持，需要传入nullptr。
  - x2Offset（aclTensor*，计算输入）：公式中的输入offset，device侧的aclTensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前版本不支持，需要传入nullptr。
  - yOffset（aclTensor*，计算输入）：
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：公式中的输入yOffset，device侧的aclTensor。[数据格式](../../../docs/context/数据格式.md)支持ND。shape支持1维（n）。数据类型支持FLOAT32。值要求为8\*x2\*x2Scale。
  - bias（aclTensor*，计算输入）：公式中的输入bias，device侧的aclTensor。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：当前版本不支持，需要传入nullptr。
  - transposeX1（bool，计算输入）：表示x1的输入shape是否包含transpose。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：transposeX1仅支持false，各个维度表示：（m, k）。
  - transposeX2（bool，计算输入）：表示x2的输入shape是否包含transpose。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：transposeX2仅支持false，各个维度表示：（k, n），其中k与x1的shape中的k一致。
  - groupSize（int64_t，计算输入）：用于输入m、n、k方向上的量化分组大小。groupSize输入由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式为：groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32。
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：[groupSizeM，groupSizeN，groupSizeK]取值组合仅支持仅支持[0，0,256]，即groupSize仅支持256。
  - out（aclTensor*，计算输出）：公式中的输出out，device侧的aclTensor。[数据格式](../../../docs/context/数据格式.md)支持ND。支持[非连续的Tensor](../../../docs/context/非连续的Tensor.md)，
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：shape支持2维，（m，n）。数据类型支持FLOAT16、BFLOAT16。
  - workspaceSize（uint64_t*，出参）：返回需要在Device侧申请的workspace大小。
  - executor（aclOpExecutor**，出参）：返回op执行器，包含了算子计算流程。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

  ```
  第一段接口完成入参校验，出现以下场景时报错：
  - 161001(ACLNN_ERR_PARAM_NULLPTR)：
  1. 传入的x1、x2、x1Scale、x2Scale、yOffset或out是空指针。
  - 161002(ACLNN_ERR_PARAM_INVALID)：
  1. x1、x2、bias、x1Scale、x2Scale、x2Offset或out的数据类型和数据格式不在支持的范围之内。
  2. x1、x2、bias、x1Scale、x2Scale、x2Offset或out的shape不满足校验条件。
  3. x1、x2、bias、x2Scale、x2Offset或out是空tensor。
  4. x1与x2的最后一维大小超过65535，x1的最后一维指transposeX1为true时的m或transposeX1为false时的k，x2的最后一维指transposeX2为true时的k或transposeX2为false时的n。
  5. 传入的groupSize不满足校验条件，或传入的groupSize为0时， x1、x2与x1Scale，x2Scale的shape关系无法推断groupSize。
  ```

## aclnnQuantMatmulV5

- **参数说明：**
  - workspace(void*, 入参)：在Device侧申请的workspace内存地址。
  - workspaceSize(uint64_t, 入参)：在Device侧申请的workspace大小，由第一段接口aclnnQuantMatmulV5GetWorkspaceSize获取。
  - executor(aclOpExecutor*, 入参)：op执行器，包含了算子计算流程。
  - stream(aclrtStream, 入参)：指定执行任务的Stream。

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明
输入和输出支持以下数据类型组合：
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  | x1                        | x2                        | x1Scale     | x2Scale     | x2Offset | yScale | bias    | yOffset    | out                                    |
  | ------------------------- | ------------------------- | ----------- | ----------- | -------- | -------| ------- | -------| -------------------------------------- |
  | INT8                      | INT32                     | FLOAT32     | UINT64            | null     | null     | null         | FLOAT32         | FLOAT16/BFLOAT16                       |

不同的[量化模式](../../../docs/context/量化介绍.md)支持的x1、 x2、x1Scale和x2Scale的输入dtype组合以及支持的平台为：
- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：

  x1、x2、x1Scale、x2Scale、yOffset和groupSize在不同量化场景下dtype、shape的取值等方面相互影响，关系如下：
    | x1数据类型                 | x2数据类型                 | x1Scale数据类型| x2Scale数据类型| x1 shape | x2 shape| x1Scale shape| x2Scale shape| yOffset shape| [groupSizeM，groupSizeN，groupSizeK]取值|
    | ------------------------- | ------------------------- | -------------- | ------------- | -------- | ------- | ------------ | ------------ | ------------ | ------------ |
    | INT8                    |INT32                   |FLOAT32              |UINT64             | （m, k）|（k, n // 8）|（m, 1）|（（k // 256），n）| (n) | [0, 0, 256]|

## 调用示例

- <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
x1为INT8，x2为INT32，x1Scale为FLOAT32，x2Scale为UINT64的示例代码如下，仅供参考，具体编译和执行过程请参考编译与运行样例。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_matmul_v5.h"

  #define CHECK_RET(cond, return_expr) \
      do {                             \
          if (!(cond)) {               \
              return_expr;             \
          }                            \
      } while (0)

  #define CHECK_FREE_RET(cond, return_expr) \
      do {                                  \
          if (!(cond)) {                    \
              Finalize(deviceId, stream);   \
              return_expr;                  \
          }                                 \
      } while (0)

  #define LOG_PRINT(message, ...)         \
      do {                                \
          printf(message, ##__VA_ARGS__); \
      } while (0)

  int64_t GetShapeSize(const std::vector<int64_t> &shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream *stream)
  {
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
  int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                      aclDataType dataType, aclTensor **tensor)
  {
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

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int aclnnQuantMatmulV5Test(int32_t deviceId, aclrtStream &stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      std::vector<int64_t> x1Shape = {1, 8192};     // (m,k)
      std::vector<int64_t> x2Shape = {8192, 128};  // (k,n)
      std::vector<int64_t> yoffsetShape = {1024};

      std::vector<int64_t> x1ScaleShape = {1,1};
      std::vector<int64_t> x2ScaleShape = {32, 1024}; // x2ScaleShape = [KShape / groupsize, N]
      std::vector<int64_t> outShape = {1, 1024};

      void *x1DeviceAddr = nullptr;
      void *x2DeviceAddr = nullptr;
      void *x2ScaleDeviceAddr = nullptr;
      void *x1ScaleDeviceAddr = nullptr;
      void *yoffsetDeviceAddr = nullptr;
      void *outDeviceAddr = nullptr;
      aclTensor *x1 = nullptr;
      aclTensor *x2 = nullptr;
      aclTensor *yoffset = nullptr;
      aclTensor *x2Scale = nullptr;
      aclTensor *x2Offset = nullptr;
      aclTensor *x1Scale = nullptr;
      aclTensor *out = nullptr;
      std::vector<int8_t> x1HostData(GetShapeSize(x1Shape), 1);
      std::vector<int32_t> x2HostData(GetShapeSize(x2Shape), 1);
      std::vector<int32_t> yoffsetHostData(GetShapeSize(yoffsetShape), 1);
      std::vector<int32_t> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1);
      float tmp = 1;
      uint64_t ans = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
      std::vector<int64_t> x2ScaleHostData(GetShapeSize(x2ScaleShape), ans);
      std::vector<uint16_t> outHostData(GetShapeSize(outShape), 1);  // 实际上是float16半精度方式

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_INT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_INT32, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_UINT64, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> scaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yoffset aclTensor
      ret = CreateAclTensor(yoffsetHostData, yoffsetShape, &yoffsetDeviceAddr, aclDataType::ACL_FLOAT, &yoffset);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> yoffsetTensorPtr(yoffset, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> yoffsetDeviceAddrPtr(yoffsetDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建out aclTensor
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT16, &out);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor *)> outTensorPtr(out, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void *)> outDeviceAddrPtr(outDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      bool transposeX1 = false;
      bool transposeX2 = false;
      int64_t groupSize = 256;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor *executor = nullptr;

      ret = aclnnQuantMatmulV5GetWorkspaceSize(x1, x2, x1Scale, x2Scale, nullptr, nullptr, nullptr, yoffset, nullptr,
                                              transposeX1, transposeX2, groupSize, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void *workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void *)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnQuantMatmulV5第二段接口
      ret = aclnnQuantMatmulV5(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5 failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(outShape);
      std::vector<uint16_t> resultData(size, 0);  // C语言中无法直接打印fp16的数据，需要用uint16读出来，自行通过二进制转成fp16
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
              return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %u\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 1;
      aclrtStream stream;
      auto ret = aclnnQuantMatmulV5Test(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantMatmulV5Test failed. ERROR: %d\n", ret); return ret);
      Finalize(deviceId, stream);
      return 0;
  }
  ```