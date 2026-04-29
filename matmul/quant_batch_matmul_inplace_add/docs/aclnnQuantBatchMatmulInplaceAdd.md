# aclnnQuantBatchMatmulInplaceAdd

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/quant_batch_matmul_inplace_add)

## 产品支持情况

| 产品                                                                            | 是否支持 |
| :------------------------------------------------------------------------------ | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                                          |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：在 micro-batch 训练场景，需要做 micro-batch 的梯度累计，会存在大量 QuantBatchMatmul 后接 InplaceAdd 的融合场景。QuantBatchMatmulInplaceAdd 算子将上述算子融合起来，提高网络性能。实现量化矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。

- 计算公式：

  - **mx 量化：**

    $$
    y[m,n] = \sum_{j=0}^{kLoops-1} ((\sum_{k=0}^{gsK-1} (x1Slice * x2Slice)) * (scale1[m, j] * scale2[j, n])) + y[m,n]
    $$

    其中，$gsK$ 代表 K 轴的量化的 block size 即 32，$x1Slice$代表$x1$第 m 行长度为 $gsK$ 的向量，$x2Slice$代表$x2$第 n 列长度为 $gsK$ 的向量，K 轴均从$j*gsK$起始切片，j 的取值范围[0, kLoops), kLoops=ceil($K_i$ / $gsK$)，支持最后的切片长度不足 $gsK$。

  - **HIFLOAT8 T-T 量化：**

    $$
    y[m,n] = (\sum_{k=0}^{K_i-1} (x1[m,k] * x2[k,n])) * (scale1 * scale2) + y[m,n]
    $$

    其中，$scale1$和$scale2$分别对应$x1Scale$和$x2Scale$，均为标量（shape 为 $(1)$）；当设置transposeX1/transposeX2时，公式中的$x1$和$x2$按转置后的视图参与计算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize”接口获取入参并根据计算流程计算所需 workspace 大小，再调用“aclnnQuantBatchMatmulInplaceAdd”接口执行计算。

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *x1Scale,
    const aclTensor *x2Scale,
    aclTensor       *yRef,
    bool             transposeX1,
    bool             transposeX2,
    int64_t          groupSize,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed;width: 1507px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 130px">
  <col style="width: 145px">
  </colgroup>
  <thead>
    <tr>
      <th>参数名</th>
      <th style="white-space: nowrap">输入/输出</th>
      <th>描述</th>
      <th>使用说明</th>
      <th>数据类型</th>
      <th>数据格式</th>
      <th style="white-space: nowrap">维度</th>
      <th>非连续的Tensor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x1（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入x1。</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2（aclTensor*）</td>
      <td>输入</td>
      <td>公式中的输入x2。</td>
      <td>-</td>
      <td>FLOAT8_E4M3FN、FLOAT8_E5M2、HIFLOAT8</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1Scale（aclTensor*）</td>
      <td>可选输入</td>
      <td>表示量化参数中的由x1量化引入的缩放因子。</td>
      <td>
        综合约束请参见<a href="#约束说明" target="_blank">约束说明。
      </td>
      <td>FLOAT8_E8M0 、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2Scale（aclTensor*）</td>
      <td>输入</td>
      <td>表示量化参数中的由x2量化引入的缩放因子。</td>
      <td>
        综合约束请参见<a href="#约束说明" target="_blank">约束说明。
      </td>
      <td>FLOAT8_E8M0 、FLOAT32 </td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yRef（aclTensor*）</td>
      <td>输入输出</td>
      <td>对应公式中的输入输出y。</td>
      <td>
      当输入x1为m=0的空tensor或x2为n=0的空tensor时，输出为空tensor。
      </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>transposeX1（bool）</td>
      <td>输入</td>
      <td>表示x1的输入shape是否转置</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>transposeX2（bool）</td>
      <td>输入</td>
      <td>表示x2的输入shape是否转置</td>
      <td>-</td>
      <td>bool</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      </tr>
    <tr>
      <td>groupSize（int64）</td>
      <td>输入</td>
      <td>整数型参数，用于输入m、n、k方向上的量化分组大小。</td>
      <td>
      由3个方向的groupSizeM，groupSizeN，groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式见表格下方。
      </td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t）</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor（aclOpExecutor*）</td>
      <td>输出</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - 计算公式：<a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1030px"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>如果传入参数是必选输入、输出或者必选属性，且是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>x1、x2、x1Scale、x2Scale、yRef、groupSize的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>x1、x2、x1Scale、x2Scale、yRef的shape不满足校验条件。</td>
    </tr>
    <tr>
      <td>x1、x2、x2Scale、yRef是空tensor。</td>
    </tr>
    <tr>
      <td>传入的groupSize不满足校验条件，或传入的groupSize为0时，x1、x2与x1Scale，x2Scale的shape关系无法推断groupSize。</td>
    </tr>
  </tbody></table>

## aclnnQuantBatchMatmulInplaceAdd

- **参数说明：**
  <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的AscendCL stream流。</td></tr>
    </tbody>
  </table>

- **返回值：**

  返回 aclnnStatus 状态码，具体参见[aclnn 返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnQuantBatchMatmulInplaceAdd 默认确定性实现。
- 当前仅支持 transposeX1=true 且 transposeX2=false。
- groupSize相关约束：
  - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据x1/x2/x1Scale/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。原理：假设groupSizeM=0，表示m方向量化分组值由接口推断，推断公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与x1 shape中的m一致，scaleM与x1Scale shape中的m一致。
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
- 动态量化（mx 量化）场景约束：
  - 输入和输出支持以下数据类型组合：

    | x1 | x2 | x1Scale | x2Scale | outRef |
    |:-------:|:-------:| :------- | :------ | :------ |
    |FLOAT8_E5M2/FLOAT8_E4M3FN |FLOAT8_E5M2/FLOAT8_E4M3FN| FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |

  - x1数据类型、x2数据类型、x1、x2、x1Scale、x2Scale和groupSize的取值关系：

      | x1数据类型 | x2数据类型 | x1 shape | x2 shape | x1Scale Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      |:-------:|:-------:| :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      |FLOAT8_E5M2/FLOAT8_E4M3FN |FLOAT8_E5M2/FLOAT8_E4M3FN| (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 32 |

- HIFLOAT8 T-T 场景约束：
  - 输入和输出支持以下数据类型组合：

    | x1 | x2 | x1Scale | x2Scale | outRef |
    |:-------:|:-------:| :------- | :------ | :------ |
    |HIFLOAT8 |HIFLOAT8| FLOAT32 | FLOAT32 | FLOAT32 |

  - x1数据类型、x2数据类型、x1、x2、x1Scale、x2Scale和groupSize的取值关系：

      | x1数据类型 | x2数据类型 | x1 shape | x2 shape | x1Scale Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      |:-------:|:-------:| :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      |HIFLOAT8 |HIFLOAT8| (k, m) | (k, n) | (1) | (1) | (m, n) | [0, 0, 0] | 0 |

## 调用示例

- <term>Ascend 950PR/Ascend 950DT</term>：mxFP8 量化场景示例代码如下，其中 `x1/x2` 为 mxFP8、`x1Scale/x2Scale` 为 FLOAT8_E8M0，`groupSize=32`。具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <vector>
  #include <memory>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

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

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
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
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
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
      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  template <typename T1, typename T2>
  auto CeilDiv(T1 a, T2 b) -> T1
  {
      if (b == 0) {
          return a;
      }
      return (a + b - 1) / b;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnQuantBatchMatmulInplaceAddTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t M = 8;
      int64_t K = 16;
      int64_t N = 8;

      std::vector<int64_t> x1Shape = {K, M};
      std::vector<int64_t> x2Shape = {K, N};
      std::vector<int64_t> x2ScaleShape = {CeilDiv(K, 64), N, 2};
      std::vector<int64_t> yInputShape = {M, N};
      std::vector<int64_t> x1ScaleShape = {CeilDiv(K, 64), M, 2};
      std::vector<int64_t> yOutShape = {M, N};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yInputDeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* yOutputDeviceAddr = nullptr;

      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yInput = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* yOutput = nullptr;

      std::vector<uint8_t> x1HostData(M * K, 1);                 // 0b00111000 为 fp8_e4m3fn的1.0
      std::vector<uint8_t> x2HostData(N * K, 1); // 0b0010为fp4_e2m1的1.0，这里用uint8代表2个fp4
      std::vector<uint8_t> x2ScaleHostData(CeilDiv(K, 64) * N * 2, 1);
      std::vector<float> yInputHostData(M * N, 1);                        // fp32的1.0
      std::vector<uint8_t> x1ScaleHostData(M * CeilDiv(K, 64) * 2, 1);
      std::vector<float> yOutputHostData(M * N, 1);

      // 创建x1 aclTensor
      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建x2 aclTensor
      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT8_E4M3FN, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建x2Scale aclTensor
      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建yInput aclTensor
      ret = CreateAclTensor(yInputHostData, yInputShape, &yInputDeviceAddr, aclDataType::ACL_FLOAT, &yInput);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yInputTensorPtr(yInput, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yInputDeviceAddrPtr(yInputDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 创建x1Scale aclTensor
      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT8_E8M0, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      bool transposeX1 = true;
      bool transposeX2 = false;
      int64_t groupSize = 32;

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      void* workspaceAddr = nullptr;

      // 调用aclnnQuantBatchMatmulInplaceAdd第一段接口
      ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(x1, x2, x1Scale, x2Scale, yInput, transposeX1, transposeX2, groupSize, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }
      // 调用aclnnTransQuantParamV2第二段接口
      ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yInputShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(uint32_t), yInputDeviceAddr,
                      size * sizeof(uint32_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantBatchMatmulInplaceAddTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
    ```

- <term>Ascend 950PR/Ascend 950DT</term>：HIFLOAT8 TT 场景示例代码如下，当前支持组合 `transposeX1=true`、`transposeX2=false`，且 `x1/x2` 为 HIFLOAT8、`x1Scale/x2Scale` 为 FLOAT32、`groupSize=0`。具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  /**
   * Copyright (c) 2026 Huawei Technologies Co., Ltd.
   * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
   * CANN Open Software License Agreement Version 2.0 (the "License").
   * Please refer to the License for details. You may not use this file except in compliance with the License.
   * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
   * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
   * See LICENSE in the root of the software repository for the full text of the License.
   */

  /*!
  * \file test_aclnn_quant_batch_matmul_inplace_add_TT.cpp
  * \brief
  */

  #include <iostream>
  #include <memory>
  #include <vector>

  #include "acl/acl.h"
  #include "aclnnop/aclnn_quant_batch_matmul_inplace_add.h"

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

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
      }
      return shapeSize;
  }

  int Init(int32_t deviceId, aclrtStream* stream)
  {
      auto ret = aclInit(nullptr);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
      ret = aclrtSetDevice(deviceId);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
      ret = aclrtCreateStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
      return 0;
  }

  template <typename T>
  int CreateAclTensor(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor)
  {
      auto size = GetShapeSize(shape) * sizeof(T);
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
          strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(
          shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
          *deviceAddr);
      return 0;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnQuantBatchMatmulInplaceAddTTDemo(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      int64_t m = 8;
      int64_t k = 16;
      int64_t n = 8;

      // 当前支持的调用组合为 transposeX1=true, transposeX2=false，
      // 因此 x1/x2 的 shape 需要分别构造成 (k, m) 和 (k, n)。
      std::vector<int64_t> x1Shape = {k, m};
      std::vector<int64_t> x2Shape = {k, n};
      std::vector<int64_t> x1ScaleShape = {1};
      std::vector<int64_t> x2ScaleShape = {1};
      std::vector<int64_t> yRefShape = {m, n};

      void* x1DeviceAddr = nullptr;
      void* x2DeviceAddr = nullptr;
      void* x1ScaleDeviceAddr = nullptr;
      void* x2ScaleDeviceAddr = nullptr;
      void* yRefDeviceAddr = nullptr;

      aclTensor* x1 = nullptr;
      aclTensor* x2 = nullptr;
      aclTensor* x1Scale = nullptr;
      aclTensor* x2Scale = nullptr;
      aclTensor* yRef = nullptr;

      std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape), 0b1000); // 0b1000 为 hifloat8 的 1.0
      std::vector<uint8_t> x2HostData(GetShapeSize(x2Shape), 0b1000);
      std::vector<float> x1ScaleHostData(GetShapeSize(x1ScaleShape), 1.0f);
      std::vector<float> x2ScaleHostData(GetShapeSize(x2ScaleShape), 1.0f);
      std::vector<float> yRefHostData(GetShapeSize(yRefShape), 1.0f);

      ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_HIFLOAT8, &x1);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1TensorPtr(x1, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1DeviceAddrPtr(x1DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_HIFLOAT8, &x2);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2TensorPtr(x2, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2DeviceAddrPtr(x2DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x1ScaleHostData, x1ScaleShape, &x1ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x1Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x1ScaleTensorPtr(x1Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x1ScaleDeviceAddrPtr(x1ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(x2ScaleHostData, x2ScaleShape, &x2ScaleDeviceAddr, aclDataType::ACL_FLOAT, &x2Scale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> x2ScaleTensorPtr(x2Scale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> x2ScaleDeviceAddrPtr(x2ScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      ret = CreateAclTensor(yRefHostData, yRefShape, &yRefDeviceAddr, aclDataType::ACL_FLOAT, &yRef);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yRefTensorPtr(yRef, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yRefDeviceAddrPtr(yRefDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      bool transposeX1 = true;
      bool transposeX2 = false;
      int64_t groupSize = 0;

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);

      ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
          x1, x2, x1Scale, x2Scale, yRef, transposeX1, transposeX2, groupSize, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS,
                LOG_PRINT("aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }

      ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      std::vector<float> resultData(GetShapeSize(yRefShape), 0.0f);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yRefDeviceAddr,
          resultData.size() * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);

      for (size_t i = 0; i < resultData.size(); ++i) {
          LOG_PRINT("result[%zu] is: %f\n", i, resultData[i]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnQuantBatchMatmulInplaceAddTTDemo(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTTDemo failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
  
