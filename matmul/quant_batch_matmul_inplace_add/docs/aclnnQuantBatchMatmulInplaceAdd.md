# aclnnQuantBatchMatmulInplaceAdd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：在micro-batch训练场景，需要做micro-batch的梯度累积，会存在大量QuantBatchMatmul后接InplaceAdd的融合场景。QuantBatchMatmulInplaceAdd算子将上述算子融合起来，提高网络性能。实现量化矩阵乘计算和加法计算，基本功能为矩阵乘和加法的组合。

- 计算公式：

  $X_1$、$X_2$表示应用转置属性后的矩阵，shape分别为$(M,K)$、$(K,N)$。$Y_{\mathrm{in}}$为调用前的`yRef`，$Y_{\mathrm{out}}$为原地更新后的结果；两者的shape均为$(M,N)$。

  - MX（Microscaling，微缩放）量化：

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+\sum_{j=0}^{L-1}
    \left(\sum_{t=0}^{g_j-1}X_1[m,jg_K+t]X_2[jg_K+t,n]\right)S_1[m,j]S_2[j,n]
    $$

    其中，$g_K=32$，$L=\lceil K/g_K\rceil$，$g_j=\min(g_K,K-jg_K)$。$S_1$、$S_2$分别为`x1ScaleOptional`、`x2Scale`中对应分组的scale，存储shape见[约束说明](#约束说明)。

  - HIFLOAT8 T-T量化：

    $$
    Y_{\mathrm{out}}[m,n]=Y_{\mathrm{in}}[m,n]+
    \left(\sum_{k=0}^{K-1}X_1[m,k]X_2[k,n]\right)s_1s_2
    $$

    $s_1$、$s_2$分别对应`x1ScaleOptional`、`x2Scale`，shape均为$(1,)$。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnQuantBatchMatmulInplaceAdd”接口执行计算。

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
  const aclTensor* x1,
  const aclTensor* x2,
  const aclTensor* x1ScaleOptional,
  const aclTensor* x2Scale,
  aclTensor*       yRef,
  bool             transposeX1,
  bool             transposeX2,
  int64_t          groupSize,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```cpp
aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize

- **参数说明**
  <table style="table-layout: fixed; width: 1500px">
    <colgroup>
      <col style="width: 210px">
      <col style="width: 100px">
      <col style="width: 240px">
      <col style="width: 390px">
      <col style="width: 230px">
      <col style="width: 120px">
      <col style="width: 110px">
      <col style="width: 100px">
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
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>x1（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的输入x1。</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2（const aclTensor*）</td>
        <td>输入</td>
        <td>公式中的输入x2。</td>
        <td>-</td>
        <td>FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT4_E2M1、HIFLOAT8</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x1ScaleOptional（const aclTensor*）</td>
        <td>输入</td>
        <td>表示量化参数中的由x1量化引入的缩放因子。</td>
        <td>综合约束请参见<a href="#约束说明" target="_blank">约束说明</a>。<br/>当前支持的MX和HIFLOAT8 T-T场景均必须提供此参数，不能传入nullptr。</td>
        <td>FLOAT8_E8M0、FLOAT32</td>
        <td>ND</td>
        <td>3、1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2Scale（const aclTensor*）</td>
        <td>输入</td>
        <td>表示量化参数中的由x2量化引入的缩放因子。</td>
        <td>综合约束请参见<a href="#约束说明" target="_blank">约束说明</a>。</td>
        <td>FLOAT8_E8M0、FLOAT32</td>
        <td>ND</td>
        <td>3、1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>yRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>累加输入及原地更新后的输出，对应公式中的Y_in和Y_out。</td>
        <td>当输入x1为m=0的空tensor或x2为n=0的空tensor时，输出为空tensor。<br/>须提供连续Tensor。</td>
        <td>FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>×</td>
      </tr>
      <tr>
        <td>transposeX1（bool）</td>
        <td>输入</td>
        <td>表示x1的输入shape是否转置</td>
        <td>当前仅支持true。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>transposeX2（bool）</td>
        <td>输入</td>
        <td>表示x2的输入shape是否转置</td>
        <td>当前仅支持false。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>groupSize（int64_t）</td>
        <td>输入</td>
        <td>整数型参数，用于输入m、n、k方向上的量化分组大小。</td>
        <td>由3个方向的groupSizeM、groupSizeN、groupSizeK三个值拼接组成，每个值占16位，共占用int64_t类型groupSize的低48位（groupSize中的高16位的数值无效），计算公式见表格下方。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小，单位为字节。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor**）</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
    </tbody>
  </table>

  - 计算公式：<a name='f1'></a>

    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 300px">
      <col style="width: 150px">
      <col style="width: 550px">
    </colgroup>
    <thead>
      <tr>
        <th>返回码/状态码</th>
        <th>码值</th>
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
        <td>ACLNN_ERR_PARAM_INVALID</td>
        <td>161002</td>
        <td><ul><li>x1、x2、x1ScaleOptional、x2Scale、yRef的数据类型和数据格式不在支持的范围内，或groupSize的取值不在支持的范围内。</li><li>x1、x2、x1ScaleOptional、x2Scale、yRef的shape不满足校验条件。</li><li>x1、x2的K维度为0。</li><li>x1、x2数据类型为FLOAT4_E2M1时，M或N维度为奇数；或M、N、K均大于0时，M × K ≤ 2或N × K ≤ 2。</li><li>传入的groupSize不满足校验条件，或传入的groupSize为0时，x1、x2与x1ScaleOptional，x2Scale的shape关系无法推断groupSize。</li></ul></td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_CREATE_EXECUTOR</td>
        <td>561101</td>
        <td>创建算子执行器失败。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_INNER_NULLPTR</td>
        <td>561103</td>
        <td>内部Tensor创建、预处理、算子计算或结果拷贝失败，返回空指针。</td>
      </tr>
    </tbody>
  </table>

## aclnnQuantBatchMatmulInplaceAdd

- **参数说明**
  <table style="table-layout: fixed; width: 1000px">
    <colgroup>
      <col style="width: 220px">
      <col style="width: 100px">
      <col style="width: 680px">
    </colgroup>
    <thead>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>workspace（void*）</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t）</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，单位为字节，由第一段接口aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize获取。</td>
      </tr>
      <tr>
        <td>executor（aclOpExecutor*）</td>
        <td>输入</td>
        <td>op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
        <td>stream（aclrtStream）</td>
        <td>输入</td>
        <td>指定执行任务的AscendCL stream流。</td>
      </tr>
    </tbody>
  </table>

- **返回值**

  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  执行器运行失败时返回ACLNN_ERR_INNER（561000）。

## 约束说明

- 确定性说明：aclnnQuantBatchMatmulInplaceAdd默认确定性实现。
- 当前仅支持transposeX1=true且transposeX2=false。M、N须为非负整数，K须为正整数。
- groupSize相关约束：
  - 传入的groupSize内部会按如下公式分解得到groupSizeM、groupSizeN、groupSizeK，当其中有1个或多个为0，会根据x1/x2/x1ScaleOptional/x2Scale输入shape重新设置groupSizeM、groupSizeN、groupSizeK用于计算。原理：假设groupSizeM=0，表示m方向量化分组值由接口推断，推断公式为groupSizeM = m / scaleM（需保证m能被scaleM整除），其中m与x1 shape中的m一致，scaleM与x1ScaleOptional shape中的m一致。
    $$
    groupSize = groupSizeK | groupSizeN << 16 | groupSizeM << 32
    $$
- 动态量化（MX量化）场景约束：
  - 输入和输出支持以下数据类型组合：

    | x1 | x2 | x1ScaleOptional | x2Scale | yRef |
    | :-------: | :-------: | :------- | :------ | :------ |
    | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |
    | FLOAT4_E2M1 | FLOAT4_E2M1 | FLOAT8_E8M0 | FLOAT8_E8M0 | FLOAT32 |

  - x1数据类型、x2数据类型、x1、x2、x1ScaleOptional、x2Scale和groupSize的取值关系：

      | x1数据类型 | x2数据类型 | x1 shape | x2 shape | x1ScaleOptional Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      | :-------: | :-------: | :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      | FLOAT8_E5M2/FLOAT8_E4M3FN | FLOAT8_E5M2/FLOAT8_E4M3FN | (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 4295032864 |
      | FLOAT4_E2M1 | FLOAT4_E2M1 | (k, m) | (k, n) | (ceil(k / 64), m, 2) | (ceil(k / 64), n, 2) | (m, n) | [1, 1, 32] | 4295032864 |

  - 注：上表中gsM、gsK和gsN分别表示groupSizeM、groupSizeK和groupSizeN。gsM、gsK和gsN为0的维度会自动推导，上表中是不用自动推导的情况。
  - FLOAT4_E2M1场景额外约束：
    - x1的M维度、x2的N维度必须为偶数。
    - 当M、N、K均大于0时，必须同时满足M × K > 2、N × K > 2。

- HIFLOAT8 T-T场景约束：
  - 输入和输出支持以下数据类型组合：

    | x1 | x2 | x1ScaleOptional | x2Scale | yRef |
    | :-------: | :-------: | :------- | :------ | :------ |
    | HIFLOAT8 | HIFLOAT8 | FLOAT32 | FLOAT32 | FLOAT32 |

  - x1数据类型、x2数据类型、x1、x2、x1ScaleOptional、x2Scale和groupSize的取值关系：

      | x1数据类型 | x2数据类型 | x1 shape | x2 shape | x1ScaleOptional Shape | x2Scale Shape | yRef Shape | [gsM, gsN, gsK] | groupSize |
      | :-------: | :-------: | :------- | :------ | :------ | :------ | :------ | :------ | :------ |
      | HIFLOAT8 | HIFLOAT8 | (k, m) | (k, n) | (1) | (1) | (m, n) | [0, 0, 0] | 0 |

## 调用示例

<!-- npu="950" id7 -->
- <term>Ascend 950PR&950DT系列产品</term>：mxFP8量化场景示例代码如下，其中`x1/x2`为mxFP8、`x1Scale/x2Scale`为FLOAT8_E8M0，`groupSize=32`。具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```cpp
  #include <cstdint>
  #include <cstdio>
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
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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

      std::vector<uint8_t> x1HostData(M * K, 1); // 示例填充值
      std::vector<uint8_t> x2HostData(N * K, 1); // 示例填充值
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

      // 3. 调用CANN算子库API
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
      // 调用aclnnQuantBatchMatmulInplaceAdd第二段接口
      ret = aclnnQuantBatchMatmulInplaceAdd(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnQuantBatchMatmulInplaceAdd failed. ERROR: %d\n", ret); return ret);

      // 4.（固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧
      auto size = GetShapeSize(yInputShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), size * sizeof(resultData[0]), yInputDeviceAddr,
                       size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t j = 0; j < size; j++) {
          LOG_PRINT("result[%ld] is: %f\n", j, resultData[j]);
      }
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1.（固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream = nullptr;
      auto ret = AclnnQuantBatchMatmulInplaceAddTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTest failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR&950DT系列产品</term>：HIFLOAT8 TT场景示例代码如下，当前支持组合`transposeX1=true`、`transposeX2=false`，且`x1/x2`为HIFLOAT8、`x1ScaleOptional/x2Scale`为FLOAT32、`groupSize=0`。具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

  ```cpp
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

  #include <cstdint>
  #include <cstdio>
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
      CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed.\n"); return ACL_ERROR_INVALID_PARAM);
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

      // 当前支持的调用组合为transposeX1=true, transposeX2=false，
      // 因此x1/x2的shape需要分别构造成(k, m)和(k, n)。
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

      std::vector<uint8_t> x1HostData(GetShapeSize(x1Shape), 0b1000); // 0b1000为hifloat8的1.0
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
      aclrtStream stream = nullptr;
      auto ret = AclnnQuantBatchMatmulInplaceAddTTDemo(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS,
                    LOG_PRINT("AclnnQuantBatchMatmulInplaceAddTTDemo failed. ERROR: %d\n", ret); return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```

<!-- end id8 -->
