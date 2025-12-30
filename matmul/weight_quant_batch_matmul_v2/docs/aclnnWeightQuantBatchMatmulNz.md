# aclnnWeightQuantBatchMatmulNz

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √    |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |

## 功能说明

- **接口功能**：完成一个输入为伪量化场景的矩阵乘计算。此接口仅支持矩阵乘的右输入矩阵为FRACTAL_NZ格式。
- **计算公式**：

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

  公式中的$weight$为伪量化场景的输入，其反量化公式$ANTIQUANT(weight)$为

  $$
  ANTIQUANT(weight) = (weight + antiquantOffset) * antiquantScale
  $$

  当需要对输出进行量化处理时，其量化公式为

  $$
  \begin{aligned}
  y &= QUANT(x @ ANTIQUANT(weight) + bias) \\
  &= (x @ ANTIQUANT(weight) + bias) * quantScale + quantOffset \\
  \end{aligned}
  $$

  当不需要对输出再进行量化操作时，其计算公式为

  $$
  y = x @ ANTIQUANT(weight) + bias
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnWeightQuantBatchMatmulNzGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnWeightQuantBatchMatmulNz”接口执行计算。

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
  const aclTensor *x,
  const aclTensor *weight,
  const aclTensor *antiquantScale,
  const aclTensor *antiquantOffsetOptional,
  const aclTensor *quantScaleOptional,
  const aclTensor *quantOffsetOptional,
  const aclTensor *biasOptional,
  int              antiquantGroupSize,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnWeightQuantBatchMatmulNz(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnWeightQuantBatchMatmulNzGetWorkspaceSize

- **参数说明**

  <table style="table-layout: fixed; width: 1550px">
    <colgroup>
      <col style="width: 180px">
      <col style="width: 120px">
      <col style="width: 300px">
      <col style="width: 330px">
      <col style="width: 212px">
      <col style="width: 100px">
      <col style="width: 190px">
      <col style="width: 145px">
    </colgroup>
    <thread>
      <tr>
        <th>参数名</th>
        <th style="white-space: nowrap">输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th><a href="../../../docs/zh/context/数据格式.md" target="_blank">数据格式</a></th>
        <th style="white-space: nowrap">维度(shape)</th>
        <th><a href="../../../docs/zh/context/非连续的Tensor.md" target="_blank">非连续的Tensor</a></th>
      </tr>
    </thread>
    <tbody>
      <tr>
        <td>x</td>
        <td>输入</td>
        <td>矩阵乘的左输入矩阵，公式中的输入x，device侧的aclTensor。</td>
        <td>不支持转置场景。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>维度支持2维，shape支持(m, k)，m表示矩阵第1维的大小，k表示矩阵的第2维的大小，其中Reduce维度k需要与weight的Reduce维度k大小相等。m大小在[1, 2^31-1]范围内。</td>
        <td>×</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>矩阵乘的右输入矩阵，公式中的输入weight，device侧的aclTensor。</td>
        <td>非连续的Tensor仅支持转置场景。
        若weight数据类型为INT32/FLOAT时，表示紧密排布的INT4/FLOAT4_E2M1，要求INT4/FLOAT4_E2M1尾轴8对齐。
        若数据类型为INT4或FLOAT4_E2M1，则weight的内轴应为偶数。
        </td>
        <td>INT4、FLOAT4_E2M1、INT32、FLOAT</td>
        <td>FRACTAL_NZ</td>
        <td>维度支持2维，Reduce维度k需要与x的Reduce维度k大小相等。支持(k, n)，其中k表示矩阵第1维的大小，n表示矩阵第2维的大小。k、n大小在[1, 65535]范围内。k, n要求32B对齐。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>antiquantScale</td>
        <td>输入</td>
        <td>实现输入反量化计算的反量化scale参数，反量化公式中的输入antiquantScale。</td>
        <td>连续性要求和weight保持一致。</td>
        <td>FLOAT16、BFLOAT16或FLOAT8_E8M0</td>
        <td>ND</td>
        <td>维度支持1维或2维。shape支持(k/antiquantGroupSize, n)、(1, n)或(n,)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>antiquantOffsetOptional</td>
        <td>输入</td>
        <td>实现输入反量化计算的反量化offset参数，反量化公式中的antiquantOffset，device侧的aclTensor。</td>
        <td>可选输入, 连续性要求和weight保持一致。当x是FLOAT16或者BFLOAT16，同时weight是FLOAT、FLOAT4_E2M1时，不支持antiquantOffsetOptional参数，填空指针。</td>
        <td>FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>shape要求与antiquantScale一致。</td>
        <td>√</td>
      </tr>
      <tr>
        <td>quantScaleOptional</td>
        <td>输入</td>
        <td>预留参数，暂未使用，固定传入空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>quantOffsetOptional</td>
        <td>输入</td>
        <td>预留参数，暂未使用，固定传入空指针。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>biasOptional</td>
        <td>输入</td>
        <td>偏置输入，公式中的bias，device侧的aclTensor。</td>
        <td>可选输入, 当不需要时传空指针。</td>
        <td>BFLOAT16、FLOAT16、FLOAT</td>
        <td>ND</td>
        <td>维度支持1维或2维，shape支持(n,)或(1, n)。</td>
        <td>×</td>
      </tr>
      <tr>
        <td>antiquantGroupSize</td>
        <td>输入</td>
        <td>表示在伪量化pergroup和mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，对输入weight进行反量化计算的groupSize输入，描述一组反量化参数对应的待反量化数据量在Reduce方向的大小。</td>
        <td>pergroup<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，量化传入值的范围为[32, k-1]且值要求是32的倍数；mx<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，值固定要求传32；perchannel<a href="../../../docs/zh/context/量化介绍.md" target="_blank">量化模式</a>下，值传0。</td>
        <td>UINT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>计算输出，公式中的y，device侧的aclTensor。</td>
        <td>-</td>
        <td>与输入x的数据类型一致</td>
        <td>ND</td>
        <td>shape支持(m, n)</td>
        <td>×</td>
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
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed;width: 1030px">
    <colgroup>
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
        <td>如果传入参数是必选输入，输出或者必选属性，且是空指针，则返回161001。</td>
      </tr>
      <tr>
        <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="2">161002</td>
        <td>输入tensor的shape、数据格式、连续性等不符合要求。</td>
      </tr>
      <tr>
        <td>不支持空tensor场景。</td>
      </tr>
      <tr>
        <td>ACLNN_ERR_RUNTIME_ERROR</td>
        <td>361001</td>
        <td>产品型号不支持。</td>
      </tr>
    </tbody>
  </table>

## aclnnWeightQuantBatchMatmulNz

- **参数说明**
  <table>
    <thead>
      <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
    </thead>
    <tbody>
      <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
      <tr><td>workspaceSize</td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口`aclnnWeightQuantBatchMatmulNzGetWorkspaceSize`获取。</td></tr>
      <tr><td>executor</td><td>输入</td><td>op执行器，包含了算子计算流程。</td></tr>
      <tr><td>stream</td><td>输入</td><td>指定执行任务的Stream。</td></tr>
    </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。


## 约束说明
  - 确定性说明：aclnnWeightQuantBatchMatmulNz默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。
  - 支持的量化模式：perchannel[量化模式](../../../docs/zh/context/量化介绍.md)、pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)。
  - weight数据类型为FLOAT/FLOAT4_E2M1时，仅支持pergroup[量化模式](../../../docs/zh/context/量化介绍.md)和mx[量化模式](../../../docs/zh/context/量化介绍.md)，pergroup[量化模式](../../../docs/zh/context/量化介绍.md)下，要求`antiquantScale`与`x`的数据类型相同，mx[量化模式](../../../docs/zh/context/量化介绍.md)下，要求`antiquantScale`类型为FLOAT8_E8M0类型，且不支持antiquantOffsetOptional输入。
  - weight数据类型为INT32/INT4时，仅支持perchannel[量化模式](../../../docs/zh/context/量化介绍.md)和pergroup[量化模式](../../../docs/zh/context/量化介绍.md)，要求`antiquantScale`与`x`的数据类型相同。

输入和输出支持以下数据类型和shape组合：
- <term>Ascend 950PR/Ascend 950DT</term>：
  | x        | weight            | antiquantScale | antiquantOffsetOptional | biasOptional          | y        |weight shape      |antiquantScale shape                      |antiquantOffsetOptional shape                     |biasOptional shape|
  | ----     | ------------------| ---------------| ------------------------| ----------------------| -------- | -----------------| --------                                 | --------                                         | --------         |
  | BFLOAT16 | INT32/INT4        | BFLOAT16       | NULL/BFLOAT16           | null/BFLOAT16/FLOAT32 | BFLOAT16 |（k, n/8）/（k, n）|（k/antiquantGroupSize, n）/（1, n）/（n,）| null /（k/antiquantGroupSize, n）/（1, n）/（n,） | null /（1, n）   |
  | FLOAT16  | INT32/INT4        | FLOAT16        | NULL/FLOAT16            | null/FLOAT16          | FLOAT16  |（k, n/8）/（k, n）|（k/antiquantGroupSize, n）/（1, n）/（n,）| null /（k/antiquantGroupSize, n）/（1, n）/（n,） | null /（1, n）   |
  | BFLOAT16 | FLOAT/FLOAT4_E2M1 | BFLOAT16       | NULL                    | null/BFLOAT16         | BFLOAT16 |（k, n/8）/（k, n）|（k/antiquantGroupSize, n）               | null                                             | null /（1, n）   |
  | FLOAT16  | FLOAT/FLOAT4_E2M1 | FLOAT16        | NULL                    | null/FLOAT16          | FLOAT16  |（k, n/8）/（k, n）|（k/antiquantGroupSize, n）               | null                                             | null /（1, n）   |
  | BFLOAT16 | FLOAT/FLOAT4_E2M1 | FLOAT8_E8M0    | NULL                    | null/BFLOAT16         | BFLOAT16 |（n, k/8）/（n, k）|（n, k/32）                               | null                                             | null /（1, n）   |
  | FLOAT16  | FLOAT/FLOAT4_E2M1 | FLOAT8_E8M0    | NULL                    | null/FLOAT16          | FLOAT16  |（n, k/8）/（n, k）|（n, k/32）                               | null                                             | null /（1, n）   |
  - 组合说明：
    - x的shape均为（m, k），y的shape均为（m, n）。
    - perchannel[量化模式](../../../docs/zh/context/量化介绍.md)下antiquantScale和antiquantOffsetOptional的shape为（1, n）或（n,）。
    - pergroup[量化模式](../../../docs/zh/context/量化介绍.md)下antiquantScale和antiquantOffsetOptional的shape为（k/antiquantGroupSize, n），其中antiquantGroupSize表示k要分组的每组的大小。
    - mx[量化模式](../../../docs/zh/context/量化介绍.md)下antiquantScale和antiquantOffsetOptional的shape为（k/antiquantGroupSize, n）。

## 调用示例

  仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

  ```Cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  template <typename T1>
  inline T1 CeilDiv(T1 a, T1 b)
  {
      return b == 0 ? a : (a + b - 1) / b;
  };
  template <typename T1>
  inline T1 CeilAlign(T1 a, T1 b)
  {
      return (a + b - 1) / b * b;
  };

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto i : shape) {
          shapeSize *= i;
      }
      return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
      const aclTensor* weight, const aclTensor* weightInt4Pack, uint64_t* workspaceSize, aclOpExecutor** executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(
      void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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

  int GetSize(std::vector<int64_t>& shape)
  {
      int64_t size = 1;
      for (auto i : shape) {
          size *= i;
      }
      return size;
  }

  template <typename T>
  int CreateAclTensorB4(
      const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
      aclTensor** tensor, aclFormat format)
  {
      auto size = hostData.size() * sizeof(T);
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
      if (format == aclFormat::ACL_FORMAT_ND) {
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(),
              shape.size(), *deviceAddr);
      } else {
          std::vector<int64_t> nzShape;
          if (dataType == aclDataType::ACL_INT4 || dataType == aclDataType::ACL_FLOAT4_E2M1) {
              nzShape = {CeilDiv(shape[1], (int64_t)16), CeilDiv(shape[0], (int64_t)16), 16, 16};
          } else {
              nzShape = {CeilDiv(shape[1], (int64_t)2), CeilDiv(shape[0], (int64_t)16), 16, 2};
          }
          *tensor = aclCreateTensor(
              shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(),
              nzShape.size(), *deviceAddr);
      }

      return 0;
  }

  void PrintMat(std::vector<float> resultData, std::vector<int64_t> resultShape)
  {
      int64_t m = resultShape[0];
      int64_t n = resultShape[1];
      for (size_t i = 0; i < m; i++) {
          printf(i == 0 ? "[[" : " [");
          for (size_t j = 0; j < n; j++) {
              printf(j == n - 1 ? "%.1f" : "%.1f, ", resultData[i * n + j]);
              if (j == 2 && j + 3 < n) {
                  printf("..., ");
                  j = n - 4;
              }
          }
          printf(i < m - 1 ? "],\n" : "]]\n");
          if (i == 2 && i + 3 < m) {
              printf(" ... \n");
              i = m - 4;
          }
      }
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      aclrtDestroyStream(stream);
      aclrtResetDevice(deviceId);
      aclFinalize();
  }

  int AclnnWeightQuantBatchMatmulNzTest(int32_t deviceId, aclrtStream& stream)
  {
      auto ret = Init(deviceId, &stream);
      aclDataType weightPackedDtype = aclDataType::ACL_FLOAT; // 可选：ACL_FLOAT类型
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      // 2. 构造输入与输出，需要根据API的接口自定义构造
      int64_t m = 16;
      int64_t k = 64;
      int64_t n = 64;
      int64_t groupSize = 32;
      int64_t weightDim0 = k;
      int64_t weightDim1 = n;
      bool isWeightTransposed = false;
      std::vector<int64_t> xShape = {m, k};
      std::vector<int64_t> weightShape = {k, n};
      std::vector<int64_t> antiquantScaleShape = {k / groupSize, n};
      std::vector<int64_t> yShape = {m, n};
      void* xDeviceAddr = nullptr;
      void* weightDeviceAddr = nullptr;
      void* weightB4PackDeviceAddr = nullptr;
      void* antiquantScaleDeviceAddr = nullptr;
      void* yDeviceAddr = nullptr;
      aclTensor* x = nullptr;
      aclTensor* weight = nullptr;
      aclTensor* y = nullptr;
      aclTensor* antiquantScale = nullptr;
      std::vector<int64_t> weightPackedShape;
      weightPackedShape = {weightDim0, weightDim1 / 8};
      std::vector<uint16_t> xHostData(GetSize(xShape), 0b0011110000000000); // float16的1.0
      std::vector<float> weightHostData(GetSize(weightShape), 1.0); // fp32的1.0，经过int4pack后转到fp4_e2m1的1.0
      std::vector<float> yHostData(GetSize(yShape), 0);

      std::vector<uint16_t> antiquantScaleHostData(GetSize(antiquantScaleShape), 0b0011110000000000); // float16的1.0

      // 创建x aclTensor
      ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建other aclTensor
      ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建y aclTensor
      ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建antiquantScale aclTensor
      ret = CreateAclTensor(
          antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT16,
          &antiquantScale);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(
          antiquantScale, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      // 创建yFp16 aclTensor
      void* yFp16DeviceAddr = nullptr;
      aclTensor* yFp16 = nullptr;
      ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 3. 调用CANN算子库API，需要修改为具体的Api名称
      aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ; // 可选：ACL_FORMAT_ND
      aclTensor* weightPacked = nullptr;

      std::vector<int8_t> weightB4PackHostData(n * k / 2, 0); // 一个B8数据存放2个B4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
          weightB4PackHostData.resize(CeilAlign(weightDim1 / 2, (int64_t)8) * CeilAlign(weightDim0, (int64_t)16), 0);
      }
      // 创建weightInt4Pack aclTensor
      ret = CreateAclTensorB4(
          weightB4PackHostData, weightPackedShape, &weightB4PackDeviceAddr, weightPackedDtype, &weightPacked,
          weightFormat);
      std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightPackedTensorPtr(weightPacked, aclDestroyTensor);
      std::unique_ptr<void, aclError (*)(void*)> weightPackedDeviceAddrPtr(weightB4PackDeviceAddr, aclrtFree);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // 对weight做int32转int4pack
      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightPacked, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      void* workspacePackAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspacePackAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspacePackAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspacePackAddrPtr.reset(workspacePackAddr);
      }
      ret = aclnnConvertWeightToINT4Pack(workspacePackAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

      // 调用aclnnWeightQuantBatchMatmulNz第一段接口
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(
          x, weightPacked, antiquantScale, nullptr, nullptr, nullptr, nullptr, groupSize, yFp16, &workspaceSize,
          &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceAddrPtr.reset(workspaceAddr);
      }
      // 调用aclnnWeightQuantBatchMatmulNz第二段接口
      ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);

      // 4. （固定写法）同步等待任务执行结束
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 将输出转为FP32
      workspaceSize = 0;
      executor = nullptr;
      ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      // 根据第一段接口计算出的workspaceSize申请device内存
      void* workspaceCastAddr = nullptr;
      std::unique_ptr<void, aclError (*)(void*)> workspaceCastAddrPtr(nullptr, aclrtFree);
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceCastAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
          workspaceCastAddrPtr.reset(workspaceCastAddr);
      }
      ret = aclnnCast(workspaceCastAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
      auto size = GetShapeSize(yShape);
      std::vector<float> resultData(size, 0);
      ret = aclrtMemcpy(
          resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr, size * sizeof(resultData[0]),
          ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      PrintMat(resultData, yShape);
      return ACL_SUCCESS;
  }

  int main()
  {
      // 1. （固定写法）device/stream初始化，参考acl API手册
      // 根据自己的实际device填写deviceId
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = AclnnWeightQuantBatchMatmulNzTest(deviceId, stream);
      CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("AclnnWeightQuantBatchMatmulNzTest failed. ERROR: %d\n", ret);
                    return ret);

      Finalize(deviceId, stream);
      return 0;
  }
  ```
