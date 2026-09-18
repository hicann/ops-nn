# aclnnConvertWeightToINT4Pack

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/convert_weight_to_int4_pack)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

对输入weight数据做预处理，实现低比特数据由稀疏存储到紧密存储的排布转换。输出weightInt4Pack的[数据格式](../../../docs/zh/context/data_format.md)声明为FRACTAL_NZ时，该算子将[数据格式](../../../docs/zh/context/data_format.md)从ND转为FRACTAL_NZ。

<!-- npu="A3,910b" id7 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：将INT32类型的weight输入数据打包为紧密排布的INT4数据。
<!-- end id7 -->
<!-- npu="950" id8 -->
- <term>Ascend 950PR/Ascend 950DT</term> ：将INT32类型的weight打包为紧密排布的INT4类型，将FLOAT类型的weight打包为紧密排布的FLOAT4_E2M1类型。

<!-- end id8 -->

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnConvertWeightToINT4PackGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnConvertWeightToINT4Pack”接口执行计算。

```cpp
aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(
  const aclTensor *weight,
  aclTensor       *weightInt4Pack,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnConvertWeightToINT4Pack(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnConvertWeightToINT4PackGetWorkspaceSize

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1078px"><colgroup>
  <col style="width: 149px">
  <col style="width: 121px">
  <col style="width: 320px">
  <col style="width: 183px">
  <col style="width: 183px">
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
      <td>weight</td>
      <td>输入</td>
      <td>Matmul类算子的低比特量化后的权重，由32bit类型承载4bit的权重值</td>
      <td>-</td>
      <td>INT32</td>
      <td>ND</td>
      <td>2-3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>weightInt4Pack</td>
      <td>输出</td>
      <td>Matmul类算子的低比特量化后的权重，权重值为4bit且紧密排布</td>
      <td>-</td>
      <td>INT4,INT32</td>
      <td>ND,FRACTAL_NZ</td>
      <td>-</td>
      <td>-</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>如果传入的必选输入、输出、属性是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>传入weight、weightInt4Pack的shape维度不符合要求。</td>
    </tr>
    <tr>
      <td>传入weight、weightInt4Pack的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>传入weight、weightInt4Pack的shape大小不符合约束要求。</td>
    </tr>
    <tr>
      <td>传入空tensor场景。</td>
    </tr>
    <tr>
      <td>输入tensor的Format不是ND。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_RUNTIME_ERROR</td>
      <td rowspan="2">361001</td>
      <td>数据从host侧拷贝到device侧异常。</td>
    </tr>
    <tr>
      <td>数据从device侧拷贝到host侧异常。</td>
    </tr>
  </tbody>
  </table>

## aclnnConvertWeightToINT4Pack

- **参数说明**
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnConvertWeightToINT4PackGetWorkspaceSize获取。</td>
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

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性说明：

  <!-- npu="950,A3,910b" id9 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：aclnnConvertWeightToINT4Pack默认确定性实现。

  <!-- end id9 -->

- 参数间数据类型、数据格式间关系如下：

    <!-- npu="A3,910b" id10 -->
    - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight数据类型</th>
          <th>weight数据格式</th>
          <th>weightInt4Pack数据类型</th>
          <th>weightInt4Pack数据格式</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4</td>
          <td>ND</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 64)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）</td>
          <td>ND</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/64⌉, ⌈dim0/16⌉, 16, 8)</td>
        </tr>
      </tbody></table>

    <!-- end id10 -->
    <!-- npu="950" id11 -->
    - <term>Ascend 950PR/Ascend 950DT</term>：
      <table style="undefined;table-layout: fixed; width: 1532px"><colgroup>
      <col style="width: 200px">
      <col style="width: 121px">
      <col style="width: 175px">
      <col style="width: 170px">
      <col style="width: 160px">
      <col style="width: 208px">
      <col style="width: 185px">
      </colgroup>
      <thead>
        <tr>
          <th>weight数据类型</th>
          <th>weight数据格式</th>
          <th>weightInt4Pack数据类型</th>
          <th>weightInt4Pack数据格式</th>
          <th>weight shape</th>
          <th>weightInt4Pack view shape</th>
          <th>weightInt4Pack storage shape</th>
        </tr></thead>
      <tbody>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT4或FLOAT4_E2M1</td>
          <td>ND</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT4或FLOAT4_E2M1</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为2对齐</td>
          <td>和输入weight保持一致，即(dim0, dim1)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 16)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）或FLOAT（1个FLOAT数据存储8个FLOAT4_E2M1数据）</td>
          <td>ND</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>同view shape</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）或FLOAT（承载FLOAT4_E2M1类型数据，数据表示范围[-6.0, 6.0]）</td>
          <td>ND</td>
          <td>INT32（1个INT32数据存储8个INT4数据）或FLOAT（1个FLOAT数据存储8个FLOAT4_E2M1数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/16⌉, ⌈dim0/16⌉, 16, 2)</td>
        </tr>
        <tr>
          <td>INT32（承载INT4类型数据，数据表示范围[-8, 7]）</td>
          <td>ND</td>
          <td>INT4（1个INT32数据存储8个INT4数据）</td>
          <td>FRACTAL_NZ</td>
          <td>最后一维度为8对齐</td>
          <td>最后一维度为weight最后一维度的1/8，即(dim0, dim1/8)</td>
          <td>(⌈dim1/32⌉, ⌈dim0/16⌉, 16, 32)</td>
        </tr>
      </tbody></table>

    <!-- end id11 -->

## 调用示例

<!-- npu="A3,910b" id12 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：
  示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
  伪量化有aclnnWeightQuantBatchMatmulV2和aclnnWeightQuantBatchMatmulV3接口，这里以aclnnWeightQuantBatchMatmulV2为例。

  ```cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_v2.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))
  #define CEIL_ALIGN(x, y) ((((x) + (y)) - 1) / (y) * (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  extern "C" aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *weightInt4Pack,
      uint64_t *workspaceSize, aclOpExecutor **executor);

  extern "C" aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
      aclrtStream stream);

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
  int CreateAclTensorInt4(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor, aclFormat format) {
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
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
    } else {
      std::vector<int64_t> nzShape;
      if (dataType == aclDataType::ACL_INT4) {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 64};
      } else {
          nzShape = {CEIL_DIV(shape[1], 64), CEIL_DIV(shape[0], 16), 16, 8};
      }
      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                                aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    }

    return 0;
  }

  int main() {
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    aclDataType weightInt4PackDtype = aclDataType::ACL_INT4;
    aclFormat weightFormat = aclFormat::ACL_FORMAT_FRACTAL_NZ;
    bool isWeightTransposed = true;

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    int64_t m = 16;
    int64_t k = 72;
    int64_t n = 17;
    int64_t weightDim0 = k;
    int64_t weightDim1 = n;
    if (isWeightTransposed) {
      weightDim0 = n;
      weightDim1 = k;
    }
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {weightDim0, weightDim1};
    std::vector<int64_t> weightInt4PackShape;
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      weightInt4PackShape = {weightDim0, weightDim1};
    } else {
      weightInt4PackShape = {weightDim0, weightDim1/8};
    }
    std::vector<int64_t> yShape = {m, n};
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* weightInt4PackDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    aclTensor* y = nullptr;
    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1);
    std::vector<float> yHostData(m * n, 0);

    std::vector<int64_t> antiquantScaleShape = {n};
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    std::vector<float> antiquantScaleHostData(n, 1);

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    if (weightInt4PackDtype == aclDataType::ACL_INT4) {
      std::vector<int8_t> weightInt4PackHostData(n * k / 2, 0); //一个int8数据存放2个int4数据，所以这里除以2
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/2, 32) * CEIL_ALIGN(weightDim0, 16), 0);
      }
      // 创建weightInt4Pack aclTensor
      ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack, weightFormat);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    } else {
      std::vector<int32_t> weightInt4PackHostData(n * k / 8, 1); //一个int32数据存放8个int4数据，所以这里除以8
      if (weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
        weightInt4PackHostData.resize(CEIL_ALIGN(weightDim1/8, 8) * CEIL_ALIGN(weightDim0, 16), 0);
        ret = CreateAclTensorInt4(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                  weightInt4PackDtype, &weightInt4Pack, weightFormat);
      } else {
          // 创建weightInt4Pack aclTensor
          ret = CreateAclTensor(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                                weightInt4PackDtype, &weightInt4Pack);
      }
      CHECK_RET(ret == ACL_SUCCESS, return ret);
    }
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr, aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建xFp16 aclTensor
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建antiquantScale aclTensor
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr, aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建yFp16 aclTensor
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    void* workspaceAddr = nullptr;

    // 对weight做int32转int4pack
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);

    // weight为转置场景，且weightInt4Pack shape为NZ时，需要调用aclInitTensor转换为非连续的tensor
    if (isWeightTransposed && weightFormat == aclFormat::ACL_FORMAT_FRACTAL_NZ) {
      std::vector<int64_t> strides(weightInt4PackShape.size(), 1);
      for (int64_t i = weightInt4PackShape.size() - 2; i >= 0; i--) {
          strides[i] = weightInt4PackShape[i + 1] * strides[i + 1];
      }
      std::swap(strides[0], strides[1]);
      std::swap(weightInt4PackShape[0], weightInt4PackShape[1]);
      std::vector<int64_t> nzShape = {CEIL_DIV(k, 64), CEIL_DIV(n, 16), 16, 8};
      if (weightInt4PackDtype == aclDataType::ACL_INT4) {
          nzShape[3] = 64;
      }
      aclInitTensor(weightInt4Pack, weightInt4PackShape.data(), weightInt4PackShape.size(), weightInt4PackDtype, strides.data(), 0,
                    weightFormat, nzShape.data(), nzShape.size(), weightInt4PackDeviceAddr);
    }

    // 调用cast生成FP16的输入
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize0 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast0 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize1 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast1 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 调用aclnnWeightQuantBatchMatmulV2第一段接口
    ret = aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr, nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    // 调用aclnnWeightQuantBatchMatmulV2第二段接口
    ret = aclnnWeightQuantBatchMatmulV2(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulV2 failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 将输出转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize2 failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存

    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast2 failed. ERROR: %d\n", ret); return ret);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(weightInt4Pack);
    aclDestroyTensor(antiquantScale);
    aclDestroyTensor(y);
    aclDestroyTensor(xFp16);
    aclDestroyTensor(antiquantScaleFp16);
    aclDestroyTensor(yFp16);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(weightInt4PackDeviceAddr);
    aclrtFree(antiquantScaleDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(xFp16DeviceAddr);
    aclrtFree(antiquantScaleFp16DeviceAddr);
    aclrtFree(yFp16DeviceAddr);

    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }

<!-- end id12 -->
<!-- npu="950" id13 -->
- <term>Ascend 950PR/Ascend 950DT</term>：

  示例代码如下（INT32 输入），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
  Ascend 950 上 FRACTAL_NZ 场景需调用 aclnnWeightQuantBatchMatmulNz 接口，这里以 aclnnWeightQuantBatchMatmulNz 为例。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  // 固定写法，资源初始化
  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  // 创建连续排布的ND格式aclTensor，并将hostData拷贝到device侧
  // device侧内存大小按hostData的实际字节数申请（每个元素占sizeof(T)字节）
  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
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

  // 创建4bit紧密排布（INT4/FLOAT4_E2M1）且存储格式为FRACTAL_NZ的aclTensor
  // 逻辑shape按原始ND矩阵(k, n)传入；Ascend 950上NZ的storageShape为(ceil(n/16), ceil(k/16), 16, 16)，
  // 每个分块为16*16个4bit数据（占128字节），device侧内存按storageShape的大小申请
  // hostData为打包后的字节数据（2个4bit存放在1个int8中），仅拷贝hostData.size()个字节，
  // 尾部对齐部分无需初始化，调用aclnnConvertWeightToINT4Pack后整块内存会被重写
  template <typename T>
  int CreateAclTensorNz(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = CEIL_DIV(shape[1], 16) * CEIL_DIV(shape[0], 16) * 128;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, hostData.size() * sizeof(T), hostData.data(), hostData.size() * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor，storageShape按NZ分块排布计算
    std::vector<int64_t> nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // 本示例为NZ场景：weightInt4Pack的存储格式为FRACTAL_NZ，matmul需调用aclnnWeightQuantBatchMatmulNz接口
    // （Ascend 950上aclnnWeightQuantBatchMatmulV2/V3接口的weight仅支持ND格式）
    // 计算规模：x为(m, k)，weight为(k, n)，y为(m, n)
    // 输入数据全为1，antiquantScale全为1，预期y的每个元素均为k
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> weightInt4PackShape = {k, n}; // INT4的shape按元素个数传入
    std::vector<int64_t> antiquantScaleShape = {n};    // perchannel场景下scale的shape为(n,)
    std::vector<int64_t> yShape = {m, n};

    std::vector<float> xHostData(m * k, 1);
    std::vector<int32_t> weightHostData(k * n, 1); // INT32的weight，每个int32承载1个int4数据
    std::vector<float> antiquantScaleHostData(n, 1);
    std::vector<float> yHostData(m * n, 0);
    // INT4紧密排布后2个int4存放在1个int8中，host数据个数为weight元素个数的一半
    std::vector<int8_t> weightInt4PackHostData(k * n / 2, 0);

    // 3. 创建各输入输出aclTensor，aclTensor和device内存均由unique_ptr管理，任意路径退出时自动释放
    // 创建x aclTensor（FP32）
    void* xDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // 创建weight aclTensor（INT32）
    void* weightDeviceAddr = nullptr;
    aclTensor* weight = nullptr;
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_INT32, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // 创建weightInt4Pack aclTensor（INT4，FRACTAL_NZ），存放打包后的weight
    void* weightInt4PackDeviceAddr = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    ret = CreateAclTensorNz(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                            aclDataType::ACL_INT4, &weightInt4Pack);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightInt4PackTensorPtr(weightInt4Pack, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightInt4PackDeviceAddrPtr(weightInt4PackDeviceAddr, aclrtFree);
    // 创建antiquantScale aclTensor（FP32）
    void* antiquantScaleDeviceAddr = nullptr;
    aclTensor* antiquantScale = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleDeviceAddr,
                          aclDataType::ACL_FLOAT, &antiquantScale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleTensorPtr(antiquantScale, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleDeviceAddrPtr(antiquantScaleDeviceAddr, aclrtFree);
    // 创建y aclTensor（FP32）
    void* yDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    // 创建xFp16 aclTensor（FP16），matmul要求x与antiquantScale的数据类型一致
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    // 创建antiquantScaleFp16 aclTensor（FP16）
    void* antiquantScaleFp16DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp16 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp16DeviceAddr,
                          aclDataType::ACL_FLOAT16, &antiquantScaleFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp16TensorPtr(antiquantScaleFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp16DeviceAddrPtr(antiquantScaleFp16DeviceAddr, aclrtFree);
    // 创建yFp16 aclTensor（FP16），matmul的输出
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);

    // 4. 调用aclnnConvertWeightToINT4Pack，将INT32稀疏存储的weight转换为INT4紧密存储，
    // weightInt4Pack为FRACTAL_NZ格式时，算子内部同时完成ND到NZ的排布转换；该算子无需workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 后续各算子按需申请workspace：第一段接口返回workspaceSize，第二段接口执行前申请，由unique_ptr自动释放

    // 5. 调用aclnnCast，将x由FP32转为FP16
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 调用aclnnCast，将antiquantScale由FP32转为FP16
    ret = aclnnCastGetWorkspaceSize(antiquantScale, aclDataType::ACL_FLOAT16, antiquantScaleFp16, &workspaceSize,
                                    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 6. 调用aclnnWeightQuantBatchMatmulNz执行伪量化matmul，antiquantGroupSize传0表示perchannel场景
    ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp16, nullptr, nullptr,
                                                        nullptr, nullptr, 0, yFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 7. 调用aclnnCast，将输出由FP16转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 8.（固定写法）同步等待任务执行结束，将device侧结果拷贝至host侧并打印
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 9.（固定写法）释放stream等资源，aclTensor和device内存已由unique_ptr自动释放
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

<!-- end id13 -->
<!-- npu="950" id14 -->
- <term>Ascend 950PR/Ascend 950DT</term>：
  示例代码如下（FLOAT 输入），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。
  Ascend 950 上 FRACTAL_NZ 场景需调用 aclnnWeightQuantBatchMatmulNz 接口，这里以 aclnnWeightQuantBatchMatmulNz 为例。

  ```cpp
  #include <iostream>
  #include <memory>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cast.h"
  #include "aclnnop/aclnn_convert_weight_to_int4_pack.h"
  #include "aclnnop/aclnn_weight_quant_batch_matmul_nz.h"

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

  #define CEIL_DIV(x, y) ((((x) + (y)) - 1) / (y))

  int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for (auto i : shape) {
      shapeSize *= i;
    }
    return shapeSize;
  }

  // 固定写法，资源初始化
  int Init(int32_t deviceId, aclrtStream* stream) {
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
  }

  // 创建连续排布的ND格式aclTensor，并将hostData拷贝到device侧
  // device侧内存大小按hostData的实际字节数申请（每个元素占sizeof(T)字节）
  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor) {
    auto size = hostData.size() * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
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

  // 创建4bit紧密排布（INT4/FLOAT4_E2M1）且存储格式为FRACTAL_NZ的aclTensor
  // 逻辑shape按原始ND矩阵(k, n)传入；Ascend 950上NZ的storageShape为(ceil(n/16), ceil(k/16), 16, 16)，
  // 每个分块为16*16个4bit数据（占128字节），device侧内存按storageShape的大小申请
  // hostData为打包后的字节数据（2个4bit存放在1个int8中），仅拷贝hostData.size()个字节，
  // 尾部对齐部分无需初始化，调用aclnnConvertWeightToINT4Pack后整块内存会被重写
  template <typename T>
  int CreateAclTensorNz(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                        aclDataType dataType, aclTensor** tensor) {
    auto size = CEIL_DIV(shape[1], 16) * CEIL_DIV(shape[0], 16) * 128;
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, hostData.size() * sizeof(T), hostData.data(), hostData.size() * sizeof(T),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }

    // 调用aclCreateTensor接口创建aclTensor，storageShape按NZ分块排布计算
    std::vector<int64_t> nzShape = {CEIL_DIV(shape[1], 16), CEIL_DIV(shape[0], 16), 16, 16};
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0,
                              aclFormat::ACL_FORMAT_FRACTAL_NZ, nzShape.data(), nzShape.size(), *deviceAddr);
    return 0;
  }

  int main() {
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // 本示例为NZ场景：weightInt4Pack的存储格式为FRACTAL_NZ，matmul需调用aclnnWeightQuantBatchMatmulNz接口
    // （Ascend 950上aclnnWeightQuantBatchMatmulV2/V3接口的weight仅支持ND格式）
    // 计算规模：x为(m, k)，weight为(k, n)，y为(m, n)，antiquantScale为(ceil(k/groupSize), n)
    // 输入数据全为1，antiquantScale全为1，预期y的每个元素均为k
    int64_t m = 16;
    int64_t k = 64;
    int64_t n = 64;
    int64_t antiquantGroupSize = 32; // pergroup场景的groupSize
    std::vector<int64_t> xShape = {m, k};
    std::vector<int64_t> weightShape = {k, n};
    std::vector<int64_t> weightInt4PackShape = {k, n}; // FLOAT4_E2M1的shape按元素个数传入
    std::vector<int64_t> antiquantScaleShape = {k / antiquantGroupSize, n}; // pergroup场景下scale的shape为(ceil(k/groupSize), n)
    std::vector<int64_t> yShape = {m, n};

    std::vector<float> xHostData(m * k, 1);
    std::vector<float> weightHostData(k * n, 1); // FLOAT的weight，每个float承载1个fp4数据
    // 使用uint8承载float8_e8m0数据，0b01111111表示fp8_e8m0的1.0
    std::vector<uint8_t> antiquantScaleHostData(k * n / antiquantGroupSize, 0b01111111);
    std::vector<float> yHostData(m * n, 0);
    // FLOAT4_E2M1紧密排布后2个fp4存放在1个int8中，host数据个数为weight元素个数的一半
    std::vector<int8_t> weightInt4PackHostData(k * n / 2, 0);

    // 3. 创建各输入输出aclTensor，aclTensor和device内存均由unique_ptr管理，任意路径退出时自动释放
    // 创建x aclTensor（FP32）
    void* xDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDeviceAddrPtr(xDeviceAddr, aclrtFree);
    // 创建weight aclTensor（INT32）
    void* weightDeviceAddr = nullptr;
    aclTensor* weight = nullptr;
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightTensorPtr(weight, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightDeviceAddrPtr(weightDeviceAddr, aclrtFree);
    // 创建weightInt4Pack aclTensor（FLOAT4_E2M1，FRACTAL_NZ），存放打包后的weight
    void* weightInt4PackDeviceAddr = nullptr;
    aclTensor* weightInt4Pack = nullptr;
    ret = CreateAclTensorNz(weightInt4PackHostData, weightInt4PackShape, &weightInt4PackDeviceAddr,
                            aclDataType::ACL_FLOAT4_E2M1, &weightInt4Pack);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> weightInt4PackTensorPtr(weightInt4Pack, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> weightInt4PackDeviceAddrPtr(weightInt4PackDeviceAddr, aclrtFree);
    // 创建antiquantScaleFp8 aclTensor（FLOAT8_E8M0），matmul直接使用，无需转换
    void* antiquantScaleFp8DeviceAddr = nullptr;
    aclTensor* antiquantScaleFp8 = nullptr;
    ret = CreateAclTensor(antiquantScaleHostData, antiquantScaleShape, &antiquantScaleFp8DeviceAddr,
                          aclDataType::ACL_FLOAT8_E8M0, &antiquantScaleFp8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> antiquantScaleFp8TensorPtr(antiquantScaleFp8, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> antiquantScaleFp8DeviceAddrPtr(antiquantScaleFp8DeviceAddr, aclrtFree);
    // 创建y aclTensor（FP32）
    void* yDeviceAddr = nullptr;
    aclTensor* y = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(y, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yDeviceAddrPtr(yDeviceAddr, aclrtFree);
    // 创建xFp16 aclTensor（FP16），antiquantScale为FLOAT8_E8M0时x的数据类型支持FLOAT16
    void* xFp16DeviceAddr = nullptr;
    aclTensor* xFp16 = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xFp16DeviceAddr, aclDataType::ACL_FLOAT16, &xFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xFp16TensorPtr(xFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xFp16DeviceAddrPtr(xFp16DeviceAddr, aclrtFree);
    // 创建yFp16 aclTensor（FP16），matmul的输出
    void* yFp16DeviceAddr = nullptr;
    aclTensor* yFp16 = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yFp16DeviceAddr, aclDataType::ACL_FLOAT16, &yFp16);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yFp16TensorPtr(yFp16, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yFp16DeviceAddrPtr(yFp16DeviceAddr, aclrtFree);

    // 4. 调用aclnnConvertWeightToINT4Pack，将FLOAT稀疏存储的weight转换为FLOAT4_E2M1紧密存储，
    // weightInt4Pack为FRACTAL_NZ格式时，算子内部同时完成ND到NZ的排布转换；该算子无需workspace
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    ret = aclnnConvertWeightToINT4PackGetWorkspaceSize(weight, weightInt4Pack, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnConvertWeightToINT4PackGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnConvertWeightToINT4Pack(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnConvertWeightToINT4Pack failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 后续各算子按需申请workspace：第一段接口返回workspaceSize，第二段接口执行前申请，由unique_ptr自动释放

    // 5. 调用aclnnCast，将x由FP32转为FP16
    ret = aclnnCastGetWorkspaceSize(x, aclDataType::ACL_FLOAT16, xFp16, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 5. 调用aclnnWeightQuantBatchMatmulNz执行伪量化matmul，antiquantGroupSize为pergroup场景的groupSize
    ret = aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(xFp16, weightInt4Pack, antiquantScaleFp8, nullptr, nullptr,
                                                        nullptr, nullptr, antiquantGroupSize, yFp16, &workspaceSize,
                                                        &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnWeightQuantBatchMatmulNzGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnWeightQuantBatchMatmulNz(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnWeightQuantBatchMatmulNz failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 6. 调用aclnnCast，将输出由FP16转为FP32
    ret = aclnnCastGetWorkspaceSize(yFp16, aclDataType::ACL_FLOAT, y, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCastGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      workspaceAddrPtr.reset(workspaceAddr);
    }
    ret = aclnnCast(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnCast failed. ERROR: %d\n", ret); return ret);
    workspaceAddr = nullptr;
    workspaceAddrPtr.reset();

    // 7.（固定写法）同步等待任务执行结束，将device侧结果拷贝至host侧并打印
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    auto size = GetShapeSize(yShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), yDeviceAddr,
                      size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 8.（固定写法）释放stream等资源，aclTensor和device内存已由unique_ptr自动释放
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
  ```

<!-- end id14 -->
