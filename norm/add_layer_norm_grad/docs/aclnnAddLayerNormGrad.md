# aclnnAddLayerNormGrad

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 接口功能：LayerNorm是一种归一化方法，可以将网络层输入数据归一化到[0, 1]之间。LayerNormGrad算子是深度学习中用于反向传播阶段的一个关键算子，主要用于计算LayerNorm操作的梯度。AddLayerNormGrad算子是将Add和LayerNormGrad融合起来，减少搬入搬出操作。

- 计算公式：

  - 正向公式：（D为reduce轴大小）

    $$
    x= inputx1 + inputx2
    $$

    $$
    \operatorname{LayerNorm}(x)=\frac{x_i−\operatorname{E}(x)}{\sqrt{\operatorname{Var}(x)+ eps}}*gamma + beta
    $$

    $$
    其中\operatorname{E}(x_i)=\frac{1}{D}\sum_{1}^{D}{x_i}
    $$

    $$
    \operatorname{Var}(x_i)=\frac{1}{D}\sum_{1}^{D}{(x_i-\operatorname{E}(x))^2}
    $$

  - 反向公式：

    $$
    x= inputx1 + inputx2
    $$

    $$
    dxOut = \sum_{j}{inputdy_i * gamma_j * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}} + dsumOptional
    $$

    $$
    dgammaOut = \sum_{j}{inputdy_i * \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}}
    $$

    $$
    dbetaOut = \sum_{j}{inputdy_i}
    $$

    其中：

    - $\hat{x_j}$：

      $$
      \hat{x_j}=({x_i-\operatorname{E}(x)}) * {rstd}
      $$

    - $rstd$：

      $$
      rstd=\frac {1}{\sqrt{\operatorname{Var}(x)}}
      $$

    - $\frac{{\rm d}\hat{x_j}}{{\rm d}x_i}$：

      $$
      \frac{{\rm d}\hat{x_j}}{{\rm d}x_i}=(\delta_{ij} - \frac{{\rm d}\operatorname{E}(x)}{{\rm d}  x_i}) * \frac{1}{\sqrt{\operatorname{Var}(x_i)}}-\frac{1}{\operatorname{Var}(x_i)}  (x_j-\operatorname{E}(x))\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}
      $$

      其中，当i=j时，$\delta_{ij}$=1；当i!=j时，$\delta_{ij}$=0。


    - $\frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}$：

      $$
      \frac{{\rm d}\operatorname{E}(x)}{{\rm d}x_i}=\frac{1}{D}
      $$

      其中，D为x中参加均值计算的数量。

    - $\frac{\rm d \operatorname{Var}(x_i)}{\rm dx}$：

      $$
      \frac{\rm d \operatorname{Var}(x_i)}{\rm dx}=\frac{1}{D}\frac{1}{\sqrt{\operatorname{Var}  (x_i)}}(x_i-\operatorname{E}(x))
      $$

    - 化简后的$dxOut$：

      $$
      dxOut = rstd * ({inputdy_i * gamma_j} - \frac{1}{D} * (\sum_{j}{inputdy_i * gamma_j} + \hat      {x_j} * \sum_{j}{inputdy_i * gamma_j * \hat{x_j}})) + dsumOptional
      $$

## 函数原型

每个算子分为[两段式接口](../../../docs/context/两段式接口.md)，必须先调用“aclnnAddLayerNormGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnAddLayerNormGrad”接口执行计算。

```Cpp
aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *rstd,
  const aclTensor *mean,
  const aclTensor *gamma,
  const aclTensor *dsumOptional,
  const aclTensor *dxOut,
  const aclTensor *dgammaOut,
  const aclTensor *dbetaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAddLayerNormGrad(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnAddLayerNormGradGetWorkspaceSize

- **参数说明：**


  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
  <col style="width: 145px">
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
      <td>dy</td>
      <td>输入</td>
      <td>表示主要的grad输入。对应公式中的`inputdy`。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x1</td>
      <td>输入</td>
      <td>表示为正向融合算子的输入x1。对应公式中的`inputx1`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据类型与`dy`的shape、数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x2</td>
      <td>输入</td>
      <td>表示为正向融合算子的输入x2。对应公式中的`inputx2`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据类型与`dy`的shape、数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd</td>
      <td>输入</td>
      <td>表示正向输入x1、x2之和的标准差的倒数。对应公式中的`rstd`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与`dy`满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>（前几维的维度和`dy`前几维的维度相同，前几维指`dy`的维度减去`gamma`的维度，表示不需要norm的维度）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>输入</td>
      <td>表示正向输入x1、x2之和的均值。对应公式中的`E(x)`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与`dy`满足<a href="../../../docs/context/broadcast关系.md">broadcast关系</a>（前几维的维度和`dy`前几维的维度相同，前几维指`dy`的维度减去`gamma`的维度，表示不需要norm的维度）。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma</td>
      <td>输入</td>
      <td>表示正向输入的gamma。对应公式中的`gamma`。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与`dy`的数据类型保持一致。</li><li>shape的维度值与`dy`需要norm的维度值相同。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dsumOptional</td>
      <td>输入</td>
      <td>表示额外的反向梯度累加输入。对应公式中的`dsumOptional`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据类型与`dy`的shape、数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dxOut</td>
      <td>输出</td>
      <td>表示Add的结果输出`x`的梯度。对应公式中的`dxOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape、数据类型与`dy`的shape、数据类型保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut</td>
      <td>输出</td>
      <td>表示入参gamma的梯度。对应公式中的`dgammaOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与输入`gamma`一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dbetaOut</td>
      <td>输出</td>
      <td>表示正向入参beta的反向梯度。对应公式中的`dbetaOut`。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与输入`gamma`一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-8</td>
      <td>√</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
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
      <td rowspan="1">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="1">161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
  </tbody></table>

## aclnnAddLayerNormGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 953px"><colgroup>
  <col style="width: 173px">
  <col style="width: 112px">
  <col style="width: 668px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAddLayerNormGradGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/context/aclnn返回码.md)。

## 约束说明

- **功能维度**
  - 数据类型支持
    - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：dy、x1、x2、gamma、dsumOptional、dxOut支持FLOAT32、FLOAT16、BFLOAT16。
    - rstd、mean、dgammaOut、dbetaOut支持：FLOAT32。
  - 数据格式支持：ND。
- **未支持类型说明**

  DOUBLE：指令不支持DOUBLE。

- **边界值场景说明**
  - 当输入是Inf时，输出为Inf。
  - 当输入是NaN时，输出为NaN。
- 确定性计算：
  - aclnnAddLayerNormGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_add_layer_norm_grad.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
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

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> dyShape = {3, 1, 4};
    std::vector<int64_t> x1Shape = {3, 1, 4};
    std::vector<int64_t> x2Shape = {3, 1, 4};
    std::vector<int64_t> rstdShape = {3, 1, 1};
    std::vector<int64_t> meanShape = {3, 1, 1};
    std::vector<int64_t> gammaShape = {4};
    std::vector<int64_t> dsumOptionalShape = {3, 1, 4};
    std::vector<int64_t> outputpdxShape = {3, 1, 4};
    std::vector<int64_t> outputpdgammaShape = {4};
    std::vector<int64_t> outputpdbetaShape = {4};
    void* dyDeviceAddr = nullptr;
    void* x1DeviceAddr = nullptr;
    void* x2DeviceAddr = nullptr;
    void* rstdDeviceAddr = nullptr;
    void* meanDeviceAddr = nullptr;
    void* gammaDeviceAddr = nullptr;
    void* dsumOptionalDeviceAddr = nullptr;
    void* outputpdxDeviceAddr = nullptr;
    void* outputpdgammaDeviceAddr = nullptr;
    void* outputpdbetaDeviceAddr = nullptr;
    aclTensor* dy = nullptr;
    aclTensor* x1 = nullptr;
    aclTensor* x2 = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* mean = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* dsumOptional = nullptr;
    aclTensor* outputpdx = nullptr;
    aclTensor* outputpdgamma = nullptr;
    aclTensor* outputpdbeta = nullptr;
    std::vector<float> dyHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x1HostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> x2HostData = {2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8};
    std::vector<int32_t> rstdHostData = {0, 1, 2};
    std::vector<int32_t> meanHostData = {0, 1, 2};
    std::vector<int32_t> gammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> dsumOptionalHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdxHostData = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> outputpdgammaHostData = {0, 1, 2, 3};
    std::vector<int32_t> outputpdbetaHostData = {0, 1, 2, 3};

    // 创建self aclTensor
    ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x1HostData, x1Shape, &x1DeviceAddr, aclDataType::ACL_FLOAT, &x1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT, &x2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        dsumOptionalHostData, dsumOptionalShape, &dsumOptionalDeviceAddr, aclDataType::ACL_FLOAT, &dsumOptional);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = CreateAclTensor(outputpdxHostData, outputpdxShape, &outputpdxDeviceAddr, aclDataType::ACL_FLOAT, &outputpdx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdgammaHostData, outputpdgammaShape, &outputpdgammaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdgamma);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(
        outputpdbetaHostData, outputpdbetaShape, &outputpdbetaDeviceAddr, aclDataType::ACL_FLOAT, &outputpdbeta);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // aclnnAddLayerNormGrad接口调用示例
    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    // 调用aclnnAddLayerNormGrad第一段接口
    LOG_PRINT("\nUse aclnnAddLayerNormGrad Port.");
    ret = aclnnAddLayerNormGradGetWorkspaceSize(
        dy, x1, x2, rstd, mean, gamma, dsumOptional, outputpdx, outputpdgamma, outputpdbeta, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGradGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnAddLayerNormGrad第二段接口
    ret = aclnnAddLayerNormGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAddLayerNormGrad failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto outputpdxsize = GetShapeSize(outputpdxShape);
    std::vector<float> resultDataPdx(outputpdxsize, 0);
    ret = aclrtMemcpy(
        resultDataPdx.data(), resultDataPdx.size() * sizeof(resultDataPdx[0]), outputpdxDeviceAddr,
        outputpdxsize * sizeof(resultDataPdx[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdx output");
    for (int64_t i = 0; i < outputpdxsize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdx[i]);
    }
    auto outputpdgammasize = GetShapeSize(outputpdgammaShape);
    std::vector<float> resultDataPdGamma(outputpdgammasize, 0);
    ret = aclrtMemcpy(
        resultDataPdGamma.data(), resultDataPdGamma.size() * sizeof(resultDataPdGamma[0]), outputpdgammaDeviceAddr,
        outputpdgammasize * sizeof(resultDataPdGamma[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdgamma output");
    for (int64_t i = 0; i < outputpdgammasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdGamma[i]);
    }
    auto outputpdbetasize = GetShapeSize(outputpdbetaShape);
    std::vector<float> resultDataPdBeta(outputpdbetasize, 0);
    ret = aclrtMemcpy(
        resultDataPdBeta.data(), resultDataPdBeta.size() * sizeof(resultDataPdBeta[0]), outputpdbetaDeviceAddr,
        outputpdbetasize * sizeof(resultDataPdBeta[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    LOG_PRINT("== pdbeta output");
    for (int64_t i = 0; i < outputpdbetasize; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultDataPdBeta[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(dy);
    aclDestroyTensor(x1);
    aclDestroyTensor(x2);
    aclDestroyTensor(rstd);
    aclDestroyTensor(mean);
    aclDestroyTensor(gamma);
    aclDestroyTensor(dsumOptional);
    aclDestroyTensor(outputpdx);
    aclDestroyTensor(outputpdgamma);
    aclDestroyTensor(outputpdbeta);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(dyDeviceAddr);
    aclrtFree(x1DeviceAddr);
    aclrtFree(x2DeviceAddr);
    aclrtFree(rstdDeviceAddr);
    aclrtFree(meanDeviceAddr);
    aclrtFree(gammaDeviceAddr);
    aclrtFree(dsumOptionalDeviceAddr);
    aclrtFree(outputpdxDeviceAddr);
    aclrtFree(outputpdgammaDeviceAddr);
    aclrtFree(outputpdbetaDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
