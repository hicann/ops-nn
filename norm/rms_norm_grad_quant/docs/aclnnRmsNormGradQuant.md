# aclnnRmsNormGradQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/rms_norm_grad_quant)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                   |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> |    x     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    x     |
| <term>Atlas 200I/500 A2 推理产品</term>                  |    ×     |
| <term>Atlas 推理系列产品</term>                          |    x     |
| <term>Atlas 训练系列产品</term>                          |    ×     |

## 功能说明

- 接口功能：RmsNormGrad是用于计算RmsNorm的梯度，即在反向传播过程中计算输入张量的梯度的算子。RmsNormGradQuant算子将RmsNormGrad和Quantize两个算子融合，RmsNormGrad计算完dx后进行quant计算，减少搬入搬出操作。
- 算子公式：

  $$
  dx_i= (dy_i * g_i - \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * \operatorname{Mean}(\mathbf{y})) * \frac{1} {\operatorname{Rms}(\mathbf{x})},  \quad \text { where } \operatorname{Mean}(\mathbf{y}) = \frac{1}{n}\sum_{i=1}^n (dy_i * g_i * x_i * \frac{1}{\operatorname{Rms}(\mathbf{x})})
  $$

  - divMode为True时：

    $$
    dx_i\_quant=round((dx_i/scales\_x)+offset\_x)
    $$

  - divMode为False时：

    $$
    dx_i\_quant=round((dx_i*scales\_x)+offset\_x)
    $$

    $$
    dg_i = \frac{1}{\operatorname{Rms}(\mathbf{x})} * x_i * dy_i
    $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用`aclnnRmsNormGradQuantGetWorkspaceSize`接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用`aclnnRmsNormGradQuant`接口执行计算。

```Cpp
aclnnStatus aclnnRmsNormGradQuantGetWorkspaceSize(
  const aclTensor *dy,
  const aclTensor *x,
  const aclTensor *rstd,
  const aclTensor *gamma,
  const aclTensor *scalesX,
  const aclTensor *offsetXOptional,
  const char      *quantMode,
  const bool       divMode,
  aclTensor       *dxOut,
  aclTensor       *dgammaOut,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnRmsNormGradQuant(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnRmsNormGradQuantGetWorkspaceSize

- **参数说明**

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
      <td>dy（aclTensor*）</td>
      <td>输入</td>
      <td>表示反向传回的梯度，对应公式中的dy。</td>
      <td><ul><li>支持空Tensor。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子的输入，表示被标准化的数据，对应公式中的x。</td>
      <td><ul><li>支持空Tensor。</li><li>shape和dtype与dy保持一致。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rstd（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子的中间计算结果，对应公式中的Rms(x)。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要满足rstd_shape = x_shape[0:n]，n < x_shape.dims()，n与gamma的n一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gamma（aclTensor*）</td>
      <td>输入</td>
      <td>表示正向算子进行归一化计算的缩放因子（权重），对应公式中的gamma。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要满足gamma_shape = x_shape[n:], n < x_shape.dims()。</li><li>dtype与dy相同或为FLOAT32。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scalesX（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入梯度量化缩放因子，对应公式中的scales_x。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[1]，维度为1。</li><li>dtype与dy相同或为FLOAT32。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>offsetXOptional（aclTensor*）</td>
      <td>输入</td>
      <td>表示输入梯度量化零点，对应公式中的offset_x。</td>
      <td><ul><li>支持空Tensor。</li><li>shape为[1]，维度为1。</li></ul></td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>x</td>
    </tr>
    <tr>
      <td>quantMode（char*）</td>
      <td>输入</td>
      <td>量化模式。</td>
      <td><ul><li>仅支持“static”，表示静态量化模式。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>divMode（bool）</td>
      <td>输入</td>
      <td>公式中决定量化公式是否使用除法的参数，数据类型为bool。</td>
      <td><ul><li>支持True和False。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dxOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示输入x的梯度，对应公式中的dx。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与入参dy的shape保持一致。</li></ul></td>
      <td>HIFLOAT8、INT8</td>
      <td>ND</td>
      <td>2-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>dgammaOut（aclTensor*）</td>
      <td>输出</td>
      <td>表示gamma的梯度，对应公式中的dg。</td>
      <td><ul><li>支持空Tensor。</li><li>shape与入参gamma的shape保持一致。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1-7</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize（uint64_t*）</td>
      <td>输出</td>
      <td>返回用户需要在Device侧申请的workspace大小。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
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
      <td>输入或输出的数据类型不在支持范围之内。</td>
    </tr>
  </tbody></table>

## aclnnRmsNormGradQuant

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnRmsNormGradQuantGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 各产品支持数据类型说明：
  - <term>Ascend 950PR/Ascend 950DT</term>：
    
    | `dy`数据类型 | `x`数据类型 | `rstd`数据类型 | `gamma`数据类型 | `scalesX`数据类型 | `offsetXOptional`数据类型 | `dxOut`数据类型 | `dgammaOut`数据类型 |
    | ------------ | ----------- | -------------- | --------------- | ------------------------- | ---------------------------- | --------------- | ------------------- |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT32      | FLOAT32     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | FLOAT32                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT16                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | BFLOAT16                  | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT16                   | INT32                        | HIFLOAT8        | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | BFLOAT16                  | INT32                        | HIFLOAT8        | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT32      | FLOAT32     | FLOAT32        | FLOAT32         | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | FLOAT32                   | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT16         | FLOAT16                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | BFLOAT16        | BFLOAT16                  | INT32                        | INT8            | FLOAT32             |
    | FLOAT16      | FLOAT16     | FLOAT32        | FLOAT32         | FLOAT16                   | INT32                        | INT8            | FLOAT32             |
    | BFLOAT16     | BFLOAT16    | FLOAT32        | FLOAT32         | BFLOAT16                  | INT32                        | INT8            | FLOAT32             |

- 确定性计算：
  - aclnnRmsNormGradQuant默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <memory>
#include "acl/acl.h"
#include "aclnnop/aclnn_rms_norm_grad_quant.h"

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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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

void Finalize(int32_t deviceId, aclrtStream stream)
{
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();
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
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

int main()
{
    // 1. device/stream 初始化
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出
    // RmsNormGradQuant: dy[M, D], x[M, D], rstd[M], gamma[D], scalesX[1], offsetX[1] -> dx[M, D], dgamma[D]
    std::vector<int64_t> dy_shape = {8, 64};
    std::vector<int64_t> x_shape = {8, 64};
    std::vector<int64_t> rstd_shape = {8};
    std::vector<int64_t> gamma_shape = {64};
    std::vector<int64_t> scales_x_shape = {1};
    std::vector<int64_t> offset_x_shape = {1};
    std::vector<int64_t> dx_shape = {8, 64};
    std::vector<int64_t> dgamma_shape = {64};

    void* dy_device_addr = nullptr;
    void* x_device_addr = nullptr;
    void* rstd_device_addr = nullptr;
    void* gamma_device_addr = nullptr;
    void* scales_x_device_addr = nullptr;
    void* offset_x_device_addr = nullptr;
    void* dx_device_addr = nullptr;
    void* dgamma_device_addr = nullptr;
    void* workspace_addr = nullptr;

    aclTensor* dy = nullptr;
    aclTensor* x = nullptr;
    aclTensor* rstd = nullptr;
    aclTensor* gamma = nullptr;
    aclTensor* scalesX = nullptr;
    aclTensor* offsetX = nullptr;
    aclTensor* dxOut = nullptr;
    aclTensor* dgammaOut = nullptr;

    // 初始化输入数据
    // dy, x: FP16, 1.0 = 0x3C00
    std::vector<uint16_t> dy_host_data(GetShapeSize(dy_shape), 0x3C00);
    std::vector<uint16_t> x_host_data(GetShapeSize(x_shape), 0x3C00);
    // rstd: FP32, 1.0
    std::vector<float> rstd_host_data(GetShapeSize(rstd_shape), 1.0f);
    // gamma: FP32, 1.0
    std::vector<float> gamma_host_data(GetShapeSize(gamma_shape), 1.0f);
    // scalesX: FP32, 1.0
    std::vector<float> scales_x_host_data(GetShapeSize(scales_x_shape), 1.0f);
    // offsetX: INT32, 0
    std::vector<int32_t> offset_x_host_data(GetShapeSize(offset_x_shape), 0);
    // dx: INT8 输出
    std::vector<int8_t> dx_host_data(GetShapeSize(dx_shape), 0);
    // dgamma: FP32 输出
    std::vector<float> dgamma_host_data(GetShapeSize(dgamma_shape), 0.0f);

    char quantMode[] = "static";
    bool divMode = true;

    LOG_PRINT("Input shape: dy[8, 64], x[8, 64], rstd[8], gamma[64]\n");
    LOG_PRINT("Output shape: dx[8, 64](INT8), dgamma[64](FP32)\n");
    LOG_PRINT("quantMode: static, divMode: true\n");

    // 创建 dy aclTensor (FP16)
    ret = CreateAclTensor(dy_host_data, dy_shape, &dy_device_addr, aclDataType::ACL_FLOAT16, &dy);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dyPtr(dy, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dyDevPtr(dy_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dy failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 x aclTensor (FP16)
    ret = CreateAclTensor(x_host_data, x_shape, &x_device_addr, aclDataType::ACL_FLOAT16, &x);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xPtr(x, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xDevPtr(x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor x failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 rstd aclTensor (FP32)
    ret = CreateAclTensor(rstd_host_data, rstd_shape, &rstd_device_addr, aclDataType::ACL_FLOAT, &rstd);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rstdPtr(rstd, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rstdDevPtr(rstd_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor rstd failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 gamma aclTensor (FP32)
    ret = CreateAclTensor(gamma_host_data, gamma_shape, &gamma_device_addr, aclDataType::ACL_FLOAT, &gamma);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> gammaPtr(gamma, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> gammaDevPtr(gamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor gamma failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 scalesX aclTensor (FP32)
    ret = CreateAclTensor(scales_x_host_data, scales_x_shape, &scales_x_device_addr, aclDataType::ACL_FLOAT, &scalesX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scalesXPtr(scalesX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scalesXDevPtr(scales_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor scalesX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 offsetX aclTensor (INT32)
    ret = CreateAclTensor(offset_x_host_data, offset_x_shape, &offset_x_device_addr, aclDataType::ACL_INT32, &offsetX);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> offsetXPtr(offsetX, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> offsetXDevPtr(offset_x_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor offsetX failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 dxOut aclTensor (INT8)
    ret = CreateAclTensor(dx_host_data, dx_shape, &dx_device_addr, aclDataType::ACL_INT8, &dxOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dxOutPtr(dxOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dxDevPtr(dx_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dxOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 创建 dgammaOut aclTensor (FP32)
    ret = CreateAclTensor(dgamma_host_data, dgamma_shape, &dgamma_device_addr, aclDataType::ACL_FLOAT, &dgammaOut);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> dgammaOutPtr(dgammaOut, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> dgammaDevPtr(dgamma_device_addr, aclrtFree);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor dgammaOut failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 3. 调用 CANN 算子库 API
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;

    // 调用 aclnnRmsNormGradQuant 第一段接口
    LOG_PRINT("Calling aclnnRmsNormGradQuantGetWorkspaceSize...\n");
    ret = aclnnRmsNormGradQuantGetWorkspaceSize(
        dy, x, rstd, gamma, scalesX, offsetX,
        quantMode, divMode,
        dxOut, dgammaOut, &workspace_size, &executor);

    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuantGetWorkspaceSize failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    LOG_PRINT("Workspace size: %lu bytes (%.2f KB)\n", workspace_size, workspace_size / 1024.0);

    // 根据第一段接口计算出的 workspaceSize 申请 device 内存
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspace_size > 0) {
        ret = aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);
        workspaceAddrPtr.reset(workspace_addr);
    }

    // 调用 aclnnRmsNormGradQuant 第二段接口
    LOG_PRINT("Calling aclnnRmsNormGradQuant...\n");
    ret = aclnnRmsNormGradQuant(workspaceAddrPtr.get(), workspace_size, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclnnRmsNormGradQuant failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 4. 同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
              Finalize(deviceId, stream); return ret);

    // 5. 获取输出的值，将 device 侧内存上的结果拷贝至 host 侧
    {
        // 拷贝 dx 结果 (INT8)
        auto size = GetShapeSize(dx_shape);
        std::vector<int8_t> dx_result(size, 0);
        ret = aclrtMemcpy(dx_result.data(), dx_result.size() * sizeof(dx_result[0]),
                          dxDevPtr.get(), size * sizeof(int8_t), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dx from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dx (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dx[%ld] = %d\n", i, dx_result[i]);
        }

        // 拷贝 dgamma 结果 (FP32)
        size = GetShapeSize(dgamma_shape);
        std::vector<float> dgamma_result(size, 0.0f);
        ret = aclrtMemcpy(dgamma_result.data(), dgamma_result.size() * sizeof(dgamma_result[0]),
                          dgammaDevPtr.get(), size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("copy dgamma from device to host failed. ERROR: %d\n", ret);
                  Finalize(deviceId, stream); return ret);

        LOG_PRINT("Output dgamma (first 10 values):\n");
        for (int64_t i = 0; i < std::min(size, (int64_t)10); i++) {
            LOG_PRINT("  dgamma[%ld] = %f\n", i, dgamma_result[i]);
        }
    }

    LOG_PRINT("\n=== RmsNormGradQuant Test PASSED ===\n");

    // 6. 资源自动释放
    Finalize(deviceId, stream);

    return 0;
}
```
