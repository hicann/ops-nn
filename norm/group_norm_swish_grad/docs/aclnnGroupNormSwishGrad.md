# aclnnGroupNormSwishGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/norm/group_norm_swish_grad)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：[aclnnGroupNormSwish](../../group_norm_swish/docs/aclnnGroupNormSwish.md)的反向操作。
- 计算公式：

$$
dYTemp_i = \hat{x_i} \cdot gamma + beta
$$

$$
dSwishTemp_i = dYTemp_i  - \frac{dYTemp_i}{ \exp {dYTemp_i * (-swishScale)} + 1} + 1
$$

$$
dYNew_i = \frac{dSwishTemp_i}{ \exp (- dYTemp_i * swishScale) + 1} * dy
$$

$$
dBeta = \sum_{i=1}^n dYNew_i
$$

$$
dGamma = \sum_{i=1}^n (dYNew_i \cdot \hat{x_i})
$$

$$
dx = rstd \cdot (dYNew * gamma - \hat{x} * (\sum_{i=1}^n gamma_i * dGamma) - (\sum_{i=1}^n gamma_i * dBeta))
$$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupNormSwishGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupNormSwishGrad”接口执行计算。

```c++
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(
    const aclTensor *dy, 
    const aclTensor *mean, 
    const aclTensor *rstd, 
    const aclTensor *x, 
    const aclTensor *gamma, 
    const aclTensor *beta, 
    int64_t          numGroups, 
    char            *dataFormatOptional, 
    double           swishScale, 
    bool             dgammaIsRequire, 
    bool             dbetaIsRequire, 
    const aclTensor *dxOut, 
    const aclTensor *dgammaOut, 
    const aclTensor *dbetaOut, 
    uint64_t        *workspaceSize, 
    aclOpExecutor  **executor)
```

```c++
aclnnStatus aclnnGroupNormSwishGrad(
    void *         workspace, 
    uint64_t       workspaceSize, 
    aclOpExecutor *executor, 
    aclrtStream    stream)
```

## aclnnGroupNormSwishGradGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
      <col style="width: 220px">
      <col style="width: 120px">
      <col style="width: 287px">
      <col style="width: 387px">
      <col style="width: 187px">
      <col style="width: 187px">
      <col style="width: 187px">
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
          <td>dy (aclTensor*)</td>
          <td>输入</td>
          <td>反向计算的梯度，公式中的<code>dy</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>维度支持2D到8D，1维为N，第2维为C。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>mean (aclTensor*)</td>
          <td>输入</td>
          <td>正向计算的第二个输出，表示input分组后每个组的均值，公式中的<code>mean</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>数据类型与gamma相同，其中N与dy的第0维度保持一致。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>rstd (aclTensor*)</td>
          <td>输入</td>
          <td>正向计算的第三个输出，表示input分组后每个组的标准差倒数，公式中的<code>rstd</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>数据类型与gamma相同，其中N与dy的第0维度保持一致。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>√</td>
      </tr>
      <tr>
          <td>x (aclTensor*)</td>
          <td>输入</td>
          <td>正向输入，公式中的<code>x</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>数据类型和shape与dy相同。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>√</td>
      </tr>
      <tr>
          <td>gamma (aclTensor*)</td>
          <td>输入</td>
          <td>每个channel的缩放系数，公式中的<code>gamma</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>数据类型和维度与dy相同，元素个数需要等于C。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>beta (aclTensor*)</td>
          <td>输入</td>
          <td>每个channel的偏移系数，公式中的<code>beta</code>。</td>
          <td><ul><li>不支持空tensor。</li><li>数据类型和维度与dy相同，元素个数需要等于C。</li></ul></td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>1</td>
          <td>√</td>
      </tr>
      <tr>
          <td>numGroups (int64_t)</td>
          <td>输入</td>
          <td>输入gradOut的C维度分为group组。</td>
          <td>group需大于0，C必须可以被group整除并且比值不能超过4000。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dataFormatOptional (char*)</td>
          <td>输入</td>
          <td>数据格式。</td>
          <td>建议值NCHW。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>swishScale (double)</td>
          <td>输入</td>
          <td>计算系数。</td>
          <td>建议值1.0。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dgammaIsRequire (bool)</td>
          <td>输入</td>
          <td>是否需要输出dgamma。</td>
          <td>建议值TRUE。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dbetaIsRequire (bool)</td>
          <td>输入</td>
          <td>是否需要输出dbeta。</td>
          <td>建议值TRUE。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>dxOut (aclTensor*)</td>
          <td>输出</td>
          <td>公式中的<code>dx</code>。</td>
          <td>数据类型和shape与x相同。</td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2-8</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dgammaOut (aclTensor*)</td>
          <td>输出</td>
          <td>公式中的<code>dGamma</code>。</td>
          <td>数据类型和shape与gamma相同。</td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>
          <td>x</td>
      </tr>
      <tr>
          <td>dbetaOut (aclTensor*)</td>
          <td>输出</td>
          <td>beta 的输出梯度，公式中的<code>dBeta</code>。</td>
          <td>数据类型和shape与gamma相同。</td>
          <td>FLOAT16、FLOAT、BFLOAT16</td>
          <td>ND</td>
          <td>2</td>  
          <td>x</td>
      </tr>
      <tr>
          <td>workspaceSize (uint64_t*)</td>
          <td>输出</td>
          <td>返回需要在Device侧申请的workspace大小。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      <tr>
          <td>executor (aclOpExecutor**)</td>
          <td>输出</td>
          <td>返回op执行器，包含了算子计算流程。</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
      </tr>
      </tbody></table>

- **返回值**
  
  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
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
      <td>传入的dy、mean、rstd、x、gamma、beta、dxOut、dgammaOut、dbetaOut是空指针时。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>dy数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>mean、rstd、x、gamma、beta的数据类型与dy不同。</td>
    </tr>
    <tr>
      <td>dxOut的数据类型与dy不同。</td>
    </tr>
  </tbody></table>

## aclnnGroupNormSwishGrad

- **参数说明**
  <table>
  <thead>
      <tr>
          <th>参数名</th>
          <th>输入/输出</th>
          <th>描述</th>
      </tr>
  </thead>
  <tbody>
      <tr>
          <td>workspace</td>
          <td>输入</td>
          <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
          <td>workspaceSize</td>
          <td>输入</td>
          <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupNormSwishGradGetWorkspaceSize获取。</td>
      </tr>
      <tr>
          <td>executor</td>
          <td>输入</td>
          <td> op执行器，包含了算子计算流程。</td>
      </tr>
      <tr>
          <td>stream</td>
          <td>输入</td>
          <td> 指定执行任务的Stream。</td>
      </tr>
  </tbody></table>

- **返回值**
  
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性计算
  - aclnnGroupNormSwishGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 输入shape限制：
    1. numGroups大于0。
    2. C能被group整除。
    3. dy的元素个数等于$N * C * HxW$。
    4. mean的元素个数等于$N * group$。
    5. rstd的元素个数等于$N * group$。
    6. x的元素个数等于$N * C * HxW$。
    7. gamma的元素个数等于C。
    8. beta的元素个数等于C。
    9. C与group比值不超过4000。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_group_norm_swish_grad.h"

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
  std::vector<int64_t> dyShape = {2, 3, 4};
  std::vector<int64_t> meanShape = {2, 1};
  std::vector<int64_t> rstdShape = {2, 1};
  std::vector<int64_t> xShape = {2, 3, 4};
  std::vector<int64_t> gammaShape = {3};
  std::vector<int64_t> betaShape = {3};
  std::vector<int64_t> dxOutShape = {2, 3, 4};
  std::vector<int64_t> dgammaOutShape = {3, 1};
  std::vector<int64_t> dbetaOutShape = {3, 1};
  void* dyDeviceAddr = nullptr;
  void* meanDeviceAddr = nullptr;
  void* rstdDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* gammaDeviceAddr = nullptr;
  void* betaDeviceAddr = nullptr;
  void* dxOutDeviceAddr = nullptr;
  void* dgammaOutDeviceAddr = nullptr;
  void* dbetaOutDeviceAddr = nullptr;
  aclTensor* dy = nullptr;
  aclTensor* mean = nullptr;
  aclTensor* rstd = nullptr;
  aclTensor* x = nullptr;
  aclTensor* gamma = nullptr;
  aclTensor* beta = nullptr;
  aclTensor* dxOut = nullptr;
  aclTensor* dgammaOut = nullptr;
  aclTensor* dbetaOut = nullptr;
  std::vector<float> dyHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> meanHostData = {2.0, 2};
  std::vector<float> rstdHostData = {2.0, 2};
  std::vector<float> xHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                  13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> gammaHostData = {2.0, 2, 2};
  std::vector<float> betaHostData = {2.0, 2, 2};
  std::vector<float> dxOutHostData = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                                   13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  std::vector<float> dgammaOutHostData = {2.0, 2, 2};
  std::vector<float> dbetaOutHostData = {2.0, 2, 2};
  int64_t numGroups = 1;
  char* dataFormatOptional = nullptr;
  float swishScale = 1.0f;
  bool dgammaIsRequire = true;
  bool dbetaIsRequire = true;
  // 创建dy aclTensor
  ret = CreateAclTensor(dyHostData, dyShape, &dyDeviceAddr, aclDataType::ACL_FLOAT, &dy);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mean aclTensor
  ret = CreateAclTensor(meanHostData, meanShape, &meanDeviceAddr, aclDataType::ACL_FLOAT, &mean);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建rstd aclTensor
  ret = CreateAclTensor(rstdHostData, rstdShape, &rstdDeviceAddr, aclDataType::ACL_FLOAT, &rstd);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建gamma aclTensor
  ret = CreateAclTensor(gammaHostData, gammaShape, &gammaDeviceAddr, aclDataType::ACL_FLOAT, &gamma);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建beta aclTensor
  ret = CreateAclTensor(betaHostData, betaShape, &betaDeviceAddr, aclDataType::ACL_FLOAT, &beta);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建dxOut aclTensor
  ret = CreateAclTensor(dxOutHostData, dxOutShape, &dxOutDeviceAddr, aclDataType::ACL_FLOAT, &dxOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建dgammaOut aclTensor
  ret = CreateAclTensor(dgammaOutHostData, dgammaOutShape, &dgammaOutDeviceAddr, aclDataType::ACL_FLOAT, &dgammaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建dbetaOut aclTensor
  ret = CreateAclTensor(dbetaOutHostData, dbetaOutShape, &dbetaOutDeviceAddr, aclDataType::ACL_FLOAT, &dbetaOut);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的API
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnGroupNormSwishGrad第一段接口
  ret = aclnnGroupNormSwishGradGetWorkspaceSize(dy, mean, rstd, x, gamma, beta, numGroups, dataFormatOptional, swishScale, dgammaIsRequire, dbetaIsRequire, dxOut, dgammaOut, dbetaOut, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
  }
  // 调用aclnnGroupNormSwishGrad第二段接口
  ret = aclnnGroupNormSwishGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupNormSwishGrad failed. ERROR: %d\n", ret); return ret);
  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(dxOutShape);
  ret = aclrtMemcpy(dxOutHostData.data(), dxOutHostData.size() * sizeof(dxOutHostData[0]), dxOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dxOutHostData[%ld] is: %f\n", i, dxOutHostData[i]);
  }

  size = GetShapeSize(dgammaOutShape);
  ret = aclrtMemcpy(dgammaOutHostData.data(), dgammaOutHostData.size() * sizeof(dgammaOutHostData[0]), dgammaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dgammaOutHostData[%ld] is: %f\n", i, dgammaOutHostData[i]);
  }

  size = GetShapeSize(dbetaOutShape);
  ret = aclrtMemcpy(dbetaOutHostData.data(), dbetaOutHostData.size() * sizeof(dbetaOutHostData[0]), dbetaOutDeviceAddr, size * sizeof(float),
                    ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("dbetaOutHostData[%ld] is: %f\n", i, dbetaOutHostData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(dy);
  aclDestroyTensor(mean);
  aclDestroyTensor(rstd);
  aclDestroyTensor(x);
  aclDestroyTensor(gamma);
  aclDestroyTensor(beta);
  aclDestroyTensor(dxOut);
  aclDestroyTensor(dgammaOut);
  aclDestroyTensor(dbetaOut);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(dyDeviceAddr);
  aclrtFree(meanDeviceAddr);
  aclrtFree(rstdDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(gammaDeviceAddr);
  aclrtFree(betaDeviceAddr);
  aclrtFree(dxOutDeviceAddr);
  aclrtFree(dgammaOutDeviceAddr);
  aclrtFree(dbetaOutDeviceAddr);

  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
