# aclnnBatchMatMul

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/batch_mat_mul_v3)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>昇腾910_95 AI处理器</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品 </term>                             |    √     |
| <term>Atlas 训练系列产品</term>                              |    √     |
| <term>Atlas 200/300/500 推理产品</term>                      |    ×     |

## 功能说明

- 接口功能：完成张量self与张量mat2的矩阵乘计算。仅支持三维的Tensor传入。第一维是batch维度，最后两个维度做矩阵乘法。也支持其中一个输入的batch轴为1时做broadcast，见示例。

- 计算公式：

  $$
  out = self@mat2
  $$

- 示例：

  self的shape是[A, M, K]，mat2的shape是[A, K, N]，计算输出out的shape是[A, M, N]。第一维相等，后两维做矩阵乘运算。
  self的shape是[A, M, K]，mat2的shape是[1, K, N]，计算输出out的shape是[A, M, N]。mat2第一维为1，会broadcast到A，后两维做矩阵乘运算。
  self的shape是[1, M, K]，mat2的shape是[B, K, N]，计算输出out的shape是[B, M, N]。self第一维为1，会broadcast到B，后两维做矩阵乘运算。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnBatchMatMulGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnBatchMatMul”接口执行计算。
```cpp
aclnnStatus aclnnBatchMatMulGetWorkspaceSize(
  const aclTensor *self, 
  const aclTensor *mat2, 
  aclTensor       *out, 
  int8_t           cubeMathType, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnBatchMatMul(
  void            *workspace, 
  uint64_t         workspaceSize, 
  aclOpExecutor   *executor, 
  aclrtStream      stream)
```

## aclnnBatchMatMulGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1508px"><colgroup>
  <col style="width: 151px">
  <col style="width: 121px">
  <col style="width: 301px">
  <col style="width: 331px">
  <col style="width: 237px">
  <col style="width: 111px">
  <col style="width: 111px">
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
      <th>非连续tensor</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>self</td>
      <td>输入</td>
      <td>表示矩阵乘的第一个矩阵，公式中的self。</td>
      <td>数据类型需要与mat2满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>mat2</td>
      <td>输入</td>
      <td>表示矩阵乘的第二个矩阵，公式中的mat2。</td>
      <td><ul><li>数据类型需要与self满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</li><li>mat2的Reduce维度需要与self的Reduce维度大小相等。</li> </ul>
      </td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示矩阵乘的输出矩阵，公式中的out。</td>
      <td>数据类型需要与self与mat2推导之后的数据类型保持一致（参见<a href="../../../docs/zh/context/互推导关系.md">互推导关系</a>和<a href="#约束说明">约束说明</a>）。</td>
      <td>BFLOAT16、FLOAT16、FLOAT32</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>cubeMathType</td>
      <td>输入</td>
      <td>用于指定Cube单元的计算逻辑。</td>
      <td>如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。支持的枚举值如下：<ul>
        <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
        <li>1：ALLOW_FP32_DOWN_PRECISION，支持将输入数据降精度计算。</li>
        <li>2：USE_FP16，支持将输入降精度至FLOAT16计算。</li>
        <li>3：USE_HF32，支持将输入降精度至数据类型HFLOAT32计算。</li></ul>
      </td>
      <td>INT8</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>workspaceSize</td>
      <td>出参</td>
      <td>返回需要在Device侧申请的workspace大小。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>出参</td>
      <td>返回op执行器，包含了算子计算流程。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody></table>

  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：
    - 不支持BFLOAT16数据类型；
    - 当输入数据类型为FLOAT32时不支持cubeMathType=0；
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为FLOAT16计算，当输入为其他数据类型时不做处理；
    - 不支持cubeMathType=3。
  - <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>昇腾910_95 AI处理器</term>：
    - cubeMathType=1，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理；
    - cubeMathType=2，当输入数据类型是FLOAT32，会转换为FLOAT16计算；当输入为其他数据类型时不做处理；
    - cubeMathType=3，当输入数据类型为FLOAT32时，会转换为HFLOAT32计算，当输入为其他数据类型时不做处理。




- **返回值：**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 809px"><colgroup>
  <col style="width: 300px">
  <col style="width: 200px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的self、mat2或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="5">161002</td>
      <td>self、mat2或out的数据类型不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self、mat2或out的数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>self和mat2的维度不是三维。</td>
    </tr>
    <tr>
      <td>self的最后一维和mat2的倒数第二维不相等。</td>
    </tr>
  </tbody>
  </table>

## aclnnBatchMatMul

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 250px">
  <col style="width: 150px">
  <col style="width: 750px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnBatchMatMulGetWorkspaceSize获取。</td>
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

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
- 确定性计算
  - <term>Atlas 训练系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnBatchMatMul默认确定性实现。
  - <term>昇腾910_95 AI处理器</term>: aclnnBatchMatMul默认确定性实现。


## 调用示例
示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_batch_matmul.h"

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
  // 调用aclrtMalloc申请Device侧内存
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

  // 调用aclrtMemcpy将Host侧数据拷贝到Device侧内存上
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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {1, 2, 3};
  std::vector<int64_t> mat2Shape = {1, 3, 4};
  std::vector<int64_t> outShape = {1, 2, 4};
  void* selfDeviceAddr = nullptr;
  void* mat2DeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* mat2 = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5};
  std::vector<float> mat2HostData = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4};
  std::vector<float> outHostData(8, 0);
  int8_t cubeMathType = 1;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建mat2 aclTensor
  ret = CreateAclTensor(mat2HostData, mat2Shape, &mat2DeviceAddr, aclDataType::ACL_FLOAT, &mat2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;

  // aclnnBatchMatMul接口调用示例
  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnBatchMatMul第一段接口
  ret = aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, cubeMathType, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnBatchMatMul第二段接口
  ret = aclnnBatchMatMul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBatchMatMul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将Device侧内存上的结果拷贝至Host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(self);
  aclDestroyTensor(mat2);
  aclDestroyTensor(out);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(selfDeviceAddr);
  aclrtFree(outDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```
