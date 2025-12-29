# aclnnFusedMatmul

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/fused_mat_mul)

## 产品支持情况

<table style="undefined;table-layout: fixed; width: 1000px"><colgroup>
    <col style="width: 600px">
    <col style="width: 120px">
    </colgroup>
    <thread>
      <tr>
        <th>产品</th>
        <th>是否支持</th>
      </tr></thread>
    <tbody>
      <tr>
        <td><term>昇腾910_95 AI处理器</term></td>
        <td>√</td>
      </tr>
      <tr>
        <td><term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas 200I/500 A2 推理产品</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas 推理系列产品</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas 训练系列产品</term></td>
        <td>×</td>
      </tr>
      <tr>
        <td><term>Atlas 200/300/500 推理产品</term></td>
        <td>×</td>
      </tr>
  </tbody></table>

## 功能说明

- 接口功能：矩阵乘与通用向量计算融合。
- 计算公式：

  $$
  y = OP((x1 @ x2 + bias), x3)
  $$

  OP类型由fusedOpType输入定义，支持如下：

  add运算:

  $$
  y=(x1 @ x2 + bias) + x3
  $$

  mul运算:

  $$
  y=(x1 @ x2 + bias) ∗ x3
  $$

  gelu_tanh运算:

  $$
  y = gelu\_tanh(x1 @ x2 + bias)
  $$

  gelu_erf运算:

  $$
  y = gelu\_erf(x1 @ x2 + bias)
  $$

  relu运算:

  $$
  y = relu(x1 @ x2 + bias)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnFusedMatmulGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnFusedMatmul”接口执行计算。

```cpp
aclnnStatus aclnnFusedMatmulGetWorkspaceSize(
  const aclTensor *x1,
  const aclTensor *x2,
  const aclTensor *bias,
  const aclTensor *x3,
  const char      *fusedOpType,
  int8_t           cubeMathType,
  const aclTensor *y,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```
```cpp
aclnnStatus aclnnFusedMatmul(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnFusedMatmulGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 170px">
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
        <th>输入/输出</th>
        <th>描述</th>
        <th>使用说明</th>
        <th>数据类型</th>
        <th>数据格式</th>
        <th>维度</th>
        <th>非连续</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>x1</td>
        <td>输入</td>
        <td>表示矩阵乘的第一个矩阵，对应公式中的x1。</td>
        <td><ul><li>数据类型需要与x2满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x2</td>
        <td>输入</td>
        <td>表示矩阵乘的第二个矩阵，对应公式中的x2。</td>
        <td><ul><li>数据类型需要与x1满足数据类型推导规则（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>bias</td>
        <td>输入</td>
        <td>表示偏置项，对应公式中的bias。</td>
        <td><ul><li>仅当fusedOpType为""、"relu"、"add"、"mul"时生效，其他情况传入空指针即可。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>1-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>x3</td>
        <td>输入</td>
        <td>表示融合操作的第二个矩阵，对应公式中的x3。</td>
        <td><ul>-</td>
        <td>数据类型与x1保持一致</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>y</td>
        <td>输出</td>
        <td>表示计算的输出矩阵，对应公式中的y。</td>
        <td><ul><li>数据类型需要与x1和x2推导后的数据类型一致（参见<a href="../../../docs/zh/context/互推导关系.md" target="_blank">互推导关系</a>）。</li></td>
        <td>FLOAT16、BFLOAT16、FLOAT32</td>
        <td>ND</td>
        <td>2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>cubeMathType</td>
        <td>输入</td>
        <td>表示Cube单元的计算逻辑，Host侧的整型。</td>
        <td><ul><li>如果输入的数据类型存在互推导关系，该参数默认对互推导后的数据类型进行处理。</li>
        <li>当前版本支持输入0和3。</li>
        <li>0：KEEP_DTYPE，保持输入的数据类型进行计算。</li>
        <li>3：USE_HF32，使能HF32。</li>
        <li>当fusedOpType为gelu_erf、gelu_tanh时，当前版本不生效，传入0即可。</li></td>
        <td>INT8</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>fusedOpType</td>
        <td>输入</td>
        <td>表示指定Matmul算子支持的融合模式，对应公式中的OP。</td>
        <td><ul><li>融合模式取值必须是""（表示不做融合）、"add"、"mul"、"gelu_erf"、"gelu_tanh"、"relu"中的一种。</li></td>
        <td>STRING</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输出</td>
        <td>返回用户需要在Device侧申请的workspace大小。</td>
        <td><ul>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>executor</td>
        <td>输出</td>
        <td>返回op执行器，包含了算子计算流程。</td>
        <td><ul>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景报错：
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thread>
      <tr>
        <th>返回值</th>
        <th>错误码</th>
        <th>描述</th>
      </tr></thread>
    <tbody>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_NULLPTR</td>
        <td rowspan="3">161001</td>
        <td>传入的x1、x2、和y是空指针。</td>
      </tr>
      <tr>
        <td>fusedOpType为add、mul时，传入的x3为空指针。</td>
      </tr>
      <tr>
        <td>fusedOpType为gelu_tanh、gelu_erf，传入的bias不是空指针。</td>
      </tr>
      <tr>
        <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="6">161002</td>
        <td>x1和x2的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>表x1、x2或y的数据格式不在支持的范围内。</td>
      </tr>
      <tr>
        <td>x1和x2的维度不是二维。</td>
      </tr>
      <tr>
        <td>fusedOpType为add、mul时，x3的shape不跟输出shape保持一致。</td>
      </tr>
      <tr>
        <td>传入的fusedOpType不属于"gelu_tanh","gelu_erf"中的一种。</td>
      </tr>
      <tr>
        <td>传入的cubeMathType只能是0,其他的值不支持。</td>
      </tr>
  </tbody></table>


## aclnnFusedMatmul

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1030px"><colgroup>
    <col style="width: 250px">
    <col style="width: 130px">
    <col style="width: 650px">
    </colgroup>
    <thread>
      <tr>
        <th>参数名</th>
        <th>输入/输出</th>
        <th>描述</th>
      </tr></thread>
    <tbody>
      <tr>
        <td>workspace</td>
        <td>输入</td>
        <td>在Device侧申请的workspace内存地址。</td>
      </tr>
      <tr>
        <td>workspaceSize</td>
        <td>输入</td>
        <td>在Device侧申请的workspace大小，由第一段接口aclnnFusedMatmulGetWorkspaceSize获取。</td>
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
  </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明
  - 确定性说明：aclnnFusedMatmul默认确定性实现。
  - 当fusedOpType取值为"gelu_erf"、"gelu_tanh"时，x1、x2、x3的数据类型必须为BFLOAT16、FLOAT16;当fusedOpType为""、"relu"、"add"、"mul"时, x1、x2、x3的数据类型必须为BFLOAT16、FLOAT16、FLOAT32。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_fused_matmul.h"

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
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> xShape = {16, 32};
  std::vector<int64_t> x2Shape = {32, 16};
  std::vector<int64_t> x3Shape = {16, 16};
  std::vector<int64_t> yShape = {16, 16};
  void* xDeviceAddr = nullptr;
  void* x2DeviceAddr = nullptr;
  void* x3DeviceAddr = nullptr;
  void* yDeviceAddr = nullptr;
  aclTensor* x = nullptr;
  aclTensor* x2 = nullptr;
  aclTensor* x3 = nullptr;
  aclTensor* y = nullptr;
  std::vector<float> xHostData(512, 1);
  std::vector<float> x2HostData(512, 1);
  std::vector<float> x3HostData(256, 1);
  std::vector<float> yHostData(256, 0);
  // 创建x aclTensor
  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x2 aclTensor
  ret = CreateAclTensor(x2HostData, x2Shape, &x2DeviceAddr, aclDataType::ACL_FLOAT16, &x2);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建x3 aclTensor
  ret = CreateAclTensor(x3HostData, x3Shape, &x3DeviceAddr, aclDataType::ACL_FLOAT16, &x3);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 创建y aclTensor
  ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT16, &y);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  int8_t cubeMathType = 0;
  const char* fusedOpType = "add";
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  // 调用aclnnFusedMatmul第一段接口
  ret = aclnnFusedMatmulGetWorkspaceSize(x, x2, nullptr, x3, fusedOpType, cubeMathType, y, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmulGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnFusedMatmul第二段接口
  ret = aclnnFusedMatmul(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnFusedMatmul failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
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
  aclDestroyTensor(x2);
  aclDestroyTensor(x3);
  aclDestroyTensor(y);

  // 7. 释放device资源，需要根据具体API的接口定义修改
  aclrtFree(xDeviceAddr);
  aclrtFree(x2DeviceAddr);
  aclrtFree(x3DeviceAddr);
  aclrtFree(yDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}
```