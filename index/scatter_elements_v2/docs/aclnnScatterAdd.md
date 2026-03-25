# aclnnScatterAdd

📄 [查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/scatter_elements_v2)

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | √ |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：将src tensor中的值按指定的轴方向和index tensor中的位置关系逐个填入self tensor中，若有多于一个src值被填入到self的同一位置，那么这些值将会在这一位置上进行累加。

  用例：
  对于一个3D tensor，self会按照如下的规则进行更新：

  ```
  self[index[i][j][k]][j][k] += src[i][j][k] # 如果 dim == 0
  self[i][index[i][j][k]][k] += src[i][j][k] # 如果 dim == 1
  self[i][j][index[i][j][k]] += src[i][j][k] # 如果 dim == 2
  ```

  在计算时需要满足以下要求：
  - self、index和src的维度数量必须相同。
  - 对于每一个维度d，有index.size(d) <= src.size(d)。
  - 对于每一个维度d，如果有d != dim，有index.size(d) <= self.size(d)。
  - dim取值范围为[-self.dim(), self.dim() - 1]。

- 示例：

  输入tensor $self = \begin{bmatrix} [1&2&3] \\ [4&5&6] \\ [7&8&9] \end{bmatrix}$，
  索引tensor $index = \begin{bmatrix} [0&2&1] \\ [0&0&1] \end{bmatrix}$， dim = 1，
  源tensor $src = \begin{bmatrix} [10&11&12] \\ [13&14&15] \end{bmatrix}$，
  输出tensor $output = \begin{bmatrix} [11&14&14] \\ [31&20&6] \\ [7&8&9] \end{bmatrix}$

  dim = 1 表示scatter_add根据$index$在tensor的列上进行累加。

  $output[0][0] = self[0][0] + src[0][0]$ = 1 + 10，

  $output[0][1] = self[0][1] + src[0][2]$ = 2 + 12，

  $output[0][2] = self[0][2] + src[0][1]$ = 3 + 11，

  $output[1][0] = self[1][0] + src[1][0] + src[1][1]$ = 4 + 13 + 14，

  $output[1][1] = self[1][1] + src[1][2]$ = 5 + 15，

  $output[1][2] = self[1][2]$ = 6，

  $output[2][0] = self[2][0]$ = 7，

  $output[2][1] = self[2][1]$ = 8，

  $output[2][2] = self[2][2]$ = 9。

  其中，$self$、$index$、$src$的维度数量均为2，$index$每个维度大小{2，3}都不大于$src$的对应维度大小{2，3}，在dim != 1的维度上（dim = 0），$index$的维度大小{2}不大于$self$的对应维度大小{3}，$index$中的最大值{2}，小于$self$在dim = 1维度的大小{3}。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScatterAddGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnScatterAdd”接口执行计算。

```Cpp
aclnnStatus aclnnScatterAddGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterAdd(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterAddGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
    <col style="width: 180px">
    <col style="width: 120px">
    <col style="width: 280px">
    <col style="width: 320px">
    <col style="width: 250px">
    <col style="width: 120px">
    <col style="width: 140px">
    <col style="width: 140px">
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
        <td>self（aclTensor*）</td>
        <td>输入</td>
        <td>公式中的输入self，scatter的目标张量。</td>
        <td>维度数量需要与index和src相同。<br>数据类型与src的数据类型一致。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim（int64_t）</td>
        <td>输入</td>
        <td>计算公式中的输入dim。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index（aclTensor*）</td>
        <td>输入</td>
        <td>公式中的输入index。</td>
        <td>维度数量需要与src相同。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src（aclTensor*）</td>
        <td>输入</td>
        <td>公式中的输入src，源张量。</td>
        <td>维度数量需要与index相同。<br>数据类型与self的数据类型一致。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>公式中的output。</td>
        <td>shape需要与self一致。<br>数据类型与self的数据类型一致。</td>
        <td>BFLOAT16、FLOAT16、FLOAT32、DOUBLE、INT64、INT32、INT16、INT8、UINT8、BOOL、COMPLEX64、COMPLEX128</td>
        <td>ND</td>
        <td>-</td>
        <td>×</td>
      </tr>
      <tr>
        <td>workspaceSize（uint64_t*）</td>
        <td>输出</td>
        <td>返回需要在Device侧申请的workspace大小。</td>
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
    </tbody></table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1150px"><colgroup>
    <col style="width: 300px">
    <col style="width: 134px">
    <col style="width: 716px">
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
        <td>传入的self、index、src或out是空指针。</td>
      </tr>
      <tr>
        <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="3">161002</td>
        <td>self、index、src或out的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
        <td>self和out的shape不一致。</td>
      </tr>
      <tr>
        <td>src或index的shape不合法。</td>
      </tr>
    </tbody></table>

## aclnnScatterAdd

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
    <col style="width: 200px">
    <col style="width: 130px">
    <col style="width: 770px">
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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnScatterAddGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnScatterAdd默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_add.h"

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
  std::vector<int64_t> selfShape = {4, 4};
  std::vector<int64_t> indexShape = {3, 4};
  std::vector<int64_t> srcShape = {4, 4};
  std::vector<int64_t> outShape = {4, 4};
  int64_t dim = 0;
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* srcDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* src = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData(16, 0);
  std::vector<int64_t> indexHostData = {0, 1, 2, 1, 0, 1, 2, 0, 2, 2, 1, 0};
  std::vector<float> srcHostData = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  std::vector<float> outHostData(16, 0);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT64, &index);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建src aclTensor
  ret = CreateAclTensor(srcHostData, srcShape, &srcDeviceAddr, aclDataType::ACL_FLOAT, &src);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnScatterAdd第一段接口
  ret = aclnnScatterAddGetWorkspaceSize(self, dim, index, src, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAddGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnScatterAdd第二段接口
  ret = aclnnScatterAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnScatterAdd failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
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
  aclDestroyTensor(index);
  aclDestroyTensor(src);
  aclDestroyTensor(out);
  // 7. 释放device资源，需要根据具体API的接口定义参数
  aclrtFree(selfDeviceAddr);
  aclrtFree(indexDeviceAddr);
  aclrtFree(srcDeviceAddr);
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