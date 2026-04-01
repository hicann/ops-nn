# aclnnInplaceScatterUpdate

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/scatter)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    √     |

## 功能说明

将tensor updates中的值按指定的轴axis和索引indices逐个更新tensor data中的值。该算子为自定义算子语义，无对应的tensorflow或pytorch接口。

- 示例：
  该算子有3个输入和一个属性：data，updates，indices和axis，其中data是待更新的tensor，updates是存储更新数据的tensor，indices表示更新位置，axis是指定的更新维度。当indices为1维，存在以下两种场景：

  **场景一：** indices为1维，axis指定更新的维度shape为1，indices指定的是每个batch维度（最高维）在axis维度的偏移。

  ```
  样例输入：
  data:(a, b, c, d)
  updates:(a, b, 1, d)
  indices:(a,)
  axis = -2
  ```

      data[i][j][indices[i]][k] = updates[i][j][0][k] # if dim=-2
      data[i][j][k][indices[i]] = updates[i][j][k][0] # if dim=-1

  **场景二：** indices为1维，axis指定更新的维度shape大于1，indices指定的是每个batch维度（最高维）在axis维度的偏移。

  ```
  样例输入：
  data:(a, b, c, d)
  updates:(a, b, e, d), indices[i] + e <= c
  indices:(a,)
  axis = -2 or 2
  ```

      data[i][j][indices[i]+k][l] = updates[i][j][k][l] # if dim=-2
      data[i][j][k][indices[i]+l] = updates[i][j][k][l] # if dim=-1

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnInplaceScatterUpdateGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnInplaceScatterUpdate”接口执行计算。

```Cpp
aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(
    aclTensor*       data,
    const aclTensor* indices,
    const aclTensor* updates,
    int64_t          axis,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatterUpdate(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnInplaceScatterUpdateGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1477px"><colgroup>
    <col style="width: 147px">
    <col style="width: 120px">
    <col style="width: 233px">
    <col style="width: 277px">
    <col style="width: 270px">
    <col style="width: 121px">
    <col style="width: 164px">
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
        <td>data（aclTensor*）</td>
        <td>输入/输出</td>
        <td>待更新的tensor。</td>
        <td>维度数需要与updates一致，不支持空Tensor。</td>
        <td>INT8、UINT8、FLOAT16、FLOAT32、INT32、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT8_E8M0、HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>indices（aclTensor*）</td>
        <td>输入</td>
        <td>更新位置索引。仅支持非负索引，索引数据不支持越界。</td>
        <td>不支持空Tensor。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
        <td>0-2</td>
        <td>√</td>
      </tr>
      <tr>
        <td>updates（aclTensor*）</td>
        <td>输入</td>
        <td>存储更新数据的tensor。</td>
        <td>数据类型需要与data相同，shape的维度数需要与data shape的维度数相同。不支持空Tensor。</td>
        <td>INT8、UINT8、FLOAT16、FLOAT32、INT32、BFLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT8_E8M0、HIFLOAT8</td>
        <td>ND</td>
        <td>2-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>axis（int64_t）</td>
        <td>输入</td>
        <td>用来scatter的维度。</td>
        <td>取值范围为(-data_rank, data_rank)，且axis不能为0。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
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

    - <term>Atlas 训练系列产品</term>：数据类型不支持UINT8、BFLOAT16。
    - <term>Ascend 950PR/Ascend 950DT</term>：FLOAT8_E4M3FN、FLOAT8_E5M2、FLOAT8_E8M0、HIFLOAT8等数据类型仅在该型号支持。

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
    <col style="width: 276px">
    <col style="width: 132px">
    <col style="width: 836px">
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
      <td>传入的data、indices或updates是空指针。</td>
      </tr>
      <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>data、indices或updates数据类型不在支持范围内。</td>
      </tr>
      <tr>
      <td>data、updates数据类型不一样。</td>
      </tr>
      <tr>
      <td>data、updates的维度数不一致。</td>
      </tr>
      <tr>
      <td>indices的维度不是零维，一维或二维。</td>
      </tr>
      <tr>
      <td>indices维度为0时，updates的第0轴不为1。</td>
      </tr>
      <tr>
      <td>data、indices、updates是空tensor。</td>
      </tr>
    </tbody>
    </table>

## aclnnInplaceScatterUpdate

- **参数说明**
  <table style="undefined;table-layout: fixed; width: 1244px"><colgroup>
      <col style="width: 200px">
      <col style="width: 162px">
      <col style="width: 882px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceScatterUpdateGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnInplaceScatterUpdate默认确定性实现。

- updates shape的0轴与indices shape的0轴一致。
- indices为0维时，updates shape的0轴为1。
- updates shape的0轴小于等于data shape的0轴。
- updates与data的shape，除axis轴和0轴以外，其余轴的shape均相同。
- 当indices shape为二维时，shape的1轴需要等于2。
- indices数据类型为INT32时，DtypeSize=4，为INT64时，DtypeSize=8，IndicesShapeSize为indices的shape乘积，需要使用的ub = IndicesShapeSize * DtypeSize + 224，当ub大于对应可以获取到的AI处理器版本总ub大小时，不支持。
- 当indices有重复时，重复位置的结果不保证。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_update.h"

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
  int64_t axis = -2;
  std::vector<int64_t> selfRefShape = {1, 1, 2, 8};
  std::vector<int64_t> indicesShape = {1};
  std::vector<int64_t> updatesShape = {1, 1, 1, 8};
  void* selfRefDeviceAddr = nullptr;
  void* indicesDeviceAddr = nullptr;
  void* updatesDeviceAddr = nullptr;
  aclTensor* selfRef = nullptr;
  aclTensor* indices = nullptr;
  aclTensor* updates = nullptr;
  std::vector<float> selfRefHostData = {1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> indicesHostData = {1};
  std::vector<float> updatesHostData = {3, 3, 3, 3, 3, 3, 3, 3};

  // 创建selfRef aclTensor
  ret = CreateAclTensor(selfRefHostData, selfRefShape, &selfRefDeviceAddr, aclDataType::ACL_FLOAT, &selfRef);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建indices aclTensor
  ret = CreateAclTensor(indicesHostData, indicesShape, &indicesDeviceAddr, aclDataType::ACL_INT64, &indices);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建updates aclTensor
  ret = CreateAclTensor(updatesHostData, updatesShape, &updatesDeviceAddr, aclDataType::ACL_FLOAT, &updates);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnInplaceScatterUpdate第一段接口
  ret = aclnnInplaceScatterUpdateGetWorkspaceSize(selfRef, indices, updates, axis, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdateGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnInplaceScatterUpdate第二段接口
  ret = aclnnInplaceScatterUpdate(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnInplaceScatterUpdate failed. ERROR: %d\n", ret); return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(selfRefShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), selfRefDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
  aclDestroyTensor(selfRef);
  aclDestroyTensor(indices);
  aclDestroyTensor(updates);

  // 7. 释放device 资源
  aclrtFree(selfRefDeviceAddr);
  aclrtFree(indicesDeviceAddr);
  aclrtFree(updatesDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```
