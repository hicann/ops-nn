# aclnnIndexAdd

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：支持
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

在指定维度上，根据给定的索引，将源张量中的值加到输入张量中对应位置的值上。

本算子为就地语义算子：计算直接累加到self的存储上，self在调用后将被原地改写（等价于self[index] += alpha × source的就地执行）；底层算子的输出与self的存储绑定（就地绑定），aclnn接口内部将结果拷贝至out交付。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnIndexAddGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnIndexAdd”接口执行计算。

```Cpp
aclnnStatus aclnnIndexAddGetWorkspaceSize(
 const aclTensor*  self,
 const int64_t     dim,
 const aclTensor*  index,
 const aclTensor*  source,
 const aclScalar*  alpha,
 aclTensor*        out,
 uint64_t*         workspaceSize,
 aclOpExecutor**   executor)
```

```Cpp
aclnnStatus aclnnIndexAdd(
 void*          workspace,
 uint64_t       workspaceSize,
 aclOpExecutor* executor,
 aclrtStream    stream)
```

## aclnnIndexAddGetWorkspaceSize

- **参数说明**

    <table style="undefined;table-layout: fixed; width: 1451px"><colgroup>
    <col style="width: 143px">
    <col style="width: 120px">
    <col style="width: 227px">
    <col style="width: 265px">
    <col style="width: 274px">
    <col style="width: 117px">
    <col style="width: 160px">
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
        <td>self（aclTensor*）</td>
        <td>输入</td>
        <td>输入张量。</td>
        <td>支持1-8维；0维self无合法dim取值（dim取值范围[-self.dim(), self.dim()-1]在0维时为空集），由第一段校验拒绝。</td>
        <td>FLOAT、FLOAT16、INT32、INT16、INT8、UINT8、DOUBLE、INT64、BOOL、BFLOAT16</td>
        <td>ND</td>
        <td>1-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim（int64_t）</td>
        <td>输入</td>
        <td>指定的维度。</td>
        <td>取值范围为[-self.dim(), self.dim()-1]。</td>
        <td>INT64</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index（aclTensor*）</td>
        <td>输入</td>
        <td>索引。</td>
        <td>index的shape大小和source在dim维度上的shape值需要相等。</td>
        <td>INT64、INT32</td>
        <td>ND</td>
        <td>0-1</td>
        <td>√</td>
      </tr>
      <tr>
        <td>source（aclTensor*）</td>
        <td>输入</td>
        <td>源张量。</td>
        <td>source的shape除dim维度之外，其他维度的值需要与self的shape相等。</td>
        <td>与self一致</td>
        <td>ND</td>
        <td>与self一致</td>
        <td>√</td>
      </tr>
      <tr>
        <td>alpha（aclScalar*）</td>
        <td>输入</td>
        <td>缩放因子。</td>
        <td>数据类型可转换成self与source推导后的数据类型。<br>浮点类型alpha与整型self的组合受理：alpha数值按向self数据类型转换的语义截断（如0.5按整型self转换为0）；alpha数值超出self数据类型可表示范围时（如整型self时alpha为inf、nan或超出表示范围的浮点值）行为无定义。<br>self为BOOL时，alpha仅支持BOOL类型（BOOL仅支持单向转换为其他数据类型，数值/浮点类型无法转换为BOOL）。</td>
        <td>标量类型（需可转换成self的数据类型，不支持复数类型）</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>输出aclTensor。</td>
        <td>底层算子将输出与self的存储就地绑定（self会被原地改写），接口内部将结果拷贝至out。</td>
        <td>与self一致</td>
        <td>ND</td>
        <td>与self一致</td>
        <td>√</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>传入的self、index、source、alpha、out是空指针。</td>
      </tr>
      <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>self、index、source、out的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td>self、source和out的数据类型不一致。</td>
      </tr>
      <tr>
      <td>dim的值不在[-self.dim(), self.dim()-1]范围内。</td>
      </tr>
      <tr>
      <td>self和source除了在维度dim上，其余维度上存在shape值不相等。</td>
      </tr>
      <tr>
      <td>index不为0-1维。</td>
      </tr>
      <tr>
      <td>index的shape大小和source在维度dim上的shape值不相等。</td>
      </tr>
      <tr>
      <td>out的shape和self的shape不相等。</td>
      </tr>
    </tbody>
    </table>

## aclnnIndexAdd

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnIndexAddGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnIndexAdd默认非确定性实现，支持通过aclrtSetSysParamOpt开启确定性。
- 就地语义
  - self在调用后被原地改写（见功能说明）；GE 图模式（GEIR）不在本算子支持范围内。
- index取值范围
  - 期望index取值范围在[0, self.shape[dim])范围内；接口第一段不校验index的取值范围，负值索引在kernel侧被守卫静默跳过（不参与累加），越界索引行为未定义（经InplaceIndexAdd路径执行时可能触发执行期异常）。
- 空Tensor
  - source为空Tensor时，接口将self拷贝至out（不进行累加）；self为空Tensor时，out为空Tensor。
- 容量限制
  - 算子32位快速路径要求计算涉及的元素规模不超过uint32表示域（2^32-1），超出时自动路由至64位实现，功能结果一致；实际可支持的张量规模上限取决于数据类型与Device内存。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_index_add.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); aclrtFree(*deviceAddr);
            *deviceAddr = nullptr; return ret);

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
  // 1.（固定写法）device/stream初始化，参考acl API手册
  // 根据自己的实际device填写deviceId
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> indexShape = {4};
  std::vector<int64_t> sourceShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* indexDeviceAddr = nullptr;
  void* sourceDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  void* workspaceAddr = nullptr;
  aclTensor* self = nullptr;
  aclTensor* index = nullptr;
  aclTensor* source = nullptr;
  aclScalar* alpha = nullptr;
  aclTensor* out = nullptr;
  std::vector<float> selfHostData = {0, 1, 2, 3, 4, 5, 6, 7};
  std::vector<int> indexHostData = {0, 1, 2, 3};
  std::vector<float> sourceHostData = {1, 1, 1, 2, 2, 2, 3, 3};
  std::vector<float> outHostData(8, 0);
  int64_t dim = 0;
  float alphaValue = 1.0f;
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int32_t exitCode = 0;
  int64_t resultSize = 0;
  std::vector<float> resultData;
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(self) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // 创建index aclTensor
  ret = CreateAclTensor(indexHostData, indexShape, &indexDeviceAddr, aclDataType::ACL_INT32, &index);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(index) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // 创建source aclTensor
  ret = CreateAclTensor(sourceHostData, sourceShape, &sourceDeviceAddr, aclDataType::ACL_FLOAT, &source);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(source) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("CreateAclTensor(out) failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // 创建alpha aclScalar
  alpha = aclCreateScalar(&alphaValue, aclDataType::ACL_FLOAT);
  CHECK_RET(alpha != nullptr, LOG_PRINT("create alpha failed.\n"); exitCode = -1; goto cleanup);

  // 3. 调用CANN算子库API，需要修改为具体的API名称
  // 调用aclnnIndexAdd第一段接口
  ret = aclnnIndexAddGetWorkspaceSize(self, dim, index, source, alpha, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAddGetWorkspaceSize failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); exitCode = -1;
              goto cleanup);
  }
  // 调用aclnnIndexAdd第二段接口
  ret = aclnnIndexAdd(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnIndexAdd failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);

  // 4.（固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); exitCode = -1;
            goto cleanup);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  resultSize = GetShapeSize(outShape);
  resultData.assign(resultSize, 0.0f);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    resultSize * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            exitCode = -1; goto cleanup);
  for (int64_t i = 0; i < resultSize; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

cleanup:
  // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改（失败路径同样释放已创建对象）
  if (self != nullptr) {
    aclDestroyTensor(self);
  }
  if (index != nullptr) {
    aclDestroyTensor(index);
  }
  if (source != nullptr) {
    aclDestroyTensor(source);
  }
  if (alpha != nullptr) {
    aclDestroyScalar(alpha);
  }
  if (out != nullptr) {
    aclDestroyTensor(out);
  }

  // 7. 释放device资源，需要根据具体API的接口定义修改（失败路径同样释放已申请资源）
  if (selfDeviceAddr != nullptr) {
    aclrtFree(selfDeviceAddr);
  }
  if (indexDeviceAddr != nullptr) {
    aclrtFree(indexDeviceAddr);
  }
  if (sourceDeviceAddr != nullptr) {
    aclrtFree(sourceDeviceAddr);
  }
  if (outDeviceAddr != nullptr) {
    aclrtFree(outDeviceAddr);
  }
  if (workspaceAddr != nullptr) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return exitCode;
}
```
