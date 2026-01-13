# aclnnNLLLoss2dBackward



## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    √     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    √     |

## 功能说明

接口功能：负对数似然损失反向。

## 函数原型

  每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnNLLLoss2dBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnNLLLoss2dBackward”接口执行计算。

```cpp
aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(
  const aclTensor *gradOutput,
  const aclTensor *self,
  const aclTensor *target,
  const aclTensor *weight,
  int64_t          reduction,
  int64_t          ignoreIndex,
  aclTensor       *totalWeight,
  aclTensor       *out,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```cpp
aclnnStatus aclnnNLLLoss2dBackward(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnNLLLoss2dBackwardGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1349px"><colgroup>
      <col style="width: 158px">
      <col style="width: 120px">
      <col style="width: 253px">
      <col style="width: 283px">
      <col style="width: 218px">
      <col style="width: 110px">
      <col style="width: 102px">
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
        <td>gradOutput</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td>shape为三维（第一维是N）或者一维（且元素个数为1）。</td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>self</td>
        <td>输入</td>
        <td>输入aclTensor。</td>
        <td><ul><li>shape为四维，第一维是N表示batch size，第二维是C表示类别。</li><li>要求self的第0维、第2维、第3维的shape分别与target的第0维、第1维、第2维的shape一致，否则返回false。</li></ul></td>
        <td>FLOAT、FLOAT16、BFLOAT16</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>target</td>
        <td>输入</td>
        <td>表示真实标签。</td>
        <td><ul><li>公式中的y，shape为3维</li><li>target的第1维与self的第1维相等、target的第2维与self的第3维相等、target的第3维与self的第4维相等</li><li>其中每个元素的取值范围是[0, C - 1]</li></ul></td>
        <td>INT64、UINT8、INT32</td>
        <td>ND</td>
        <td>-</td>
        <td>√</td>
      </tr>
      <tr>
        <td>weight</td>
        <td>输入</td>
        <td>表示每个类别的缩放权重。</td>
        <td>公式中的w，shape为(C,)。</td>
        <td>数据类型和self保持一致。</td>
        <td>ND</td>
        <td>(C)</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduction</td>
        <td>输入</td>
        <td>指定要应用到输出的缩减。</td>
        <td><ul>支持 0('none') | 1('mean') | 2('sum')。<li>'none'表示不应用缩减</li><li>'mean'表示输出的总和将除以输出中的元素数</li><li>'sum'表示输出将被求和</li><li>当reduction为0时，要求target的shape与gradOutput的shape一致，否则返回false。</li></ul></td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>ignoreIndex</td>
        <td>输入</td>
         <td>指定一个被忽略且不影响输入梯度的目标值。
        </td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
          <tr>
        <td>totalWeight</td>
        <td>输入</td>
        <td>公式中的totalWeight。</td>
        <td>仅当reduction为mean时，totalWeight是通过target取相应位置的weight，然后去除掉ignoreIndex对应的weight，将剩下的weight求和；当reduction为其他值时，该参数默认不处理。</td>
        <td>数据类型与weight相同。</td>
        <td>ND</td>
        <td>(1，)</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out</td>
        <td>输出</td>
        <td>公式中的out。</td>
        <td>shape与self相同。</td>
        <td>数据类型和self一致。</td>
        <td>ND</td>
        <td>-</td>
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

- **返回值**

  aclnnStatus: 返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

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
      <td>传入的gradOutput、self、target、weight、totalWeight或out是空指针。</td>
      </tr>
      <tr>
      <td rowspan="9">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="9">161002</td>
      <td> gradOutput、self、target、weight、totalWeight或out的数据类型不在支持的范围之内。</td>
      </tr>
      <tr>
      <td> gradOutput、self、weight、totalWeight、out的数据类型不一致。</td>
      </tr>
      <tr>
      <td> target非3维tensor，self非4维tensor。</td>
      </tr>
       <tr>
      <td>weight的元素个数不是C。</td>
      </tr>
      <tr>
      <td>self的第0，2，3维的元素个数和target的第0，1，2维元素个数不相等。</td>
      </tr>
      <tr>
      <td> totalWeight的元素个数不是1。</td>
      </tr>
      <tr>
      <td>reduction是none时，gradOutput的维数不是3或者gradOutput的第0，1，2维的元素个数和target的第0，1，2维元素个数不相等。</td>
      </tr>
      <tr>
      <td>reduction不是none时，gradOutput的维数大于1或者元素个数不为1。</td>
      </tr>
      <tr>
      <td>reduction值不在0~2范围之内。</td>
      </tr>
      </tbody>
      </table>


## aclnnNLLLoss2dBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnNLLLossGetWorkspaceSize获取。</td>
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
    - aclnnNLLLoss2dBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_nll_loss2d_backward.h"

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
  std::vector<int64_t> gradShape = {3, 1, 1};
  std::vector<int64_t> selfShape = {3, 5, 1, 1};
  std::vector<int64_t> targetShape = {3, 1, 1};
  std::vector<int64_t> weightShape = {5};
  std::vector<int64_t> totalWeightShape = {1};
  std::vector<int64_t> outShape = {3, 5, 1, 1};

  void* gradDeviceAddr = nullptr;
  void* selfDeviceAddr = nullptr;
  void* targetDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* totalWeightDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* grad = nullptr;
  aclTensor* self = nullptr;
  aclTensor* target = nullptr;
  aclTensor* weight = nullptr;
  aclTensor* totalWeight = nullptr;
  aclTensor* out = nullptr;

  std::vector<float> gradHostData = {2.7, 2.6, 2.5};
  std::vector<float> selfHostData = {4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5};
  std::vector<int64_t> targetHostData = {2, 3, 1};
  std::vector<float> weightHostData = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<float> totalWeightHostData = {1.0};
  std::vector<float> outHostData = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int64_t reduction = 0;
  int64_t ignoreIndex = -100;

  // 创建grad aclTensor
  ret = CreateAclTensor(gradHostData, gradShape, &gradDeviceAddr, aclDataType::ACL_FLOAT, &grad);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建self aclTensor
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建target aclTensor
  ret = CreateAclTensor(targetHostData, targetShape, &targetDeviceAddr, aclDataType::ACL_INT64, &target);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建weight aclTensor
  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建totalWeight aclTensor
  ret = CreateAclTensor(totalWeightHostData, totalWeightShape, &totalWeightDeviceAddr,
                        aclDataType::ACL_FLOAT, &totalWeight);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  // 调用aclnnNLLLoss2dBackward第一段接口
  ret = aclnnNLLLoss2dBackwardGetWorkspaceSize(grad, self, target, weight, reduction, ignoreIndex, totalWeight, out,
                                               &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }
  // 调用aclnnNLLLoss2dBackward第二段接口
  ret = aclnnNLLLoss2dBackward(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnNLLLoss2dBackward failed. ERROR: %d\n", ret); return ret);

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

  // 6. 释放aclTensor，需要根据具体API的接口定义修改
  aclDestroyTensor(grad);
  aclDestroyTensor(self);
  aclDestroyTensor(target);
  aclDestroyTensor(weight);
  aclDestroyTensor(totalWeight);
  aclDestroyTensor(out);

  // 7. 释放device 资源
  aclrtFree(gradDeviceAddr);
  aclrtFree(selfDeviceAddr);
  aclrtFree(targetDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(totalWeightDeviceAddr);
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

