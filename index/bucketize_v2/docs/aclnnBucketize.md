# aclnnBucketize

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/index/bucketize_v2)

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------- | ------|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：根据给定的边界数组（boundaries）确定输入张量中每个元素所属的区间索引。对于输入值 $x$ 和边界数组 $boundaries = [b_0, b_1, ..., b_{n-1}]$，输出索引 $y_i$ 满足：
  - 如果 $x < b_0$，则 $y_i = 0$
  - 如果 $b_{i-1} \le x < b_i$，则 $y_i = i$（默认左开右闭区间）
  - 如果 $x \ge b_{n-1}$，则 $y_i = n$
- 计算公式：
  
  当right=False时（左闭右开区间）：
  $$
  y_i = \min\{j \mid x_i < b_j\}
  $$
  
  当right=True时（左开右闭区间）：
  $$
  y_i = \min\{j \mid x_i \le b_j\}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnBucketizeGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnBucketize"接口执行计算。

```cpp
aclnnStatus aclnnBucketizeGetWorkspaceSize(
  const aclTensor  *self,
  const aclTensor  *boundaries,
  const bool        outInt32,
  const bool        right,
  aclTensor        *out,
  uint64_t         *workspaceSize,
  aclOpExecutor    **executor)
```

```cpp
aclnnStatus aclnnBucketize(
  void           *workspace,
  uint64_t        workspaceSize,
  aclOpExecutor  *executor,
  aclrtStream     stream)
```

## aclnnBucketizeGetWorkspaceSize

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
      <td>输入张量，待确定区间索引的数据。</td>
      <td><ul><li>支持空Tensor。</li><li>数据类型与boundaries的数据类型需满足互推导关系。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8</td>
      <td>ND</td>
      <td>0-8</td>
      <td>√</td>
    </tr>
    <tr>
      <td>boundaries（aclTensor*）</td>
      <td>输入</td>
      <td>边界数组，必须是一维张量且单调递增。</td>
      <td><ul><li>不支持空Tensor。</li><li>必须为一维张量。</li><li>必须单调递增且不重复。</li></td>
      <td>FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outInt32（bool）</td>
      <td>输入</td>
      <td>输出数据类型控制参数。</td>
      <td><ul><li>True表示输出数据类型为INT32。</li><li>False表示输出数据类型为INT64。</li></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>right（bool）</td>
      <td>输入</td>
      <td>区间边界控制参数。</td>
      <td><ul><li>True表示区间包含右边界（左开右闭）。</li><li>False表示区间不包含右边界（左闭右开）。</li></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出张量，每个元素为对应的区间索引。</td>
      <td><ul><li>支持空Tensor。</li><li>shape必须与self相同。</li><li>数据类型由outInt32决定。</li></td>
      <td>INT32、INT64</td>
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
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 650px">
  </colgroup>
  <thead>
    <tr>
      <th>返回值</th>
      <th>错误码</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> ACLNN_ERR_PARAM_NULLPTR </td>
      <td> 161001 </td>
      <td>self、boundaries、out存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="5"> ACLNN_ERR_PARAM_INVALID </td>
      <td rowspan="5"> 161002 </td>
      <td>self或boundaries的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>self和boundaries的数据类型不满足互推导关系。</td>
    </tr>
    <tr>
      <td>boundaries的维度不为1。</td>
    </tr>
    <tr>
      <td>out的shape与self的shape不匹配。</td>
    </tr>
    <tr>
      <td>out的数据类型与outInt32参数不匹配。</td>
    </tr>  
  </tbody></table>

## aclnnBucketize

- **参数说明**
  
  <table style="undefined;table-layout: fixed; width: 1100px"><colgroup>
   <col style="width: 200px">
   <col style="width: 130px">
   <col style="width: 770px">
  </colgroup>
  <thead>
            <tr><th>参数名</th><th>输入/输出</th><th>描述</th></tr>
        </thead>
        <tbody>
            <tr><td>workspace</td><td>输入</td><td>在Device侧申请的workspace内存地址。</td></tr>
            <tr><td>workspaceSize </td><td>输入</td><td>在Device侧申请的workspace大小，由第一段接口aclnnBucketizeGetWorkspaceSize获取。</td></tr>
            <tr><td>executor</td><td>输入</td><td> op执行器，包含了算子计算流程。 </td></tr>
            <tr><td>stream </td><td>输入</td><td> 指定执行任务的Stream。 </td></tr>
        </tbody>
    </table>

- **返回值**
  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- 确定性说明：aclnnBucketize默认确定性实现。
- boundaries必须是一维张量。
- boundaries必须单调递增且不重复（软件层面无法校验，需用户保证）。
- self和boundaries的数据类型需要满足互推导关系。
- out的shape必须与self相同。
- outInt32为True时，out的数据类型必须为INT32；outInt32为False时，out的数据类型必须为INT64。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

- <term>Ascend 950PR/Ascend 950DT</term>：
  
  ```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_bucketize.h"

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
      auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

      ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

      std::vector<int64_t> strides(shape.size(), 1);
      for (int64_t i = shape.size() - 2; i >= 0; i--) {
         strides[i] = shape[i + 1] * strides[i + 1];
      }

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
      return 0;
    }

   int main() {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
      
      std::vector<int64_t> selfShape = {3, 2};
      std::vector<int64_t> boundariesShape = {3};
      std::vector<int64_t> outShape = {3, 2};
      void* selfDeviceAddr = nullptr;
      void* boundariesDeviceAddr = nullptr;
      void* outDeviceAddr = nullptr;
      aclTensor* self = nullptr;
      aclTensor* boundaries = nullptr;
      aclTensor* out = nullptr;
      
      std::vector<float> selfHostData = {-5, 10000, 150, 10, 5, 100};
      std::vector<float> boundariesHostData = {0, 10, 100};
      std::vector<int64_t> outHostData = {0, 0, 0, 0, 0, 0};
      
      ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      
      ret = CreateAclTensor(boundariesHostData, boundariesShape, &boundariesDeviceAddr, aclDataType::ACL_FLOAT, &boundaries);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      
      ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_INT64, &out);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor;
      
      bool outInt32 = false;
      bool right = false;
      
      ret = aclnnBucketizeGetWorkspaceSize(self, boundaries, outInt32, right, out, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBucketizeGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
      
      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
         CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
      }
      
      ret = aclnnBucketize(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnBucketize failed. ERROR: %d\n", ret); return ret);
      
      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
      
      auto size = GetShapeSize(outShape);
      std::vector<int64_t> resultData(size, 0);
      ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(int64_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
      for (int64_t i = 0; i < size; i++) {
          LOG_PRINT("result[%ld] is: %ld\n", i, resultData[i]);
      }

      aclDestroyTensor(self);
      aclDestroyTensor(boundaries);
      aclDestroyTensor(out);

      aclrtFree(selfDeviceAddr);
      aclrtFree(boundariesDeviceAddr);
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

## 参考

### Pytorch AtenIR

```
- func: bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool outInt32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
  dispatch:
    CPU: bucketize_out_cpu
    CUDA: bucketize_out_cuda
```

### AtenIR参数描述

| 类型 | 名称 | 描述 | GPU支持的数据类型 | CPU支持的数据类型 |
| ---- | ------ | ---------------------------- | ----------------------------------------------------------------------------------------------- |---|
| 输入 | input | 输入张量 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、DOUBLE | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、DOUBLE|
| 输入 | boundaries | 边界数组，必须是一维且单调递增 | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、DOUBLE | FLOAT、FLOAT16、BFLOAT16、INT8、INT16、INT32、INT64、UINT8、DOUBLE|
| 属性 | outInt32 | 输出数据类型是否为int32 | bool | bool |
| 属性 | right | 区间是否包含右边界 | bool | bool |
| 输入 | out | 指定的输出tensor | INT32、INT64 |INT32、INT64|
| 输出 | output | 输出tensor | INT32、INT64 |INT32、INT64|
