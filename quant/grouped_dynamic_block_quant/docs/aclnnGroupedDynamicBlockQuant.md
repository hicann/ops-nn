# aclnnGroupedDynamicBlockQuant

## 产品支持情况

| 产品                                                         | 是否支持 |
| :----------------------------------------------------------- | :------: |
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×     |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×     |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×     |
| <term>Atlas 推理系列产品</term>                             |    ×     |
| <term>Atlas 训练系列产品</term>                              |    ×     |

## 功能说明

- 接口功能：根据传入的分组索引的起始值（groupList）对各个group以基本块的粒度进行量化，量化为（FP8/HiFP8），并输出量化参数scale（FP32）。

- 计算公式：
  $$
   input\_max = block\_reduce\_max(abs(input))
  $$
  $$
   scale = min(input\_max/FP8\_MAX(HiF8\_MAX, 1/min\_scale))
  $$
  $$
   y = cast\_to\_[HiF8/FP8](input/scale)
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnGroupedDynamicBlockQuantGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnGroupedDynamicBlockQuant”接口执行计算。

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(
  const aclTensor *x, 
  const aclTensor *groupList, 
  double           minScale, 
  char            *roundModeOptional, 
  int64_t          dstType, 
  int64_t          rowBlockSize, 
  int64_t          colBlockSize, 
  int64_t          groupListType, 
  const aclTensor *yOut, 
  const aclTensor *scaleOut, 
  uint64_t        *workspaceSize, 
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnGroupedDynamicBlockQuant(
  void          *workspace, 
  uint64_t       workspaceSize, 
  aclOpExecutor *executor, 
  aclrtStream    stream)
```

## aclnnGroupedDynamicBlockQuantGetWorkspaceSize

- **参数说明：**
  <table style="undefined;table-layout: fixed; width: 1401px"><colgroup>
  <col style="width: 149px">
  <col style="width: 120px">
  <col style="width: 266px">
  <col style="width: 280px">
  <col style="width: 200px">
  <col style="width: 100px">
  <col style="width: 105px">
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
      <td>x</td>
      <td>输入</td>
      <td>表示算子输入的Tensor。对应公式中的input。</td>
      <td>支持空Tensor。</td>
      <td>FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>2-3，形如[M, N]和[B, M, N]</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupList</td>
      <td>输入</td>
      <td>表示在M轴上每个group的偏移（cumsum模式）。</td>
      <td>表示量化分组的起始索引，要求大于等于0，且非递减，并且最后一个数需要与x的-2轴大小相等。</td>
      <td>INT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>minScale</td>
      <td>输入</td>
      <td>表示参与scaleOut计算的最小scale值。对应公式中的min_scale。</td>
      <td>要求该值大于等于0。</td>
      <td>FLOAT32</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>roundModeOptional</td>
      <td>输入</td>
      <td>表示最后由高bit数据cast到目标数据类型的近似模式。</td>
      <td>当dstType为35/36时，对应输出yOut数据类型为FLOAT8_E5M2/FLOAT8_E4M3FN时，仅支持{"rint"}；<br>当dstType为34时，对应输出yOut数据类型为HIFLOAT8时，支持{"round"、"hybrid"}；<br>传入空指针时，采用"rint"模式。</td>
      <td>STRING</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dstType</td>
      <td>输入</td>
      <td>表示数据转换后yOut的数据类型。</td>
      <td>输入范围为{34, 35, 36}，分别对应输出y的数据类型为{34:HIFLOAT8, 35: FLOAT8_E5M2, 36: FLOAT8_E4M3FN}。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>rowBlockSize</td>
      <td>输入</td>
      <td>表示指定M轴上的量化粒度。</td>
      <td>当前支持取值为1/128/256/512。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>colBlockSize</td>
      <td>输入</td>
      <td>表示指定N轴上的量化粒度。</td>
      <td>当前支持取值64/128/192/256。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groupListType</td>
      <td>输入</td>
      <td>表示group_list的功能类型。</td>
      <td>当前支持取值为0，对应cumsum模式。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut</td>
      <td>输出</td>
      <td>表示量化后的输出Tensor。对应公式中的y。</td>
      <td>支持空Tensor。<br>shape的维度与x保持一致。</td>
      <td>HIFLOAT8、FLOAT8_E4M3FN、FLOAT8_E5M2</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut</td>
      <td>输出</td>
      <td>表示每个分组对应的量化尺度，对应公式中的scale。</td>
      <td>支持空Tensor。<br>如果输入x的shape为[M, N]，groupList的shape为[g]，则输出scaleOut的shape维度为[(M//rowBlockSize+g), (N/colBlockSize)]。</br>如果输入x的shape为[B, M, N]，groupList的shape为[g]，则输出scaleOut的shape维度为[B, (M//rowBlockSize+g), (N/colBlockSize)]。 </td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2-3</td>
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
  </tbody></table>
   
  
- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1166px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 860px">
  </colgroup>
  <thead>
    <tr>
      <th>返回码</th>
      <th>错误码</th>
      <th>描述</th>
    </tr></thead>
  <tbody>
    <tr>
      <td>ACLNN_ERR_PARAM_NULLPTR</td>
      <td>161001</td>
      <td>传入的x、groupList、yOut或scaleOut的参数是空指针。</td>
    </tr>
    <tr>
      <td rowspan="5">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>输入或输出数据格式或数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入或输出数据的shape不在支持的范围之内。</td>
    </tr>
  </tbody>
  </table>

## aclnnGroupedDynamicBlockQuant

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnGroupedDynamicBlockQuantGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

 - 确定性说明：aclnnGroupedDynamicBlockQuant默认确定性实现。


## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。
```Cpp
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_grouped_dynamic_block_quant.h"

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

  void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
      LOG_PRINT("result[%ld] is: %d\n", i, resultData[i]);
    }
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
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType, aclTensor** tensor) {
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
    std::vector<int64_t> xShape = {4, 2};
    std::vector<int64_t> groupListShape = {1};
    std::vector<int64_t> yShape = {4, 2};
    std::vector<int64_t> scaleShape = {5, 1};

    void* xDeviceAddr = nullptr;
    void* groupListDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* x = nullptr;
    aclTensor* groupList = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    std::vector<aclFloat16> xHostData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int32_t> groupListHostData = {1};
    std::vector<uint8_t> yHostData(8, 0);
    std::vector<float> scaleHostData = {0, 0, 0, 0, 0};

    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建groupList aclTensor
    ret = CreateAclTensor(groupListHostData, groupListShape, &groupListDeviceAddr, aclDataType::ACL_INT32, &groupList);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_FLOAT8_E5M2, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建scale aclTensor
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的Api名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    const char* roundMode = "rint";
    float minScale = 0.0;
    int64_t rowBlockSize = 1;
    int64_t colBlockSize = 128;
    int64_t groupListType = 0;

    // 调用aclnnGroupedDynamicBlockQuant第一段接口
    ret = aclnnGroupedDynamicBlockQuantGetWorkspaceSize(x, groupList, minScale, (char *)roundMode, aclDataType::ACL_FLOAT8_E5M2, rowBlockSize, colBlockSize, groupListType, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuantGetWorkspaceSize failed. ERROR: %d\n", ret); 
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
      ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); 
                return ret);
    }

    // 调用aclnnGroupedDynamicBlockQuant第二段接口
    ret = aclnnGroupedDynamicBlockQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnGroupedDynamicBlockQuant failed. ERROR: %d\n", ret); 
              return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); 
              return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    LOG_PRINT("yOut is: \n");
    PrintOutResult(yShape, &yDeviceAddr);
    LOG_PRINT("scaleOut is: \n");
    PrintOutResult(scaleShape, &scaleDeviceAddr);

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(groupList);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. 释放device资源
    aclrtFree(xDeviceAddr);
    aclrtFree(groupListDeviceAddr);
    aclrtFree(yDeviceAddr);
    aclrtFree(scaleDeviceAddr);
    if (workspaceSize > 0) {
      aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
  }
```