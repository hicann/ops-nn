# aclnnShrink

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
- <term>Atlas推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：对输入张量进行非线性变换，根据输入值self与阈值lambd的关系，对输入通过偏移量bias进行缩放和偏移处理。

- 计算公式：

  $$
  out=\begin{cases}
  self-bias,  self > lambd \\
  self+bias,  self < -lambd \\
  0,  -lambd ≤self ≤lambd\\
  \end{cases}
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnShrinkGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnShrink”接口执行计算。

```Cpp
aclnnStatus aclnnShrinkGetWorkspaceSize(
  const aclTensor* self,
  const aclScalar* lambd,
  const aclScalar* bias,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnShrink(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnShrinkGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1380px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 200px">
  <col style="width: 230px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>表示待转换的目标张量，公式中的self。</td>
      <td><ul><li>支持空Tensor。</li><li>支持动态 shape（-1 未知维）与动态秩（-2）标记，图模式下按输入透传，运行期按最终具体形状校验。</li><li>元素总数受 int64 与设备内存约束。</li><li>NaN 输入按死区输出 0（不传播 NaN）；±inf 输入分别按正区/负区处理，输出为 x∓bias 的 IEEE 算术结果（与 bias=±inf 组合时为 inf∓inf=NaN）。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>ND；支持空 Tensor 与 -1/-2 动态标记</td>
      <td>√</td>
    </tr>
     <tr>
      <td>lambd（aclScalar*）</td>
      <td>输入</td>
      <td>表示转换过程中用于判断self大小的阈值，公式中的lambd。dtype须为FLOAT（非FLOAT返回161002）；FLOAT16 输入时 lambd 先转为输入 dtype（`static_cast&lt;T&gt;`）再参与比较，阈值附近与实数公式可能不同（如 x=fp16(0.5009765625)、lambd=0.5008 时实数公式判正区、设备判死区输出 0）。取值为非负数（NaN 及负数含 -inf 返回161002——NaN 非数、不属非负数域）；lambd=+inf 时全部输入按死区输出 0（含 ±inf 输入，严格比较 inf>inf 为假）。</td>
      <td>取值为非负数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>bias（aclScalar*）</td>
      <td>输入</td>
      <td>表示转换过程中添加到目标张量的偏差值，公式中的bias。dtype须为FLOAT（非FLOAT返回161002）；FLOAT16 输入时 bias 同样先转为输入 dtype 后参与加减。bias=±inf 时活跃区按 IEEE 算术计算：有限输入输出相应 ±inf，±inf 输入构成 inf∓inf 时输出 NaN；死区仍输出 0。bias 为 NaN 时返回161002（NaN 非实数，不属任意实数域）。</td>
      <td>取值为任意实数。</td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>输出目标张量，公式中的out。</td>
      <td><ul><li>支持空Tensor。</li><li>shape需要与self一致。</li><li>dtype需要与self一致。</li><li>格式需要与self一致（不一致返回161002）。</li></ul></td>
      <td>FLOAT、FLOAT16</td>
      <td>ND</td>
      <td>ND，shape 与 self 一致</td>
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
  </tbody>
  </table>

- **返回值：**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 979px"><colgroup>
  <col style="width: 272px">
  <col style="width: 103px">
  <col style="width: 604px">
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
      <td>传入的self、lambd、bias或out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="7">161002</td>
      <td>self、lambd或out的数据类型不在支持的范围之内（lambd、bias仅支持FLOAT标量）。</td>
    </tr>
    <tr>
      <td>self和out的shape不一致。</td>
    </tr>
    <tr>
      <td>self和out的dtype不一致。</td>
    </tr>
    <tr>
      <td>self和out的格式不一致。</td>
    </tr>
    <tr>
      <td>self或out使用私有存储格式（如FRACTAL_NZ；本算子仅支持ND）。</td>
    </tr>
    <tr>
      <td>lambd值为负数。</td>
    </tr>
    <tr>
      <td>lambd或bias为NaN（NaN 非数，不属声明的取值域）。</td>
    </tr>
  </tbody></table>

## aclnnShrink

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnShrinkGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnShrink默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cstdlib>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_shrink.h"

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

int main(int argc, char** argv) {
  // 1. （固定写法）device/stream初始化，参考acl API手册
  // deviceId 可由命令行参数指定（如 ./test_aclnn_shrink 6），缺省为 0；
  // 亦可用环境变量 ASCEND_RT_VISIBLE_DEVICES 把逻辑设备映射到物理设备
  int32_t deviceId = (argc > 1) ? static_cast<int32_t>(strtol(argv[1], nullptr, 10)) : 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  // check根据自己的需要处理
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  // 2. 构造输入与输出，需要根据API的接口自定义构造
  std::vector<int64_t> selfShape = {4, 2};
  std::vector<int64_t> outShape = {4, 2};
  void* selfDeviceAddr = nullptr;
  void* outDeviceAddr = nullptr;
  aclTensor* self = nullptr;
  aclScalar* lambd = nullptr;
  aclScalar* bias = nullptr;
  aclTensor* out = nullptr;
  void* workspaceAddr = nullptr;
  uint64_t workspaceSize = 0;
  std::vector<float> selfHostData = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
  std::vector<float> outHostData = {0, 0, 0, 0, 0, 0, 0, 0};
  float lambdValue = 0.5f;
  float biasValue = 0.0;
  // 创建self aclTensor（失败路径经 CleanupResources 统一释放已创建资源）
  ret = CreateAclTensor(selfHostData, selfShape, &selfDeviceAddr, aclDataType::ACL_FLOAT, &self);
  CHECK_RET(ret == ACL_SUCCESS,
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);
  // 创建lambd aclScalar
  lambd = aclCreateScalar(&lambdValue, aclDataType::ACL_FLOAT);
  CHECK_RET(lambd != nullptr, LOG_PRINT("aclCreateScalar(lambd) failed.\n");
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return 1);
  // 创建bias aclScalar
  bias = aclCreateScalar(&biasValue, aclDataType::ACL_FLOAT);
  CHECK_RET(bias != nullptr, LOG_PRINT("aclCreateScalar(bias) failed.\n");
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return 1);
  // 创建out aclTensor
  ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
  CHECK_RET(ret == ACL_SUCCESS,
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);

  // 3. 调用CANN算子库API，需要修改为具体的Api名称
  aclOpExecutor* executor;
  // 调用aclnnShrink第一段接口
  ret = aclnnShrinkGetWorkspaceSize(self, lambd, bias, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnShrinkGetWorkspaceSize failed. ERROR: %d\n", ret);
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);
  // 根据第一段接口计算出的workspaceSize申请device内存
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret);
              CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                               stream, deviceId);
              return ret);
  }
  // 调用aclnnShrink第二段接口
  ret = aclnnShrink(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnShrink failed. ERROR: %d\n", ret);
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);

  // 4. （固定写法）同步等待任务执行结束
  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret);
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);

  // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
  auto size = GetShapeSize(outShape);
  std::vector<float> resultData(size, 0);
  ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr,
                    size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret);
            CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize,
                             stream, deviceId);
            return ret);
  for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
  }

  // 6. 正常路径：释放全部资源
  CleanupResources(self, lambd, bias, out, selfDeviceAddr, outDeviceAddr, workspaceAddr, workspaceSize, stream,
                   deviceId);
  return 0;
}
```
