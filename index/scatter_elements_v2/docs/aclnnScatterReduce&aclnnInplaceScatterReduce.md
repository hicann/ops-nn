# aclnnScatterReduce&aclnnInplaceScatterReduce


## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950PR/Ascend 950DT</term> | × |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |
| <term>Atlas 200I/500 A2 推理产品</term> | × |
| <term>Atlas 推理系列产品</term> | × |
| <term>Atlas 训练系列产品</term> | × |

## 功能说明

- 接口功能：将 tensor `src` 中的值按指定的轴和 `index` 指定的位置关系写入或规约到 tensor `self` 中。
- 支持的规约语义：
  - `none`：替换写入
  - `add`：累加
  - `mul`：累乘
  - `max`：取最大值
  - `min`：取最小值
  - `mean`：求平均值
- `include_self` 语义：
  - `include_self=true`：规约时将 `self` 原始值参与计算。
  - `include_self=false`：规约时不将 `self` 原始值作为初始值参与计算；首次命中时以第一个 `src` 值作为初值。
- 接口职责边界：
  - `aclnnScatterReduce/aclnnInplaceScatterReduce` 统一承接所有显式需要 `include_self` 的规约场景。
  - `aclnnScatterAdd` 是固定 `reduce=add` 且固定 `includeSelf=true` 的快捷接口，不再承接显式 `include_self` 传参场景。
  - 若仅需要传统 `scatter/scatter_value` 语义且不显式暴露 `include_self`，使用 `aclnnScatter/aclnnInplaceScatter/aclnnScatterValue/aclnnInplaceScatterValue`。

- 示例：
  对于一个 3D tensor，`self` 会按照如下规则进行更新：

  ```text
  out[index[i][j][k]][j][k] = src[i][j][k]                                 # 如果 dim == 0 && reduce == 0(none)
  out[i][index[i][j][k]][k] += src[i][j][k]                                # 如果 dim == 1 && reduce == 1(add)
  out[i][j][index[i][j][k]] = max(out[i][j][index[i][j][k]], src[i][j][k]) # 如果 dim == 2 && reduce == 3(max)
  ```

- 在计算时需要满足以下要求：
  - `self`、`index` 和 `src` 的维度数量必须相同。
  - 对于每一个维度 `d`，有 `index.size(d) <= src.size(d)` 的限制。
  - 对于每一个维度 `d`，如果 `d != dim`，有 `index.size(d) <= self.size(d)` 的限制。
  - `dim` 的值必须在 `[-self 的维度数量, self 的维度数量-1]` 之间。
  - `self` 的维度数应该小于等于 8。
  - `index` 中对应维度 `dim` 的值必须在 `[0, self.size(dim)-1]` 之间。

## 函数原型

aclnnScatterReduce 和 aclnnInplaceScatterReduce 实现相同的规约 scatter 功能，使用区别如下，请根据自身实际场景选择合适的算子。

- `aclnnScatterReduce`：需新建一个输出张量对象存储计算结果。
- `aclnnInplaceScatterReduce`：无需新建输出张量对象，直接在输入张量的内存中存储计算结果。

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnScatterReduceGetWorkspaceSize”或者“aclnnInplaceScatterReduceGetWorkspaceSize”接口获取计算所需 workspace 大小以及包含了算子计算流程的执行器，再调用“aclnnScatterReduce”或者“aclnnInplaceScatterReduce”接口执行计算。

```Cpp
aclnnStatus aclnnScatterReduceGetWorkspaceSize(
  const aclTensor* self,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  bool             includeSelf,
  aclTensor*       out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnScatterReduce(
  void*               workspace,
  uint64_t            workspaceSize,
  aclOpExecutor*      executor,
  const aclrtStream   stream)
```

```Cpp
aclnnStatus aclnnInplaceScatterReduceGetWorkspaceSize(
  aclTensor*       selfRef,
  int64_t          dim,
  const aclTensor* index,
  const aclTensor* src,
  int64_t          reduce,
  bool             includeSelf,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnInplaceScatterReduce(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnScatterReduceGetWorkspaceSize

- **参数说明：**

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
        <td>公式中的`self`。</td>
        <td>数据类型需要与src一致。<br>维度数量需要与index、src相同。</td>
        <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim（int64_t）</td>
        <td>输入</td>
        <td>用来scatter的维度。</td>
        <td>范围为[-self的维度数量, self的维度数量-1]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index（aclTensor*）</td>
        <td>输入</td>
        <td>索引张量。</td>
        <td>维度数量需要与self、src相同。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src（aclTensor*）</td>
        <td>输入</td>
        <td>公式中的`src`。</td>
        <td>数据类型需要与self一致。<br>维度数量需要与self、index相同。</td>
        <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce（int64_t）</td>
        <td>输入</td>
        <td>选择应用的reduction操作。</td>
        <td>可选的操作选项以及对应的int值为 (none, 0)、(add, 1)、(mul, 2)、(max, 3)、(min, 4)、(mean, 5)。<br>0表示替换，1表示累加，2表示累乘，3表示取最大值，4表示取最小值，5表示求平均值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeSelf（bool）</td>
        <td>输入</td>
        <td>规约计算时是否将self中的原始值参与计算。</td>
        <td>true表示参与规约；false表示不参与规约。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>out（aclTensor*）</td>
        <td>输出</td>
        <td>计算结果。</td>
        <td>数据类型需要与self一致。<br>shape需要与self一致。</td>
        <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
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
        <td>self、index、src或out存在空指针。</td>
      </tr>
      <tr>
        <td rowspan="8">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="8">161002</td>
        <td>self、index、src或out的数据类型不在支持范围内。</td>
      </tr>
      <tr>
        <td>self、src、out的数据类型不一样。</td>
      </tr>
      <tr>
        <td>self、index、src的维度数不一致。</td>
      </tr>
      <tr>
        <td>self和out的shape不一致。</td>
      </tr>
      <tr>
        <td>dim的值不在[-self的维度数量， self的维度数量-1]之间。</td>
      </tr>
      <tr>
        <td>self的维度数超过8。</td>
      </tr>
      <tr>
        <td>shape不符合限制：对于每一个维度d，有index.size(d) <= src.size(d)；且d != dim时，index.size(d) <= self.size(d)。</td>
      </tr>
      <tr>
        <td>reduce不在支持范围内，必须属于{0, 1, 2, 3, 4, 5}。</td>
      </tr>
    </tbody></table>

## aclnnScatterReduce

- **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnScatterReduceGetWorkspaceSize获取。</td>
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

## aclnnInplaceScatterReduceGetWorkspaceSize

- **参数说明：**

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
        <td>selfRef（aclTensor*）</td>
        <td>输入/输出</td>
        <td>scatter_reduce的目标张量。</td>
        <td>shape需要与index和src的维度数量相同。<br>数据类型与src的数据类型一致。<br>支持空Tensor。</td>
        <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>dim（int64_t）</td>
        <td>输入</td>
        <td>指定沿哪个维度进行scatter_reduce操作。</td>
        <td>范围为[-selfRef的维度数量, selfRef的维度数量-1]。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>index（aclTensor*）</td>
        <td>输入</td>
        <td>索引张量。</td>
        <td>用于指定src张量中散布到selfRef张量中的位置。<br>支持空Tensor。</td>
        <td>INT32、INT64</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>src（aclTensor*）</td>
        <td>输入</td>
        <td>源张量。</td>
        <td>数据类型需要与selfRef一致。<br>支持空Tensor。</td>
        <td>UINT8、INT8、INT16、INT32、INT64、BOOL、FLOAT16、FLOAT32、DOUBLE、COMPLEX64、COMPLEX128、BFLOAT16</td>
        <td>ND</td>
        <td>0-8</td>
        <td>√</td>
      </tr>
      <tr>
        <td>reduce（int64_t）</td>
        <td>输入</td>
        <td>选择应用的reduction操作。</td>
        <td>可选操作：0表示替换，1表示累加，2表示累乘，3表示取最大值，4表示取最小值，5表示求平均值。</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
      </tr>
      <tr>
        <td>includeSelf（bool）</td>
        <td>输入</td>
        <td>规约计算时是否将selfRef中的原始值参与计算。</td>
        <td>true表示参与规约；false表示不参与规约。</td>
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
        <td>selfRef、index、src存在空指针。</td>
      </tr>
      <tr>
        <td rowspan="7">ACLNN_ERR_PARAM_INVALID</td>
        <td rowspan="7">161002</td>
        <td>selfRef、index、src数据类型不在支持范围内。</td>
      </tr>
      <tr>
        <td>selfRef、src数据类型不一样。</td>
      </tr>
      <tr>
        <td>selfRef、index、src的维度数不一致。</td>
      </tr>
      <tr>
        <td>dim的值不在[-selfRef的维度数量, selfRef的维度数量-1]之间。</td>
      </tr>
      <tr>
        <td>selfRef的维度数超过8。</td>
      </tr>
      <tr>
        <td>shape不符合限制：对于每一个维度d，有index.size(d) <= src.size(d)；且d != dim时，index.size(d) <= selfRef.size(d)。</td>
      </tr>
      <tr>
        <td>reduce不在支持范围内，必须属于{0, 1, 2, 3, 4, 5}。</td>
      </tr>
    </tbody></table>

## aclnnInplaceScatterReduce

- **参数说明：**

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
        <td>在Device侧申请的workspace大小，由第一段接口aclnnInplaceScatterReduceGetWorkspaceSize获取。</td>
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
  - 是否进入确定性路径取决于底层后端能力与上层 deterministic 配置。
  - 在 `Atlas A2/Atlas A3` 上，当 `TilingContext::GetDeterministic()` 为真，`reduce` 属于 `none/add/mul/min/max/mean` 时，会禁用 `NoTranspose` 和 `cache-op/low-memory`，收敛到 `ScatterElementsV2` legacy kernel 的单核调度路径。
  - 在 `Ascend 950PR` 上，deterministic 能力由独立的 tiling / kernel 分支承接，但不同 reduction 模式的支持范围与 `Atlas A2/Atlas A3` 并不完全一致。
- 接口边界约束：
  - 本页接口显式透传 `includeSelf`，是规约 scatter 的统一公共入口。
  - 若只需要固定 `add` 语义且不想透传 `includeSelf`，可使用 `aclnnScatterAdd`。
  - `aclnnScatter/aclnnInplaceScatter` 仍保留传统 `none/add/mul` 路径，但不显式透传 `include_self`。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

**aclnnScatterReduce示例代码：**

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_scatter_reduce.h"

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
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; --i) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream = nullptr;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> shape = {2, 3};
  std::vector<float> selfHost = {1, 2, 3, 4, 5, 6};
  std::vector<int64_t> indexHost = {0, 2, 1, 1, 0, 2};
  std::vector<float> srcHost = {10, 20, 30, 40, 50, 60};
  std::vector<float> outHost(6, 0);

  void *selfDev = nullptr, *indexDev = nullptr, *srcDev = nullptr, *outDev = nullptr, *workspace = nullptr;
  aclTensor *self = nullptr, *index = nullptr, *src = nullptr, *out = nullptr;

  CHECK_RET(CreateAclTensor(selfHost, shape, &selfDev, ACL_FLOAT, &self) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(indexHost, shape, &indexDev, ACL_INT64, &index) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(srcHost, shape, &srcDev, ACL_FLOAT, &src) == ACL_SUCCESS, return -1);
  CHECK_RET(CreateAclTensor(outHost, shape, &outDev, ACL_FLOAT, &out) == ACL_SUCCESS, return -1);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor = nullptr;
  int64_t dim = 1;
  int64_t reduce = 1;   // add
  bool includeSelf = true;

  ret = aclnnScatterReduceGetWorkspaceSize(self, dim, index, src, reduce, includeSelf, out, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
  }

  ret = aclnnScatterReduce(workspace, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  LOG_PRINT("aclnnScatterReduce run success.\n");
  return 0;
}
```

