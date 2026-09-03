# aclnnMultiScaleDeformableAttentionGrad

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR/Ascend 950DT</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- <term>Atlas 200I/500 A2 推理产品</term>：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- <term>Atlas 推理系列产品</term>：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 算子功能：

  MultiScaleDeformableAttention正向算子功能主要通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。而反向算子的功能为根据正向的输入对输出的贡献及初始梯度求出输入对应的梯度。
- 计算公式：

    设$b \in [0, bs)$、$q \in [0, num\_queries)$、$h \in [0, num\_heads)$、$\ell \in [0, num\_levels)$、$p \in [0, num\_points)$、$c \in [0, channels)$，分别为batch、查询、头、特征图、采样点、通道的索引。第$\ell$层特征图的高和宽为$H_\ell$、$W_\ell$，即$\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$；第$\ell$层特征图像素$(y, x)$在$\mathrm{value}$的$num\_keys$维上的展平索引为$k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x$。采样点$\mathrm{location}[b, q, h, \ell, p] = (u, v)$（最后一维第0、1个元素分别对应$x$、$y$方向）映射到像素坐标$x = u \cdot W_\ell - 0.5$、$y = v \cdot H_\ell - 0.5$；记$x_0 = \lfloor x \rfloor$、$x_1 = x_0 + 1$、$y_0 = \lfloor y \rfloor$、$y_1 = y_0 + 1$，$\alpha_x = x - x_0$、$\alpha_y = y - y_0$，双线性插值权重$w_{00} = (1-\alpha_y)(1-\alpha_x)$、$w_{10} = (1-\alpha_y)\alpha_x$、$w_{01} = \alpha_y(1-\alpha_x)$、$w_{11} = \alpha_y\alpha_x$，并记四邻点特征向量$V_{y_i, x_j} := \mathrm{value}[b,\; k_\ell(y_i, x_j),\; h,\; :]$（$i, j \in \{0, 1\}$，越界邻点按0处理）。

    - 正向输出为：

      $$
      \mathrm{output}[b,\; q,\; h \times channels + c] =
      \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
      \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
      \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c]
      $$

    - 反向算子根据梯度$\mathrm{gradOutput}$（shape为$(bs, num\_queries, num\_heads \times channels)$，最后一维按$h \times channels + c$排布）计算$\mathrm{value}$、$\mathrm{location}$、$\mathrm{attnWeight}$三个输入的梯度。记$\mathrm{gradOutput}$中head$h$的切片为$G_{b,q,h,c} := \mathrm{gradOutput}[b,\; q,\; h \times channels + c]$。
      1. 按注意力权重展开：

         $$
         \tilde{G}_{b,q,h,\ell,p,c} = \mathrm{attnWeight}[b, q, h, \ell, p] \cdot G_{b,q,h,c}
         $$

      2. 计算$\mathrm{value}$的梯度$\mathrm{gradValue}$（shape与value一致）：对每个采样点$(b, q, h, \ell, p)$，将其展开梯度按双线性权重累加到四个邻点位置（对所有$q$、$p$累加，越界邻点不累加）：

         $$
         \begin{aligned}
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_0),\; h,\; c] &+= w_{00} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_0, x_1),\; h,\; c] &+= w_{10} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_0),\; h,\; c] &+= w_{01} \cdot \tilde{G}_{b,q,h,\ell,p,c}, \\
         \mathrm{gradValue}[b,\; k_\ell(y_1, x_1),\; h,\; c] &+= w_{11} \cdot \tilde{G}_{b,q,h,\ell,p,c}
         \end{aligned}
         $$

      3. 计算$\mathrm{attnWeight}$的梯度$\mathrm{gradAttnWeight}$（shape与attnWeight一致），为输出梯度与采样得到的特征向量的内积：

         $$
         \mathrm{gradAttnWeight}[b, q, h, \ell, p] =
         \sum_{c} G_{b,q,h,c} \cdot
         \left( \sum_{i,j \in \{0,1\}} w_{ij} \cdot V_{y_i, x_j}[c] \right)
         $$

      4. 计算$\mathrm{location}$的梯度$\mathrm{gradLocation}$（shape与location一致）。采样点像素坐标$(x, y)$的梯度由双线性插值的一阶偏导得到：

         $$
         \nabla x_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_0, x_1}[c] - V_{y_0, x_0}[c])(1-\alpha_y)
             + (V_{y_1, x_1}[c] - V_{y_1, x_0}[c])\alpha_y \big]
         $$

         $$
         \nabla y_{b,q,h,\ell,p} =
         \sum_{c} \tilde{G}_{b,q,h,\ell,p,c} \,
         \big[ (V_{y_1, x_0}[c] - V_{y_0, x_0}[c])(1-\alpha_x)
             + (V_{y_1, x_1}[c] - V_{y_0, x_1}[c])\alpha_x \big]
         $$

      5. 由$x = u \cdot W_\ell - 0.5$、$y = v \cdot H_\ell - 0.5$，按链式法则缩放回归一化坐标的梯度，写入$\mathrm{gradLocation}$（最后一维第0、1个元素分别对应$x$、$y$方向，与location一致）：

         $$
         \mathrm{gradLocation}[b, q, h, \ell, p] = (\nabla u, \nabla v), \qquad
         \nabla u = W_\ell \cdot \nabla x_{b,q,h,\ell,p}, \qquad
         \nabla v = H_\ell \cdot \nabla y_{b,q,h,\ell,p}
         $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMultiScaleDeformableAttentionGrad”接口执行计算。

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(
    const aclTensor* value,
    const aclTensor* spatialShape,
    const aclTensor* levelStartIndex,
    const aclTensor* location,
    const aclTensor* attnWeight,
    const aclTensor* gradOutput,
    aclTensor*       gradValue,
    aclTensor*       gradLocation,
    aclTensor*       gradAttnWeight,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttentionGrad(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 100%"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 250px">
  <col style="width: 212px">
  <col style="width: 100px">
  <col style="width: 300px">
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
      <td>value</td>
      <td>输入</td>
      <td>特征图的特征值。对应公式中的`value`。</td>
      <td>shape为(bs, num_keys, num_heads, channels)，其中：<ul><li>bs为batch size。</li><li>num_keys为特征图的大小。</li><li>num_heads为头的数量。</li><li>channels为特征图的维度。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>spatialShape</td>
      <td>输入</td>
      <td>存储每个尺度特征图的高和宽。对应公式中的`spatialShape`。</td>
      <td>shape为(num_levels, 2)，其中：<ul><li>num_levels为特征图的数量。</li><li>2分别代表H，W。</li></ul></td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>levelStartIndex</td>
      <td>输入</td>
      <td>每张特征图的起始索引。对应公式中的`levelStartIndex`。</td>
      <td>shape为(num_levels)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>location</td>
      <td>输入</td>
      <td>采样点位置tensor，存储每个采样点的坐标位置。对应公式中的`location`。</td>
      <td>shape为(bs, num_queries, num_heads, num_levels, num_points, 2)，其中：<ul><li>num_queries为查询的数量。</li><li>num_points为采样点的数量。</li><li>2分别代表y，x。</li></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight</td>
      <td>输入</td>
      <td>采样点权重tensor。对应公式中的`​attnWeight`。</td>
      <td>shape为(bs, num_queries, num_heads, num_levels, num_points)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>正向输出梯度，也是反向算子的初始梯度。对应公式中的`gradOutput`。</td>
      <td>shape为(bs, num_queries, num_heads*channels)。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradValue</td>
      <td>输出</td>
      <td>输入value对应的梯度。对应公式中的`gradValue`。</td>
      <td>shape与value保持一致</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradLocation</td>
      <td>输出</td>
      <td>输入location对应的梯度。对应公式中的`gradLocation`。</td>
      <td>shape与location保持一致</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradAttnWeight</td>
      <td>输出</td>
      <td>输入attnWeight对应的梯度。对应公式中的`gradAttnWeight`。</td>
      <td>shape与attnWeight保持一致</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed"><colgroup>
  <col style="width: 250px">
  <col style="width: 130px">
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
      <td>传入的输入或输出是空指针。</td>
    </tr>
    <tr>
      <td rowspan="2">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="2">161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>不满足接口约束说明章节。</td>
    </tr>
  </tbody>
  </table>

## aclnnMultiScaleDeformableAttentionGrad

- **参数说明**：

  <table><thead>
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize获取。</td>
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
- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算
  - aclnnMultiScaleDeformableAttentionGrad默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

- 通道数channels%8 = 0，且channels<=256
- 查询的数量num_queries < 500000
- 特征图的数量num_levels <= 16
- 头的数量num_heads <= 16
- 采样点的数量num_points <= 16

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multi_scale_deformable_attention_grad.h"

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
    // 1.(固定写法)device/stream初始化,参考acl对外接口列表
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2.构造输入与输出，需要根据API的接口自定义构造

    std::vector<int64_t> valueShape = {1, 1, 1, 8};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 1, 1, 1};
    std::vector<int64_t> gradOutputShape = {1, 32, 8};
    std::vector<int64_t> gradValueShape = {1, 1, 1, 8};
    std::vector<int64_t> gradLocationShape = {1, 32, 1, 1, 1, 2};
    std::vector<int64_t> gradAttnWeightShape = {1, 32, 1, 1, 1};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* gradValueDeviceAddr = nullptr;
    void* gradLocationDeviceAddr = nullptr;
    void* gradAttnWeightDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* gradValue = nullptr;
    aclTensor* gradLocation = nullptr;
    aclTensor* gradAttnWeight = nullptr;
    std::vector<float> valueHostData = {1, 1, 1, 1, 1, 1, 1, 1};
    std::vector<float> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> gradValueHostData = {0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<float> gradLocationHostData(GetShapeSize(gradLocationShape), 0);
    std::vector<float> gradAttnWeightHostData(GetShapeSize(gradAttnWeightShape), 0);
    std::vector<float> locationHostData(GetShapeSize(locationShape), 0);
    std::vector<float> attnWeightHostData(GetShapeSize(attnWeightShape), 1);
    std::vector<float> gradOutputHostData(GetShapeSize(gradOutputShape), 1);

    // value aclTensor
    ret = CreateAclTensor(valueHostData, valueShape, &valueDeviceAddr, aclDataType::ACL_FLOAT, &value);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建spatialShape aclTensor
    ret = CreateAclTensor(spatialShapeHostData, spatialShapeShape, &spatialShapeDeviceAddr, aclDataType::ACL_INT32, &spatialShape);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建levelStartIndex aclTensor
    ret = CreateAclTensor(levelStartIndexHostData, levelStartIndexShape, &levelStartIndexDeviceAddr, aclDataType::ACL_INT32, &levelStartIndex);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建location aclTensor
    ret = CreateAclTensor(locationHostData, locationShape, &locationDeviceAddr, aclDataType::ACL_FLOAT, &location);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建attnWeight aclTensor
    ret = CreateAclTensor(attnWeightHostData, attnWeightShape, &attnWeightDeviceAddr, aclDataType::ACL_FLOAT, &attnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // gradOutput aclTensor
    ret = CreateAclTensor(gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradValue aclTensor
    ret = CreateAclTensor(gradValueHostData, gradValueShape, &gradValueDeviceAddr, aclDataType::ACL_FLOAT, &gradValue);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradLocation aclTensor
    ret = CreateAclTensor(gradLocationHostData, gradLocationShape, &gradLocationDeviceAddr, aclDataType::ACL_FLOAT, &gradLocation);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradAttnWeight aclTensor
    ret = CreateAclTensor(gradAttnWeightHostData, gradAttnWeightShape, &gradAttnWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradAttnWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMultiScaleDeformableAttentionGrad第一段接口
    ret = aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, gradOutput, gradValue, gradLocation, gradAttnWeight, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMultiScaleDeformableAttentionGrad第二段接口
    ret = aclnnMultiScaleDeformableAttentionGrad(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttentionGrad failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradValueShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), gradValueDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6.释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(value);
    aclDestroyTensor(spatialShape);
    aclDestroyTensor(levelStartIndex);
    aclDestroyTensor(location);
    aclDestroyTensor(attnWeight);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(gradValue);
    aclDestroyTensor(gradLocation);
    aclDestroyTensor(gradAttnWeight);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(gradValueDeviceAddr);
    aclrtFree(gradLocationDeviceAddr);
    aclrtFree(gradAttnWeightDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
