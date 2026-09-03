# aclnnMultiScaleDeformableAttnFunction

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
- <term>Atlas 推理系列产品</term>：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- <term>Atlas 训练系列产品</term>：不支持
<!-- end id6 -->

## 功能说明

- 接口功能：

  通过采样位置（sample location）、注意力权重（attention weights）、映射后的value特征、多尺度特征起始索引位置、多尺度特征图的空间大小（便于将采样位置由归一化的值变成绝对位置）等参数来遍历不同尺寸特征图的不同采样点。

- 计算公式：

    设$b \in [0, bs)$、$q \in [0, num\_queries)$、$h \in [0, num\_heads)$、$\ell \in [0, num\_levels)$、$p \in [0, num\_points)$、$c \in [0, channels)$，分别为batch、查询、头、特征图、采样点、通道的索引。第$\ell$层特征图的高和宽为$H_\ell$、$W_\ell$，即$\mathrm{spatialShape}[\ell] = (H_\ell, W_\ell)$；$\mathrm{value}$的$num\_keys$维将所有层的特征图像素按层展平拼接（$num\_keys = \sum_{\ell=0}^{num\_levels-1} H_\ell W_\ell$），第$\ell$层的起始索引为$\mathrm{levelStartIndex}[\ell]$。

    1. 将采样点的归一化坐标$\mathrm{location}[b, q, h, \ell, p] = (u, v) \in [0,1]^2$（$u$、$v$分别为最后一维的第0、1个元素，对应$x$、$y$方向）映射到第$\ell$层特征图的像素坐标系：

       $$
       x = u \cdot W_\ell - 0.5, \qquad y = v \cdot H_\ell - 0.5
       $$

    2. 确定采样点落在哪四个整数网格点之间：

       $$
       x_0 = \lfloor x \rfloor,\quad x_1 = x_0 + 1,\qquad
       y_0 = \lfloor y \rfloor,\quad y_1 = y_0 + 1
       $$

    3. 计算采样点相对于左上角网格点的偏移，用于插值权重：

       $$
       \alpha_x = x - x_0, \qquad \alpha_y = y - y_0
       $$

    4. 计算双线性插值权重，四个邻点的和为1：

       $$
       \begin{aligned}
       w_{00} &= (1-\alpha_y)(1-\alpha_x), \\
       w_{10} &= (1-\alpha_y)\alpha_x, \\
       w_{01} &= \alpha_y(1-\alpha_x), \\
       w_{11} &= \alpha_y\alpha_x
       \end{aligned}
       $$

    5. 第$\ell$层特征图上的像素$(y, x)$展平后在$\mathrm{value}$的$num\_keys$维上的索引为：

       $$
       k_\ell(y, x) = \mathrm{levelStartIndex}[\ell] + y \cdot W_\ell + x
       $$

    6. 对$\mathrm{value}$做双线性采样，得到采样点对应的特征向量（长度为$channels$）：

       $$
       \begin{aligned}
       \operatorname{bilinear}(\mathrm{value};\,b,h,\ell,x,y) ={}&
       w_{00} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_0),\; h,\; :] \\
       &+ w_{10} \cdot \mathrm{value}[b,\; k_\ell(y_0, x_1),\; h,\; :] \\
       &+ w_{01} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_0),\; h,\; :] \\
       &+ w_{11} \cdot \mathrm{value}[b,\; k_\ell(y_1, x_1),\; h,\; :]
       \end{aligned}
       $$

    7. 所有层、所有采样点的双线性采样结果，以$\mathrm{attnWeight}$加权求和得到最终输出$\mathrm{output}$（shape为$(bs, num\_queries, num\_heads \times channels)$，最后一维按$h \times channels + c$排布）：

       $$
       \mathrm{output}[b,\; q,\; h \times channels + c] =
       \sum_{\ell=0}^{num\_levels-1} \sum_{p=0}^{num\_points-1}
       \mathrm{attnWeight}[b, q, h, \ell, p] \cdot
       \operatorname{bilinear}\!\left(\mathrm{value};\, b, h, \ell,\;
       x_{b,q,h,\ell,p},\, y_{b,q,h,\ell,p}\right)[c]
       $$

       其中$x_{b,q,h,\ell,p}$、$y_{b,q,h,\ell,p}$为采样点$\mathrm{location}[b, q, h, \ell, p]$经上述坐标映射得到的像素坐标。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnMultiScaleDeformableAttnFunction”接口执行计算。

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(
    const aclTensor* value,
    const aclTensor* spatialShape,
    const aclTensor* levelStartIndex,
    const aclTensor* location,
    const aclTensor* attnWeight,
    aclTensor*       output,
    uint64_t*        workspaceSize,
    aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnMultiScaleDeformableAttnFunction(
    void*          workspace,
    uint64_t       workspaceSize,
    aclOpExecutor* executor,
    aclrtStream    stream)
```

## aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize

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
      <td>value（aclTensor*）</td>
      <td>输入</td>
      <td>特征图的特征值。对应公式中的`value`。</td>
      <td>shape为(bs, num_keys, num_heads, channels)。其中bs为batch size，num_keys为特征图的大小，num_heads为头的数量，channels为特征图的维度。</td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>spatialShape（aclTensor*）</td>
      <td>输入</td>
      <td>存储每个尺度特征图的高和宽。对应公式中的`spatialShape`。</td>
      <td>shape为(num_levels, 2)。其中num_levels为特征图的数量，2分别代表H，W。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>levelStartIndex（aclTensor*）</td>
      <td>输入</td>
      <td>每张特征图的起始索引。对应公式中的`levelStartIndex`。</td>
      <td>shap为(num_levels)。</td>
      <td>INT32、INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>location（aclTensor*）</td>
      <td>输入</td>
      <td>采样点位置tensor，存储每个采样点的坐标位置。对应公式中的`location`。</td>
      <td><ul><il>数据类型与value保持一致。</il><il>shape为(bs, num_queries, num_heads, num_levels, num_points, 2)。其中num_queries为查询的数量，num_points为采样点的数量，2分别代表y，x。</il></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>6</td>
      <td>√</td>
    </tr>
    <tr>
      <td>attnWeight（aclTensor*）</td>
      <td>输入</td>
      <td>采样点权重tensor。对应公式中的`attnWeight`。</td>
      <td><ul><il>数据类型与value保持一致。</il><il>shape为(bs, num_queries, num_heads, num_levels, num_points)。</il></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>5</td>
      <td>√</td>
    </tr>
    <tr>
      <td>output（aclTensor*）</td>
      <td>输出</td>
      <td>算子计算输出。对应公式中的`output`。</td>
      <td><ul><il>数据类型与value保持一致。</il><il>shape为(bs, num_queries, num_heads * channels)。</il></ul></td>
      <td>FLOAT、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>3</td>
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

  <!-- npu="310p" id7 -->
  - <term>Atlas 推理系列产品</term>：不支持BFLOAT16数据类型。
  <!-- end id7 -->

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
      <td rowspan="10">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="10">161002</td>
      <td>输入和输出的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>输入输出数据类型不一致。</td>
    </tr>
    <tr>
      <td>value的shape不是4维。</td>
    </tr>
    <tr>
      <td>spatialShape的shape不是2维。</td>
    </tr>
    <tr>
      <td>levelStartIndex的shape不是1维。</td>
    </tr>
    <tr>
      <td>location的shape不是6维。</td>
    </tr>
    <tr>
      <td>attnWeight的shape不是5维。</td>
    </tr>
    <tr>
      <td>spatialShape的最后一轴不是2。</td>
    </tr>
    <tr>
      <td>location的最后一轴不是2。</td>
    </tr>
    <tr>
      <td>不满足接口约束说明章节。</td>
    </tr>
  </tbody>
  </table>

## aclnnMultiScaleDeformableAttnFunction

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize获取。</td>
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

- 确定性计算：

  <!-- npu="A3,910b,310p" id8 -->
  - <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas 推理系列产品</term>：aclnnMultiScaleDeformableAttnFunction默认确定性实现。
  <!-- end id8 -->
  <!-- npu="950" id9 -->
  - <term>Ascend 950PR/Ascend 950DT</term>：aclnnMultiScaleDeformableAttnFunction默认非确定性实现，支持通过aclrtCtxSetSysParamOpt开启确定性。

  <!-- end id9 -->

<!-- npu="310p" id10 -->
- <term>Atlas 推理系列产品</term>：
  - 通道数channels%32 = 0，且channels <= 256
  - 查询的数量32 <= num_queries< 500000
  - 特征图的数量num_levels <= 16
  - 头的数量num_heads = [2, 4, 8]
  - 采样点的数量num_points = [4, 8]
<!-- end id10 -->
<!-- npu="950,A3,910b" id11 -->
- <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>、<term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Ascend 950PR/Ascend 950DT</term>：
  - 通道数channels%8 = 0，且channels <= 256
  - 查询的数量32 <= num_queries < 500000
  - 特征图的数量num_levels <= 16
  - 头的数量num_heads <= 16
  - 采样点的数量num_points <= 16

<!-- end id11 -->

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_multi_scale_deformable_attn_function.h"

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
    std::vector<int64_t> valueShape = {1, 1, 2, 32};
    std::vector<int64_t> spatialShapeShape = {1, 2};
    std::vector<int64_t> levelStartIndexShape = {1};
    std::vector<int64_t> locationShape = {1, 32, 2, 1, 4, 2};
    std::vector<int64_t> attnWeightShape = {1, 32, 2, 1, 4};
    std::vector<int64_t> outputShape = {1, 32, 64};
    void* valueDeviceAddr = nullptr;
    void* spatialShapeDeviceAddr = nullptr;
    void* levelStartIndexDeviceAddr = nullptr;
    void* locationDeviceAddr = nullptr;
    void* attnWeightDeviceAddr = nullptr;
    void* outputDeviceAddr = nullptr;
    aclTensor* value = nullptr;
    aclTensor* spatialShape = nullptr;
    aclTensor* levelStartIndex = nullptr;
    aclTensor* location = nullptr;
    aclTensor* attnWeight = nullptr;
    aclTensor* output = nullptr;
    std::vector<float> valueHostData = {static_cast<float>(GetShapeSize(locationShape)), 1};
    std::vector<float> spatialShapeHostData = {1, 1};
    std::vector<float> levelStartIndexHostData = {0};
    std::vector<float> locationHostData(static_cast<float>(GetShapeSize(locationShape)), 0);
    std::vector<float> attnWeightHostData = {static_cast<float>(GetShapeSize(attnWeightShape)), 1};
    std::vector<float> outputHostData = {static_cast<float>(GetShapeSize(outputShape)), 1};
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
    // 创建output aclTensor
    ret = CreateAclTensor(outputHostData, outputShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &output);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3.调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnMultiScaleDeformableAttnFunction第一段接口
    ret = aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(value, spatialShape, levelStartIndex, location, attnWeight, output,
                                                                &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnMultiScaleDeformableAttnFunction第二段接口
    ret = aclnnMultiScaleDeformableAttnFunction(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnMultiScaleDeformableAttnFunction failed. ERROR: %d\n", ret); return ret);
    // 4.(固定写法)同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5.获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outputDeviceAddr, size * sizeof(float),
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
    aclDestroyTensor(output);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(valueDeviceAddr);
    aclrtFree(spatialShapeDeviceAddr);
    aclrtFree(levelStartIndexDeviceAddr);
    aclrtFree(locationDeviceAddr);
    aclrtFree(attnWeightDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```
