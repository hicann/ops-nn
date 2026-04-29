# aclnnDeformableConv2dBackward

[📄 查看源码](../../../conv/convolution_backward)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     x    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     x    |
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |

## 功能说明

- 接口功能：实现可变形卷积反向传播功能，计算输入梯度、权重梯度、偏置梯度和偏移量梯度。

- 计算公式：
  
  输入input($N,C_{in},H_{in},W_{in}$)、输出out($N,C_{out},H_{out},W_{out}$)和卷积步长($stride$)、卷积核大小($kernelSize，K_H,K_W$)、膨胀参数($dilation$)、填充($padding$)的关系是：
  
  $$
    H_{out}=\lfloor \frac{H_{in}+padding[0]+padding[1]-((K_H - 1) * dilation[2] + 1)}{stride[2]}+1 \rfloor
  $$
  
  $$
    W_{out}=\lfloor \frac{W_{in}+padding[2]+padding[3]-((K_W - 1) * dilation[3] + 1)}{stride[3]}+1 \rfloor
  $$
  
  可变形卷积反向传播需要计算对卷积正向的输入张量 $x$（对应函数原型中的input）、卷积核权重张量 $w$（对应函数原型中的weight）、偏置 $b$（对应函数原型中的bias）和偏移量 $o$（对应函数原型中的offset）的梯度。
  
  对于可变形卷积，每个输出位置的采样位置由偏移量动态决定。设输出位置为$(n, c_{out}, i, j)$，对应的偏移量为$\Delta p_{n,c_{out},i,j}$，则实际采样位置为$p_{n,c_{out},i,j} = (i \cdot s_H, j \cdot s_W) + \Delta p_{n,c_{out},i,j}$，其中$s_H$和$s_W$分别为高度和宽度方向的步长。
  
  - 对于 $x$ 的梯度 $\frac{\partial L}{\partial x}$（对应函数原型中的gradInput参数）：
  
    $$
    \frac{\partial L}{\partial x_{n, c_{in}, h, w}} = \sum_{c_{out}=1}^{C_{out}} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}} \cdot w_{c_{out}, c_{in}, p, q} \cdot \delta(p_{n,c_{out},i,j} = (h, w))
    $$
  
    其中，$L$ 为损失函数，$\frac{\partial L}{\partial y}$ 为输出张量 $y$ 对 $L$ 的梯度（对应函数原型中的gradOutput参数），$\delta(\cdot)$ 为指示函数，当采样位置与输入位置$(h,w)$重合时为1，否则为0。实际实现中通过双线性插值将梯度分配到邻近的像素位置。
  
  - 对于 $w$ 的梯度 $\frac{\partial L}{\partial w}$（对应函数原型中的gradWeight参数）：
  
    $$
    \frac{\partial L}{\partial w_{c_{out}, c_{in}, p, q}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} x_{n, c_{in}, p_{n,c_{out},i,j}.h, p_{n,c_{out},i,j}.w} \cdot \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$
  
    其中，$p_{n,c_{out},i,j}.h$和$p_{n,c_{out},i,j}.w$分别表示采样位置的高度和宽度坐标，通过双线性插值从输入张量$x$中采样得到。
  
  - 对于 $b$ 的梯度 $\frac{\partial L}{\partial b}$（对应函数原型中的gradBias参数）：
  
    $$
    \frac{\partial L}{\partial b_{c_{out}}} = \sum_{n=1}^{N} \sum_{i=1}^{H_{out}} \sum_{j=1}^{W_{out}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}}
    $$
  
  - 对于 $o$ 的梯度 $\frac{\partial L}{\partial o}$（对应函数原型中的gradOffset参数）：
  
    $$
    \frac{\partial L}{\partial o_{n, c_{out}, i, j}} = \sum_{c_{in}=1}^{C_{in}} \frac{\partial L}{\partial y_{n, c_{out}, i, j}} \cdot w_{c_{out}, c_{in}, p, q} \cdot \frac{\partial x_{n, c_{in}, p_{n,c_{out},i,j}.h, p_{n,c_{out},i,j}.w}}{\partial o_{n, c_{out}, i, j}}
    $$
  
    其中，偏移量梯度的计算涉及双线性插值的反向传播，需要计算输入值对偏移量的偏导数，该偏导数反映了偏移量变化对采样点位置的影响。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnDeformableConv2dBackwardGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnDeformableConv2dBackward"接口执行计算。

```Cpp
aclnnStatus aclnnDeformableConv2dBackwardGetWorkspaceSize(
  const aclTensor*   input,
  const aclTensor*   gradOutput,
  const aclTensor*   offsetOut,
  const aclTensor*   weight,
  const aclTensor*   offset,
  const aclIntArray* kernelSize,
  const aclIntArray* stride,
  const aclIntArray* padding,
  const aclIntArray* dilation,
  int64_t            groups,
  int64_t            deformableGroups,
  bool               modulated,
  aclTensor*         gradInput,
  aclTensor*         gradWeight,
  aclTensor*         gradOffset,
  aclTensor*         gradBias,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```

```Cpp
aclnnStatus aclnnDeformableConv2dBackward(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnDeformableConv2dBackwardGetWorkspaceSize

- **参数说明**：

  <table style="undefined;table-layout: fixed; width: 1550px"><colgroup>
  <col style="width: 170px">
  <col style="width: 120px">
  <col style="width: 271px">
  <col style="width: 330px">
  <col style="width: 223px">
  <col style="width: 101px">
  <col style="width: 190px">
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
      <td>input</td>
      <td>输入</td>
      <td>表示前向传播的输入数据。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[N, inC, inH, inW]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOutput</td>
      <td>输入</td>
      <td>表示前向传播输出的梯度。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>shape为[N, outC, outH, outW]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offsetOut</td>
      <td>输入</td>
      <td>表示前向传播可变形卷积的输出采样点。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>shape为[N, inC, outH * K_H, outW * K_W]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示前向传播的卷积核权重。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>shape为[outC, inC/groups, K_H, K_W]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>表示前向传播的x-y坐标偏移和掩码。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>当modulated为True时，shape为[N, 3 * deformableGroups * K_H * K_W, outH, outW]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>表示卷积核大小，对应公式中的K_H、K_W。</td>
      <td>size为2(K_H, K_W)，各元素均大于零，K_H * K_W不能超过2048，K_H * K_W * inC/groups不能超过65535。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>表示每个输入维度的滑动窗口步长，对应公式中的stride。</td>
      <td>size为4，各元素均大于零，维度顺序根据input的数据格式解释。N维和C维必须设置为1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>表示要添加到输入每侧（顶部、底部、左侧、右侧）的像素数，对应公式中的padding。</td>
      <td>size为4。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>表示输入每个维度的膨胀系数，对应公式中的dilation。</td>
      <td>size为4，各元素均大于零，维度顺序根据input的数据格式解释。N维和C维必须设置为1。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>输入</td>
      <td>表示分组卷积的分组数。</td>
      <td>inC和outC需都可被groups数整除，groups数大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>deformableGroups</td>
      <td>输入</td>
      <td>表示可变形组分区的数量。</td>
      <td>inC需可被deformableGroups数整除，deformableGroups数大于零。</td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>modulated</td>
      <td>输入</td>
      <td>表示offset中是否包含掩码。若为true，offset中包含掩码；若为false，则不包含。</td>
      <td>当前只支持true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradInput</td>
      <td>输出</td>
      <td>表示输入数据的梯度。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>shape为[N, inC, inH, inW]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeight</td>
      <td>输出</td>
      <td>表示卷积核权重的梯度。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>shape为[outC, inC/groups, K_H, K_W]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradOffset</td>
      <td>输出</td>
      <td>表示偏移量的梯度。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与input保持一致。</li><li>当modulated为True时，shape为[N, 3 * deformableGroups * K_H * K_W, outH, outW]；当modulated为False时，shape为[N, 2 * deformableGroups * K_H * K_W, outH, outW]。</li></ul></td>
      <td>与input保持一致</td>
      <td>NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradBias</td>
      <td>输出</td>
      <td>表示偏置的梯度。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[outC]。</li></ul></td>
      <td>与input保持一致</td>
      <td>ND</td>
      <td>1</td>
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

- **返回值**：

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1170px"><colgroup>
  <col style="width: 268px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>传入的input、gradOutput、offsetOut、weight、offset、gradInput、gradWeight、gradBias、gradOffset是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>input、gradOutput、offsetOut、weight、offset、gradInput、gradWeight、gradBias、gradOffset的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>input、gradOutput、offsetOut、weight、offset、gradInput、gradWeight、gradBias、gradOffset的shape与参数说明中不一致。</td>
    </tr>
    <tr>
      <td>kernelSize、stride、padding、dilation的size与参数说明中不一致。</td>
    </tr>
    <tr>
      <td>K_H*K_W超过2048，或者K_H*K_W*inC/groups超过65535。</td>
    </tr>
    <tr>
      <td>groups或deformableGroups参数不满足约束条件。</td>
    </tr>
    <tr>
      <td>modulated参数不等于true。</td>
    </tr>
    <tr>
     <td>ACLNN_ERR_INNER_NULLPTR</td>
     <td>561103</td>
     <td>API内部校验错误，通常由于输入数据或属性的规格不在支持的范围之内导致。</td>
   </tr>
  </tbody></table>

## aclnnDeformableConv2dBackward

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDeformableConv2dBackwardGetWorkspaceSize获取。</td>
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

- 确定性计算：
  - aclnnDeformableConv2dBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_convolution_backward.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr)        \
    do {                                         \
        if (!(cond)) {                           \
            Finalize(deviceId, stream); \
            return_expr;                         \
        }                                        \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
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
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    // 调用aclrtMalloc申请device侧内存
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    // 调用aclrtMemcpy将host侧数据复制到device侧内存上
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    // 计算连续tensor的strides
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    auto format = shape.size() == 1 ? ACL_FORMAT_ND : ACL_FORMAT_NCHW;
    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, format, shape.data(), shape.size(), *deviceAddr);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

int aclnnDeformableConv2dBackwardTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> inputShape = {1, 6, 2, 4};
    std::vector<int64_t> gradOutputShape = {1, 4, 2, 4};
    std::vector<int64_t> offsetOutShape = {1, 6, 10, 20};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> gradInputShape = {1, 6, 2, 4};
    std::vector<int64_t> gradWeightShape = {4, 3, 5, 5};
    std::vector<int64_t> gradBiasShape = {4};
    std::vector<int64_t> gradOffsetShape = {1, 75, 2, 4};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* inputDeviceAddr = nullptr;
    void* gradOutputDeviceAddr = nullptr;
    void* offsetOutDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* gradInputDeviceAddr = nullptr;
    void* gradWeightDeviceAddr = nullptr;
    void* gradBiasDeviceAddr = nullptr;
    void* gradOffsetDeviceAddr = nullptr;
    aclTensor* input = nullptr;
    aclTensor* gradOutput = nullptr;
    aclTensor* offsetOut = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* gradInput = nullptr;
    aclTensor* gradWeight = nullptr;
    aclTensor* gradBias = nullptr;
    aclTensor* gradOffset = nullptr;
    std::vector<float> inputHostData(1 * 6 * 2 * 4, 1);
    std::vector<float> gradOutputHostData(1 * 4 * 2 * 4, 1);
    std::vector<float> offsetOutHostData(1 * 6 * 10 * 20, 1);
    std::vector<float> weightHostData(4 * 3 * 5 * 5, 1);
    std::vector<float> offsetHostData(1 * 75 * 2 * 4, 1);
    std::vector<float> gradInputHostData(1 * 6 * 2 * 4, 0);
    std::vector<float> gradWeightHostData(4 * 3 * 5 * 5, 0);
    std::vector<float> gradBiasHostData(4, 0);
    std::vector<float> gradOffsetHostData(1 * 75 * 2 * 4, 0);
    // 创建input aclTensor
    ret = CreateAclTensor(inputHostData, inputShape, &inputDeviceAddr, aclDataType::ACL_FLOAT, &input);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradOutput aclTensor
    ret = CreateAclTensor(
        gradOutputHostData, gradOutputShape, &gradOutputDeviceAddr, aclDataType::ACL_FLOAT, &gradOutput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offsetOut aclTensor
    ret = CreateAclTensor(offsetOutHostData, offsetOutShape, &offsetOutDeviceAddr, aclDataType::ACL_FLOAT, &offsetOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradInput aclTensor
    ret = CreateAclTensor(gradInputHostData, gradInputShape, &gradInputDeviceAddr, aclDataType::ACL_FLOAT, &gradInput);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradWeight aclTensor
    ret = CreateAclTensor(
        gradWeightHostData, gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradBias aclTensor
    ret = CreateAclTensor(gradBiasHostData, gradBiasShape, &gradBiasDeviceAddr, aclDataType::ACL_FLOAT, &gradBias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建gradOffset aclTensor
    ret = CreateAclTensor(
        gradOffsetHostData, gradOffsetShape, &gradOffsetDeviceAddr, aclDataType::ACL_FLOAT, &gradOffset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建kernelSize aclIntArray
    const aclIntArray* kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // 创建stride aclIntArray
    const aclIntArray* strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // 创建padding aclIntArray
    const aclIntArray* paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // 创建dilation aclIntArray
    const aclIntArray* dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnDeformableConv2dBackward第一段接口
    ret = aclnnDeformableConv2dBackwardGetWorkspaceSize(
        input, gradOutput, offsetOut, weight, offset, kernelSizeArray, strideArray, paddingArray, dilationArray, groups,
        deformableGroups, true, gradInput, gradWeight, gradOffset, gradBias, &workspaceSize, &executor);
    CHECK_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
        return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnDeformableConv2dBackward第二段接口
    ret = aclnnDeformableConv2dBackward(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackward failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(gradInputShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(
        resultData.data(), resultData.size() * sizeof(resultData[0]), gradInputDeviceAddr, size * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    size = GetShapeSize(gradWeightShape);
    std::vector<float> gradWeightShapeData(size, 0);
    ret = aclrtMemcpy(
        gradWeightShapeData.data(), gradWeightShapeData.size() * sizeof(gradWeightShapeData[0]), gradWeightDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradWeightShapeData[i]);
    }

    size = GetShapeSize(gradOffsetShape);
    std::vector<float> gradOffsetShapeData(size, 0);
    ret = aclrtMemcpy(
        gradOffsetShapeData.data(), gradOffsetShapeData.size() * sizeof(gradOffsetShapeData[0]), gradOffsetDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradOffsetShape[i]);
    }

    size = GetShapeSize(gradBiasShape);
    std::vector<float> gradBiasShapeData(size, 0);
    ret = aclrtMemcpy(
        gradBiasShapeData.data(), gradBiasShapeData.size() * sizeof(gradBiasShapeData[0]), gradBiasDeviceAddr,
        size * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, gradBiasShapeData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(input);
    aclDestroyTensor(gradOutput);
    aclDestroyTensor(offsetOut);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(gradInput);
    aclDestroyTensor(gradWeight);
    aclDestroyTensor(gradBias);
    aclDestroyTensor(gradOffset);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(inputDeviceAddr);
    aclrtFree(gradOutputDeviceAddr);
    aclrtFree(offsetOutDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(gradInputDeviceAddr);
    aclrtFree(gradWeightDeviceAddr);
    aclrtFree(gradBiasDeviceAddr);
    aclrtFree(gradOffsetDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    return ACL_SUCCESS;
}

int main()
{
    // 1. （固定写法）device/stream初始化
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnDeformableConv2dBackwardTest(deviceId, stream);
    CHECK_FREE_RET(
        ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dBackwardTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}
```
