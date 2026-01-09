# aclnnDeformableConv2d

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件</term>     |     √    |

## 功能说明

- 接口功能：实现卷积功能，支持2D卷积，同时支持可变形卷积、分组卷积。

- 计算公式：
  
  假定输入（input）的shape是[N, inC, inH, inW]，输出的（out）的shape为[N, outC, outH, outW]，根据已有参数计算outH、outW:
  
  $$
  outH = (inH + padding[0] + padding[1] - ((K_H - 1) * dilation[2] + 1)) // stride[2] + 1
  $$
  
  $$
  outW = (inW + padding[2] + padding[3] - ((K_W - 1) * dilation[3] + 1)) // stride[3] + 1
  $$
  
  标准卷积计算采样点下标：
  
  $$
  x = -padding[2] + ow*stride[3] + kw*dilation[3], kw的取值为(0, K\_W-1)
  $$
  
  $$
  y = -padding[0] + oh*stride[2] + kh*dilation[2], kh的取值为(0, K\_H-1)
  $$
  
  根据传入的offset，进行变形卷积，计算偏移后的下标：
  
  $$
  (x,y) = (x + offsetX, y + offsetY)
  $$

  使用双线性插值计算偏移后点的值：
  
  $$
  (x_{0}, y_{0}) = (int(x), int(y)) \\
  (x_{1}, y_{1}) = (x_{0} + 1, y_{0} + 1)
  $$
  
  $$
  weight_{00} = (x_{1} - x) * (y_{1} - y) \\
  weight_{01} = (x_{1} - x) * (y - y_{0}) \\ 
  weight_{10} = (x - x_{0}) * (y_{1} - y) \\ 
  weight_{11} = (x - x_{0}) * (y - y_{0}) \\ 
  $$
  
  $$
  deformOut(x, y) = weight_{00} * input(x0, y0) + weight_{01} * input(x0,y1) + weight_{10} * input(x1, y0) + weight_{11} * input(x1,y1)
  $$
  
  进行卷积计算得到最终输出：
  
  $$
  \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) + \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{deformOut}(N_i, k)
  $$
  
## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnDeformableConv2dGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnDeformableConv2d”接口执行计算。

```Cpp
aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(
  const aclTensor*   x,
  const aclTensor*   weight,
  const aclTensor*   offset,
  const aclTensor*   biasOptional,
  const aclIntArray* kernelSize,
  const aclIntArray* stride,
  const aclIntArray* padding,
  const aclIntArray* dilation,
  int64_t            groups,
  int64_t            deformableGroups,
  bool               modulated,
  aclTensor*         out,
  aclTensor*         deformOutOptional,
  uint64_t*          workspaceSize,
  aclOpExecutor**    executor)
```
```Cpp
aclnnStatus aclnnDeformableConv2d(
  void*          workspace,
  uint64_t       workspaceSize,
  aclOpExecutor* executor,
  aclrtStream    stream)
```

## aclnnDeformableConv2dGetWorkspaceSize

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
      <td>x</td>
      <td>输入</td>
      <td>表示输入的原始数据。对应公式中的`input`。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[N, inC, inH, inW]，其中inH * inW不能超过2147483647。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weight</td>
      <td>输入</td>
      <td>表示可学习过滤器的4D张量，对应公式中的`weight`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与入参`x`保持一致。</li><li>shape为[outC, inC/groups, K_H, K_W]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>offset</td>
      <td>输入</td>
      <td>表示x-y坐标偏移和掩码的4D张量。对应公式中的`offset`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与入参`x`保持一致。</li><li>当`modulated`为True时，shape为[N, 3 * deformableGroups * K_H * K_W, outH, outW]；当`modulated`为False时，shape为[N, 2 * deformableGroups * K_H * K_W, outH, outW]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>biasOptional</td>
      <td>输入</td>
      <td>可选输入参数，表示过滤器输出附加偏置的1D张量，对应公式中的`bias`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型与入参`x`保持一致。</li><li>不需要时为空指针，存在时shape为[outC]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>kernelSize</td>
      <td>输入</td>
      <td>表示卷积核大小，对应公式中的`K_H`、`K_W`。</td>
      <td>size为2(K_H, K_W)，各元素均大于零，K_H * K_W不能超过2048，K_H * K_W * inC/groups不能超过65535。</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>stride</td>
      <td>输入</td>
      <td>表示每个输入维度的滑动窗口步长，对应公式中的`stride`。</td>
      <td>size为4，各元素均大于零，维度顺序根据`x`的数据格式解释。N维和C维必须设置为1。</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>padding</td>
      <td>输入</td>
      <td>表示要添加到输入每侧（顶部、底部、左侧、右侧）的像素数，对应公式中的`padding`。</td>
      <td>size为4。</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>dilation</td>
      <td>输入</td>
      <td>表示输入每个维度的膨胀系数，对应公式中的`dilation`。</td>
      <td>size为4，各元素均大于零，维度顺序根据x的数据格式解释。N维和C维必须设置为1。</td>
      <td>aclIntArray</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>groups</td>
      <td>输入</td>
      <td>表示从输入通道到输出通道的分组连接数。</td>
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
      <td>预留参数，表示offset中是否包含掩码。若为true，`offset`中包含掩码；若为false，则不包含。</td>
      <td>当前只支持true。</td>
      <td>BOOL</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out</td>
      <td>输出</td>
      <td>表示输出的数据。对应公式中的`out`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与`x`保持一致。</li><li>shape为[N, outC, outH, outW]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW</td>
      <td>4</td>
      <td>√</td>
    </tr>
    <tr>
      <td>deformOutOptional</td>
      <td>输出</td>
      <td>可选输出，表示可变形卷积采样点。对应公式中的`deformOut`。</td>
      <td><ul><li>不支持空Tensor。</li><li>数据类型、数据格式与`x`的数据类型、数据格式保持一致。</li><li>shape为[N, inC, outH * K_H, outW * K_W]。</li></ul></td>
      <td>FLOAT32、FLOAT16、BFLOAT16</td>
      <td>ND、NCHW</td>
      <td>4</td>
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
      <td>传入的x、weight、offset、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="6">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="6">161002</td>
      <td>x、weight、offset、out的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>deformOutOptional不为空指针时，数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>biasOptional不为空指针时，数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>x、weight、offset、biasOptional、out、deformOutOptional的shape与参数说明中不一致。</td>
    </tr>
    <tr>
      <td>kernelSize、stride、padding、dilation的size与参数说明中不一致。</td>
    </tr>
    <tr>
      <td>K_H*K_W超过2048，或者K_H*K_W*inC/groups超过65535。</td>
    </tr>
  </tbody></table>

## aclnnDeformableConv2d

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnDeformableConv2dGetWorkspaceSize获取。</td>
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
  - aclnnDeformableConv2d默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_deformable_conv2d.h"
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
    int64_t shape_size = 1;
    for (auto i : shape) {
        shape_size *= i;
    }
    return shape_size;
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
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, format,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    // check根据自己的需要处理
    CHECK_RET(ret == 0, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);
    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> xShape = {1, 6, 2, 4};
    std::vector<int64_t> weightShape = {4, 3, 5, 5};
    std::vector<int64_t> offsetShape = {1, 75, 2, 4};
    std::vector<int64_t> biasShape = {4};
    std::vector<int64_t> outShape = {1, 4, 2, 4};
    std::vector<int64_t> deformOutShape = {1, 6, 10, 20};
    std::vector<int64_t> kernelSize = {5, 5};
    std::vector<int64_t> stride = {1, 1, 1, 1};
    std::vector<int64_t> padding = {2, 2, 2, 2};
    std::vector<int64_t> dilation = {1, 1, 1, 1};
    int64_t groups = 2;
    int64_t deformableGroups = 1;
    void* xDeviceAddr = nullptr;
    void* weightDeviceAddr = nullptr;
    void* offsetDeviceAddr = nullptr;
    void* biasDeviceAddr = nullptr;
    void* outDeviceAddr = nullptr;
    void* deformOutDeviceAddr = nullptr;
    aclTensor* x = nullptr;
    aclTensor* weight = nullptr;
    aclTensor* offset = nullptr;
    aclTensor* bias = nullptr;
    aclTensor* out = nullptr;
    aclTensor* deformOut = nullptr;
    std::vector<float> xHostData(1*6*2*4,1);
    std::vector<float> weightHostData(4*3*5*5,1);
    std::vector<float> offsetHostData(1*75*2*4,1);
    std::vector<float> biasHostData(4,0);
    std::vector<float> outHostData(1*4*2*4,0);
    std::vector<float> deformOutHostData(1*6*10*20,0);
    // 创建x aclTensor
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT, &x);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    
    // 创建weight aclTensor
    ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weight);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建offset aclTensor
    ret = CreateAclTensor(offsetHostData, offsetShape, &offsetDeviceAddr, aclDataType::ACL_FLOAT, &offset);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建bias aclTensor
    ret = CreateAclTensor(biasHostData, biasShape, &biasDeviceAddr, aclDataType::ACL_FLOAT, &bias);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建out aclTensor
    ret = CreateAclTensor(outHostData, outShape, &outDeviceAddr, aclDataType::ACL_FLOAT, &out);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建deformOut aclTensor
    ret = CreateAclTensor(deformOutHostData, deformOutShape, &deformOutDeviceAddr, aclDataType::ACL_FLOAT, &deformOut);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 创建kernelSize aclIntArray
    const aclIntArray *kernelSizeArray = aclCreateIntArray(kernelSize.data(), kernelSize.size());
    CHECK_RET(kernelSizeArray != nullptr, return ret);
    // 创建stride aclIntArray
    const aclIntArray *strideArray = aclCreateIntArray(stride.data(), stride.size());
    CHECK_RET(strideArray != nullptr, return ret);
    // 创建padding aclIntArray
    const aclIntArray *paddingArray = aclCreateIntArray(padding.data(), padding.size());
    CHECK_RET(paddingArray != nullptr, return ret);
    // 创建dilation aclIntArray
    const aclIntArray *dilationArray = aclCreateIntArray(dilation.data(), dilation.size());
    CHECK_RET(dilationArray != nullptr, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;
    // 调用aclnnDeformableConv2d第一段接口
    ret = aclnnDeformableConv2dGetWorkspaceSize(x, weight, offset, bias, kernelSizeArray, strideArray, paddingArray, dilationArray, groups, deformableGroups, true, out, deformOut, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2dGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);
    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret;);
    }
    // 调用aclnnDeformableConv2d第二段接口
    ret = aclnnDeformableConv2d(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnDeformableConv2d failed. ERROR: %d\n", ret); return ret);
    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);
    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    auto size = GetShapeSize(outShape);
    std::vector<float> resultData(size, 0);
    ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), outDeviceAddr, size * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return ret);
    for (int64_t i = 0; i < size; i++) {
        LOG_PRINT("result[%ld] is: %f\n", i, resultData[i]);
    }

    // 6. 释放aclTensor和aclScalar，需要根据具体API的接口定义修改
    aclDestroyTensor(x);
    aclDestroyTensor(weight);
    aclDestroyTensor(offset);
    aclDestroyTensor(bias);
    aclDestroyTensor(out);
    aclDestroyTensor(deformOut);

    // 7. 释放device资源，需要根据具体API的接口定义修改
    aclrtFree(xDeviceAddr);
    aclrtFree(weightDeviceAddr);
    aclrtFree(offsetDeviceAddr);
    aclrtFree(biasDeviceAddr);
    aclrtFree(outDeviceAddr);
    aclrtFree(deformOutDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
```