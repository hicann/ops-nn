# aclnnTurboQuant

[查看源码](https://gitcode.com/cann/ops-nn/tree/master/quant/turbo_quant)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：不支持
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

TurboQuant将MLA的KV latent按token归一化，并根据16个升序Lloyd-Max codebook中心点生成4bit量化索引；同时输出每个token的L2 norm `scale`。

对输入`latent`的第`i`行，计算过程为：

$$
scale_i = \sqrt{\sum_d latent_{i,d}^2 + 10^{-16}}
$$

$$
u_{i,d} = latent_{i,d} / scale_i
$$

$$
nibble_{i,d} = \sum_{b=0}^{14} [u_{i,d} \ge (centroids_b + centroids_{b+1}) / 2]
$$

$$
y_{i,k} = nibble_{i,2k} | (nibble_{i,2k+1} \ll 4)
$$

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用`aclnnTurboQuantGetWorkspaceSize`获取workspace大小和执行器，再调用`aclnnTurboQuant`执行计算。

## 函数原型

```cpp
aclnnStatus aclnnTurboQuantGetWorkspaceSize(
  const aclTensor *latent,
  const aclTensor *centroids,
  const aclTensor *y,
  const aclTensor *scale,
  uint64_t        *workspaceSize,
  aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnTurboQuant(
  void          *workspace,
  uint64_t      workspaceSize,
  aclOpExecutor *executor,
  aclrtStream   stream)
```

## aclnnTurboQuantGetWorkspaceSize

- **参数说明：**

  <table style="table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 300px">
  <col style="width: 350px">
  <col style="width: 250px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 100px">
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
      <td>latent（const aclTensor*）</td>
      <td>输入</td>
      <td>待压缩的KV latent，对应计算公式中的latent。</td>
      <td><ul><li>支持空Tensor。</li><li>必须为FLOAT32、2D，最后一维为512。</li><li>必须已完成signed Hadamard旋转，且未归一化。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[numTokens, 512]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>centroids（const aclTensor*）</td>
      <td>输入</td>
      <td>Lloyd-Max codebook中心点，对应计算公式中的centroids。</td>
      <td><ul><li>不支持空Tensor。</li><li>必须为1D、正好16个FLOAT32元素，且按升序排列。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>[16]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>y（const aclTensor*）</td>
      <td>输出</td>
      <td>打包后的4bit索引，对应计算公式中的y。</td>
      <td><ul><li>不支持空Tensor。</li><li>必须预先创建。</li><li>低nibble保存偶数维索引。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>[numTokens, 256]</td>
      <td>×</td>
    </tr>
    <tr>
      <td>scale（const aclTensor*）</td>
      <td>输出</td>
      <td>每个token的L2 norm，对应计算公式中的scale。</td>
      <td><ul><li>不支持空Tensor。</li><li>必须预先创建。</li></ul></td>
      <td>FLOAT16</td>
      <td>ND</td>
      <td>[numTokens]</td>
      <td>×</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口完成入参校验，出现以下场景时报错：

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 300px">
  <col style="width: 150px">
  <col style="width: 550px">
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
      <td>latent、centroids、y、scale、workspaceSize或executor是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>latent或centroids的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>latent的rank不为2，或centroids的shape不为[16]。</td>
    </tr>
    <tr>
      <td>latent的shape不满足校验条件（numTokens小于-1，或headDim不为512且不为-1）。</td>
    </tr>
    <tr>
      <td>y或scale的shape与推导结果不一致。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_RUNTIME_ERROR</td>
      <td>361001</td>
      <td>当前平台不在支持的平台范围内。</td>
    </tr>
  </tbody></table>

## aclnnTurboQuant

- **参数说明：**

  <table style="table-layout: fixed; width: 1000px"><colgroup>
  <col style="width: 180px">
  <col style="width: 120px">
  <col style="width: 700px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTurboQuantGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnTurboQuant默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

constexpr int64_t NUM_TOKENS = 2;
constexpr int64_t HEAD_DIM = 512;
constexpr int64_t PACKED_BYTES = HEAD_DIM / 2;
constexpr int64_t N_CENT = 16;

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void PrintOutResult(std::vector<int64_t>& shape, void** deviceAddr)
{
    auto size = GetShapeSize(shape);
    std::vector<uint8_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]), *deviceAddr,
                           size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    // 只打印首个y的前8个字节，每个字节承载2个4bit量化值
    for (int64_t i = 0; i < 8; i++) {
        LOG_PRINT("y[0][%ld] is: %u\n", i, static_cast<uint32_t>(resultData[i]));
    }
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
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor)
{
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
    CHECK_RET(*tensor != nullptr, LOG_PRINT("aclCreateTensor failed. dtype=%d, rank=%zu, bytes=%ld\n",
                                            static_cast<int>(dataType), shape.size(), size);
              return 1);
    return 0;
}

int main()
{
    // 1. （固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> latentShape = {NUM_TOKENS, HEAD_DIM};
    std::vector<int64_t> centroidsShape = {N_CENT};
    std::vector<int64_t> yShape = {NUM_TOKENS, PACKED_BYTES};
    std::vector<int64_t> scaleShape = {NUM_TOKENS};

    void* latentDeviceAddr = nullptr;
    void* centroidsDeviceAddr = nullptr;
    void* yDeviceAddr = nullptr;
    void* scaleDeviceAddr = nullptr;

    aclTensor* latent = nullptr;
    aclTensor* centroids = nullptr;
    aclTensor* y = nullptr;
    aclTensor* scale = nullptr;

    // latent已完成signed Hadamard旋转且未归一化，这里用一个确定性的取值填充
    std::vector<float> latentHostData(NUM_TOKENS * HEAD_DIM);
    for (int64_t i = 0; i < NUM_TOKENS * HEAD_DIM; i++) {
        latentHostData[i] = std::sin(static_cast<float>(i) * 0.01f) / std::sqrt(static_cast<float>(HEAD_DIM));
    }
    // 码本必须升序排列，取值为N(0, 1/HEAD_DIM)上的16个Lloyd-Max中心
    std::vector<float> centroidsHostData = {
        -0.1209128f, -0.0911112f, -0.0711246f, -0.0551360f, -0.0413207f, -0.0287497f, -0.0170049f, -0.0056868f,
        0.0054729f,  0.0168041f,  0.0285761f,  0.0410862f,  0.0549298f,  0.0710182f,  0.0911537f,  0.1203780f};
    std::vector<uint8_t> yHostData(NUM_TOKENS * PACKED_BYTES, 0);
    std::vector<uint16_t> scaleHostData(NUM_TOKENS, 0);

    // 创建latent aclTensor
    ret = CreateAclTensor(latentHostData, latentShape, &latentDeviceAddr, aclDataType::ACL_FLOAT, &latent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建centroids aclTensor
    ret = CreateAclTensor(centroidsHostData, centroidsShape, &centroidsDeviceAddr, aclDataType::ACL_FLOAT, &centroids);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建y aclTensor
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_UINT8, &y);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT16, &scale);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // 调用aclnnTurboQuant第一段接口
    ret = aclnnTurboQuantGetWorkspaceSize(latent, centroids, y, scale, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnTurboQuant第二段接口
    ret = aclnnTurboQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuant failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧；
    //    需要根据具体API的接口定义修改
    PrintOutResult(yShape, &yDeviceAddr);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(latent);
    aclDestroyTensor(centroids);
    aclDestroyTensor(y);
    aclDestroyTensor(scale);

    // 7. 释放device资源
    aclrtFree(latentDeviceAddr);
    aclrtFree(centroidsDeviceAddr);
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
