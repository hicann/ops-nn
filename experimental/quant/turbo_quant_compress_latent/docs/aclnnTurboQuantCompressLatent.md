# aclnnTurboQuantCompressLatent

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/experimental/quant/turbo_quant_compress_latent)

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |

## 功能说明

- 接口功能：把MLA（Multi-head Latent Attention）的KV latent按token压缩成TurboQuant 4bit slot，用于降低长序列推理的KV cache显存占用。
- 计算公式：
  - 对每个token求L2范数：

    $$
    norm_i = \sqrt{\sum_{d} latent_{i,d}^2 + \epsilon}, \quad \epsilon = 10^{-16}
    $$

  - 按范数归一化：

    $$
    u_{i,d} = latent_{i,d} \times \frac{1}{norm_i}
    $$

  - 量化到最近的码本中心，等价于数出该值越过了多少条相邻中心的中点边界：

    $$
    nibble_{i,d} = \sum_{b=0}^{14} \left[ u_{i,d} \ge \frac{centroids_b + centroids_{b+1}}{2} \right]
    $$

  - 相邻两个维度的4bit索引打包进一个字节，低位对应偶数维：

    $$
    out_{i,k} = nibble_{i,2k}\ |\ (nibble_{i,2k+1} \ll 4), \quad 0 \le k < headDim/2
    $$

## 函数原型

每个算子分为[两段式接口](../../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnTurboQuantCompressLatentGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnTurboQuantCompressLatent”接口执行计算。

```Cpp
aclnnStatus aclnnTurboQuantCompressLatentGetWorkspaceSize(
  const aclTensor* latent,
  const aclTensor* centroids,
  int64_t          outputMode,
  const aclTensor* out,
  uint64_t*        workspaceSize,
  aclOpExecutor**  executor)
```

```Cpp
aclnnStatus aclnnTurboQuantCompressLatent(
  void          *workspace,
  uint64_t       workspaceSize,
  aclOpExecutor *executor,
  aclrtStream    stream)
```

## aclnnTurboQuantCompressLatentGetWorkspaceSize

- **参数说明：**

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
      <td>latent（aclTensor*）</td>
      <td>输入</td>
      <td>表示待压缩的KV latent。对应公式中的latent。</td>
      <td><ul><li>支持空Tensor，此时numTokens为0，算子不写出任何数据。</li><li>shape为[numTokens, headDim]。</li><li>要求已完成signed Hadamard旋转且未归一化。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>centroids（aclTensor*）</td>
      <td>输入</td>
      <td>表示4bit量化码本。对应公式中的centroids。</td>
      <td><ul><li>不支持空Tensor。</li><li>元素总数必须为16。</li><li>必须按升序排列。</li></ul></td>
      <td>FLOAT32</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>outputMode（int64_t）</td>
      <td>输入</td>
      <td>表示输出slot模式。</td>
      <td><ul><li>支持0或1，默认值为0。</li><li>0：320字节对齐布局，scale为latent的L2范数。</li><li>1：258字节紧凑布局，scale为latent范数除以所选码本向量范数。</li></ul></td>
      <td>INT64</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>out（aclTensor*）</td>
      <td>输出</td>
      <td>表示压缩后的slot。对应公式中的out。</td>
      <td><ul><li>支持空Tensor。</li><li>outputMode为0时，slotSize = ceil((headDim / 2 + 2) / 64) * 64；outputMode为1时，slotSize = headDim / 2 + 2。</li><li>headDim为512时slotSize分别为320和258。</li></ul></td>
      <td>UINT8</td>
      <td>ND</td>
      <td>2</td>
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
  </tbody>
  </table>

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

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
      <td>传入的latent、centroids、out是空指针。</td>
    </tr>
    <tr>
      <td rowspan="4">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="4">161002</td>
      <td>latent、centroids、out的数据类型或数据格式不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>latent不是2维，或headDim不等于512。</td>
    </tr>
    <tr>
      <td>centroids的元素总数不等于16。</td>
    </tr>
    <tr>
      <td>outputMode不是0或1，或out的shape与outputMode不匹配。</td>
    </tr>
  </tbody></table>

## aclnnTurboQuantCompressLatent

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnTurboQuantCompressLatentGetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：
  - aclnnTurboQuantCompressLatent默认确定性实现。

- headDim当前仅支持512（MLA的kv_lora_rank）。slot布局的推导本身是按headDim泛化的，放开其他取值需要先补充硬件验证。
- centroids必须升序排列，算子不对其做排序或校验；乱序会导致中点边界不单调，量化结果不可预期。
- outputMode默认值0保持原320字节slot布局；outputMode为1时输出258字节compact corrected slot，须与对应读取布局的融合算子配套使用。
- 量化索引取最近的码本中心，平局时取较大的索引；归一化后超出码本范围的取值饱和到边界桶。
- 非有限输入的行为由归一化流程自然导出且确定：该token含NaN时全部索引为0、范数存NaN；含±INF时INF所在维度索引为0、其余为8、范数存+INF。
- slot只用2字节float16存放范数，若输入的L2范数超过65504，存储的范数为+INF，读取侧无法还原幅值。MLA的KV latent经RMSNorm后范数在1附近，不会触及该边界。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <cmath>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_compress_latent.h"

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
constexpr int64_t SLOT_SIZE = 320; // alignUp(HEAD_DIM / 2 + 2, 64)
constexpr int64_t N_CENT = 16;
constexpr int64_t OUTPUT_MODE = 0;

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
    // 只打印首个slot的前8个字节，每个字节承载2个4bit量化值
    for (int64_t i = 0; i < 8; i++) {
        LOG_PRINT("slot[0][%ld] is: %u\n", i, static_cast<uint32_t>(resultData[i]));
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
    std::vector<int64_t> slotShape = {NUM_TOKENS, SLOT_SIZE};

    void* latentDeviceAddr = nullptr;
    void* centroidsDeviceAddr = nullptr;
    void* slotDeviceAddr = nullptr;

    aclTensor* latent = nullptr;
    aclTensor* centroids = nullptr;
    aclTensor* slot = nullptr;

    // latent已完成signed Hadamard旋转且未归一化，这里用一个确定性的取值填充
    std::vector<float> latentHostData(NUM_TOKENS * HEAD_DIM);
    for (int64_t i = 0; i < NUM_TOKENS * HEAD_DIM; i++) {
        latentHostData[i] = std::sin(static_cast<float>(i) * 0.01f) / std::sqrt(static_cast<float>(HEAD_DIM));
    }
    // 码本必须升序排列，取值为N(0, 1/HEAD_DIM)上的16个Lloyd-Max中心
    std::vector<float> centroidsHostData = {
        -0.1209128f, -0.0911112f, -0.0711246f, -0.0551360f, -0.0413207f, -0.0287497f, -0.0170049f, -0.0056868f,
        0.0054729f,  0.0168041f,  0.0285761f,  0.0410862f,  0.0549298f,  0.0710182f,  0.0911537f,  0.1203780f};
    std::vector<uint8_t> slotHostData(NUM_TOKENS * SLOT_SIZE, 0);

    // 创建latent aclTensor
    ret = CreateAclTensor(latentHostData, latentShape, &latentDeviceAddr, aclDataType::ACL_FLOAT, &latent);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建centroids aclTensor
    ret = CreateAclTensor(centroidsHostData, centroidsShape, &centroidsDeviceAddr, aclDataType::ACL_FLOAT, &centroids);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    // 创建slot aclTensor
    ret = CreateAclTensor(slotHostData, slotShape, &slotDeviceAddr, aclDataType::ACL_UINT8, &slot);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor;

    // 调用aclnnTurboQuantCompressLatent第一段接口
    ret = aclnnTurboQuantCompressLatentGetWorkspaceSize(
        latent, centroids, OUTPUT_MODE, slot, &workspaceSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuantCompressLatentGetWorkspaceSize failed. ERROR: %d\n", ret);
              return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnTurboQuantCompressLatent第二段接口
    ret = aclnnTurboQuantCompressLatent(workspaceAddr, workspaceSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnTurboQuantCompressLatent failed. ERROR: %d\n", ret); return ret);

    // 4. （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果拷贝至host侧，需要根据具体API的接口定义修改
    PrintOutResult(slotShape, &slotDeviceAddr);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(latent);
    aclDestroyTensor(centroids);
    aclDestroyTensor(slot);

    // 7. 释放device资源
    aclrtFree(latentDeviceAddr);
    aclrtFree(centroidsDeviceAddr);
    aclrtFree(slotDeviceAddr);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
