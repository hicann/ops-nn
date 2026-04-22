# aclnnRotateQuant

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/matmul/rotate_quant)

## 产品支持情况

| 产品                                                     | 是否支持 |
| :------------------------------------------------------- | :------: |
| Ascend 950PR/Ascend 950DT                   |    ×     |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |
| Atlas 200I/500 A2 推理产品                  |    ×     |
| Atlas 推理系列产品                          |    ×     |
| Atlas 训练系列产品                          |    ×     |

## 功能说明

- 接口功能：对张量x进行旋转变换，再执行对称动态量化。
- 计算公式：
    1.  旋转变换
        $$
        Y = (x.\text{reshape}(*,k) @ \text{rotation}).\text{reshape}(m, n)
        $$
        其中：$\mathbf{x} \in \mathbb{R}^{m \times n}$，$\mathbf{Y} \in \mathbb{R}^{m \times n}$，$\mathbf{rotation} \in \mathbb{R}^{k \times k}$。

    2.  对称动态量化（pertoken 逐行量化）
        - 缩放因子计算（逐行计算）
            $$
            s_i = \frac{\max_{j \in [0,\ n-1]} |Y_{i,j}|}{C_{\text{MAX}}}
            $$
            其中：$s_i$ 是第 $i$ 行的缩放因子；$C_{\text{MAX}}$ 是量化范围最大值，int8 取 127，quint4x2 取 7。
        - 量化计算
            $$
            y_{i,j} = \frac{Y_{i,j}}{s_i}
            $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用“aclnnRotateQuantGetWorkspaceSize”接口获取入参并根据流程计算所需workspace大小，再调用“aclnnRotateQuant”接口执行计算。

```c++
aclnnStatus aclnnRotateQuantGetWorkspaceSize(
  const aclTensor   *x,
  const aclTensor   *rotation,
  float              alpha,
  aclTensor         *yOut,
  aclTensor         *scaleOut,
  uint64_t          *workspaceSize,
  aclOpExecutor    **executor)
```

```c++
aclnnStatus aclnnRotateQuant(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnRotateQuantGetWorkspaceSize

- **参数说明**
  
  <table style="undefined;table-layout: fixed; width: 1447px"><colgroup>
  <col style="width: 162px">
  <col style="width: 121px">
  <col style="width: 332px">
  <col style="width: 200px">
  <col style="width: 275px">
  <col style="width: 118px">
  <col style="width: 138px">
  <col style="width: 101px">
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>x(aclTensor*)</td>
      <td>输入</td>
      <td>待旋转量化的输入张量</td>
      <td>支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>rotation(aclTensor*)</td>
      <td>输入</td>
      <td>旋转矩阵</td>
      <td>支持非连续Tensor，不支持空Tensor。</td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
  <tr>
      <td>alpha(float32_t)</td>
      <td>输入</td>
      <td>clamp需要限制的范围的缩放系数</td>
      <td>如果为0.0，不做clamp处理。</td>
      <td>float</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>yOut(aclTensor*)</td>
      <td>输出</td>
      <td>量化后的输出张量</td>
      <td>输出Tensor需预先分配。</td>
      <td>INT8、INT32、FLOAT4_E2M1</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>scaleOut(aclTensor*)</td>
      <td>输出</td>
      <td>动态量化计算出的缩放系数</td>
      <td>输出Tensor需预先分配。</td>
      <td>FLOAT32、FLOAT8_E8M0</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>workspaceSize(uint64_t)</td>
      <td>输出</td>
      <td>返回需要在Device侧申请的workspace大小</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>executor(aclOpExecutor*)</td>
      <td>输出</td>
      <td>返回op执行器，包含算子计算流程</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
  </tbody>
  </table>
- **返回值**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。
  
  第一段接口完成入参校验，出现以下场景时报错：
  
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 288px">
  <col style="width: 125px">
  <col style="width: 742px">
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
      <td>x、rotation或yOut、scaleOut是空指针。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_PARAM_INVALID</td>
      <td>161002</td>
      <td>x和rotation的数据类型和格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561001</td>
      <td>内部创建OpExecutor执行器失败。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561003</td>
      <td>内部算子调用或数据拷贝操作失败。</td>
    </tr>  
  </tbody>
  </table>

## aclnnRotateQuant

- **参数说明**
  
  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 173px">
  <col style="width: 133px">
  <col style="width: 849px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnMatrixTransformRopeGetWorkspaceSize获取。</td>
    </tr>
    <tr>
      <td>executor</td>
      <td>输入</td>
      <td>op执行器，包含了算子计算流程。</td>
    </tr>
    <tr>
      <td>stream</td>
      <td>输入</td>
      <td>指定执行任务的Stream流。</td>
    </tr>
  </tbody>
  </table>
- **返回值**
  
  返回aclnnStatus状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

## 约束说明

- <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>、<term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>：
  - x的shape为(M, N)，rotation的shape为(K, K)。
  - rotation的shape必须是方阵(K, K)。
  - x第二维的长度(N)必须是K的整数倍，N必须可以整除8。
  - 当yOut的输出类型为int8时，y的shape必须和x相同(M, N)。
  - 当yOut的输出类型为int32时，y的shape必须为(M, N//8)。
  - x和rotation的数据类型必须相同。
  - scaleOut的shape必须是(M)。
  - N的范围为[128, 16000]。
  - K的范围为[16, 1024]。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <memory>
#include <vector>
#include <random>

#include "acl/acl.h"
#include "../op_host/op_api/aclnn_rotate_quant.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define CHECK_FREE_RET(cond, return_expr) \
    do {                                  \
        if (!(cond)) {                    \
            Finalize(deviceId, stream);   \
            return_expr;                  \
        }                                 \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

void Finalize(int32_t deviceId, aclrtStream stream)
{
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
}

template <typename T>
int CreateAclTensor(
    const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr, aclDataType dataType,
    aclTensor** tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
        strides[i] = shape[i + 1] * strides[i + 1];
    }

    *tensor = aclCreateTensor(
        shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND, shape.data(), shape.size(),
        *deviceAddr);
    return 0;
}

std::vector<uint16_t> GenerateRandomBf16Data(int64_t size, unsigned int seed = 42)
{
    std::vector<uint16_t> data(size);
    std::mt19937 gen(seed);
    for (int64_t i = 0; i < size; i++) {
        int sign = (gen() % 2) ? 0x8000 : 0;
        int exp = 0x3F00 + (gen() % 2);
        int mant = gen() % 128;
        data[i] = sign | exp | mant;
    }
    return data;
}

std::vector<uint16_t> GenerateIdentityMatrix(int64_t K)
{
    std::vector<uint16_t> matrix(K * K, 0);
    uint16_t bf16One = 0x3F80;
    for (int64_t i = 0; i < K; i++) {
        matrix[i * K + i] = bf16One;
    }
    return matrix;
}

int aclnnRotateQuantTest(int32_t deviceId, aclrtStream& stream)
{
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // x: (M, N), rot: (K, K), y: (M, N), scale: (M,)
    // Constraints: rot must be square, N % K == 0, N % 8 == 0
    int64_t M = 1024;
    int64_t N = 256;
    int64_t K = 64;

    std::vector<int64_t> xShape = {M, N};
    std::vector<int64_t> rotShape = {K, K};
    std::vector<int64_t> yShape = {M, N};
    std::vector<int64_t> scaleShape = {M};

    // 创建x aclTensor (BF16)
    auto xHostData = GenerateRandomBf16Data(M * N, 42);
    void* xDeviceAddr = nullptr;
    aclTensor* xTensor = nullptr;
    ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_BF16, &xTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> xTensorPtr(xTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> xAddrPtr(xDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Create x tensor failed.\n"); return ret);

    // 创建rot aclTensor (BF16, 必须为方阵)
    auto rotHostData = GenerateIdentityMatrix(K);
    void* rotDeviceAddr = nullptr;
    aclTensor* rotTensor = nullptr;
    ret = CreateAclTensor(rotHostData, rotShape, &rotDeviceAddr, aclDataType::ACL_BF16, &rotTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> rotTensorPtr(rotTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> rotAddrPtr(rotDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Create rot tensor failed.\n"); return ret);

    // 创建y aclTensor (INT8)
    std::vector<int8_t> yHostData(M * N, 0);
    void* yDeviceAddr = nullptr;
    aclTensor* yTensor = nullptr;
    ret = CreateAclTensor(yHostData, yShape, &yDeviceAddr, aclDataType::ACL_INT8, &yTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> yTensorPtr(yTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> yAddrPtr(yDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Create y tensor failed.\n"); return ret);

    // 创建scale aclTensor (FLOAT32)
    std::vector<float> scaleHostData(M, 0.0f);
    void* scaleDeviceAddr = nullptr;
    aclTensor* scaleTensor = nullptr;
    ret = CreateAclTensor(scaleHostData, scaleShape, &scaleDeviceAddr, aclDataType::ACL_FLOAT, &scaleTensor);
    std::unique_ptr<aclTensor, aclnnStatus (*)(const aclTensor*)> scaleTensorPtr(scaleTensor, aclDestroyTensor);
    std::unique_ptr<void, aclError (*)(void*)> scaleAddrPtr(scaleDeviceAddr, aclrtFree);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("Create scale tensor failed.\n"); return ret);

    // 调用aclnnRotateQuant第一段接口
    float alpha = 0.0f;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    ret = aclnnRotateQuantGetWorkspaceSize(xTensor, rotTensor, alpha, yTensor, scaleTensor,
                                            &workspaceSize, &executor);
    CHECK_FREE_RET(ret == ACL_SUCCESS,
                   LOG_PRINT("aclnnRotateQuantGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    std::unique_ptr<void, aclError (*)(void*)> workspaceAddrPtr(nullptr, aclrtFree);
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
        workspaceAddrPtr.reset(workspaceAddr);
    }

    // 调用aclnnRotateQuant第二段接口
    ret = aclnnRotateQuant(workspaceAddr, workspaceSize, executor, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuant failed. ERROR: %d\n", ret); return ret);

    // （固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 获取输出的值，将device侧内存上的结果拷贝至host侧
    auto ySize = GetShapeSize(yShape);
    std::vector<int8_t> yResult(ySize, 0);
    ret = aclrtMemcpy(yResult.data(), ySize * sizeof(int8_t), yDeviceAddr, ySize * sizeof(int8_t),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy y result failed.\n"); return ret);

    auto scaleSize = GetShapeSize(scaleShape);
    std::vector<float> scaleResult(scaleSize, 0.0f);
    ret = aclrtMemcpy(scaleResult.data(), scaleSize * sizeof(float), scaleDeviceAddr, scaleSize * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("copy scale result failed.\n"); return ret);

    // 打印部分结果
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(yResult.size()); i++) {
        LOG_PRINT("y[%ld] = %d\n", i, static_cast<int>(yResult[i]));
    }
    for (int64_t i = 0; i < 5 && i < static_cast<int64_t>(scaleResult.size()); i++) {
        LOG_PRINT("scale[%ld] = %f\n", i, scaleResult[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = aclnnRotateQuantTest(deviceId, stream);
    CHECK_FREE_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnRotateQuantTest failed. ERROR: %d\n", ret); return ret);

    Finalize(deviceId, stream);
    return 0;
}

```
