# aclnnSwigluGroupQuantGrad

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/master/activation/swiglu_group_quant_grad)

## 产品支持情况

| 产品                                                         |  是否支持   |
| :----------------------------------------------------------- |:-------:|
| <term>Ascend 950PR/Ascend 950DT</term>                             |    √     |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>     |    ×    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> |    ×    |
| <term>Atlas 200I/500 A2 推理产品</term>                      |    ×    |
| <term>Atlas 推理系列产品</term>                             |    ×    |
| <term>Atlas 训练系列产品</term>                              |    ×    |

## 功能说明

- 接口功能：SwigluGroupQuantGrad算子实现SwiGLU激活函数分组量化的反向梯度计算。用于计算输入梯度`grad_x`和权重梯度`grad_weight`。
- 算子支持范围：支持MoE场景（传入groupIndex）和非MoE场景（groupIndex传空），支持可选的Clamp反向传播掩码，支持可选的Weight梯度计算。
- 计算流程：
  - 步骤〇：GroupIndex处理（可选）→ 计算trunc
  - 步骤一：输入切分（将x切分为x0和x1）
  - 步骤二：Clamp处理（可选）
  - 步骤三：SwiGLU反向传播计算
  - 步骤四：Weight梯度计算（可选）
  - 步骤五：梯度拼接输出

- MoE场景GroupIndex处理公式：  
  $$  
    \text{trunc} = \sum_{g=0}^{G-1} \text{groupIndex}[g]  
  $$  
  其中：$G$ 为MoE专家分组数，后续所有步骤仅处理前 $\text{trunc}$ 行数据。

- 输入切分公式：  
  $$  
    \mathbf{x}_0[t, h] = \mathbf{x}[t, h], \quad h \in [0, H)  
  $$  
  
  $$  
    \mathbf{x}_1[t, h] = \mathbf{x}[t, h + H], \quad h \in [0, H)  
  $$  

- Clamp处理公式（当clamp_limit > 0时）：  
  $$  
    \mathbf{x}_0'[t, h] = \min(\mathbf{x}_0[t, h], c)  
  $$  
  
  $$  
    \mathbf{x}_1'[t, h] = \min(\max(\mathbf{x}_1[t, h], -c), c)  
  $$  
  其中 $c$ 为 `clamp_limit`。

- SiLU梯度公式：  
  $$  
    \frac{d\text{SiLU}}{d\mathbf{x}_0'} = \sigma(\mathbf{x}_0') \cdot \left(1 + \mathbf{x}_0' \cdot (1 - \sigma(\mathbf{x}_0'))\right)  
  $$  
  其中：$\sigma(\mathbf{x}_0') = \frac{1}{1 + e^{-\mathbf{x}_0'}}$

- 输入梯度计算公式：  
  $$  
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \mathbf{x}_1'[t, h] \cdot \frac{d\text{SiLU}}{d\mathbf{x}_0'}[t, h]  
  $$  
  
  $$  
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{y_0}[t, h] \cdot \text{SiLU}(\mathbf{x}_0'[t, h])  
  $$  
  其中：如果提供了weight，则 $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}} \cdot \mathbf{weight}$；如果未提供weight，则 $\mathbf{grad}_{y_0} = \mathbf{grad}_{\text{output}}$

- Weight梯度计算公式（可选）：  
  $$  
    \mathbf{grad}_{\text{weight}}[t] = \sum_{h=0}^{H-1} \mathbf{grad}_{\text{output}}[t, h] \cdot \mathbf{y}_{\text{origin}}[t, h]  
  $$  
  其中：$\mathbf{y}_{\text{origin}}$ 为SwiGLU前向传播的原始激活值输出，沿最后一维（H维度）求和。

- Clamp反向传播掩码公式（当clamp_limit > 0时）：  
  $$  
    \mathbf{grad}_{x_0}[t, h] = \mathbf{grad}_{x_0}[t, h] \cdot \mathbb{I}(\mathbf{x}_0[t, h] < c)  
  $$  
  
  $$  
    \mathbf{grad}_{x_1}[t, h] = \mathbf{grad}_{x_1}[t, h] \cdot \mathbb{I}(-c < \mathbf{x}_1[t, h] < c)  
  $$  
  其中 $\mathbb{I}$ 为指示函数。

- 梯度拼接与GroupIndex处理公式：  
  $$  
    \mathbf{grad}_x[t, h] = \begin{cases}  
    \mathbf{grad}_{x_0}[t, h] & h \in [0, H) \\  
    \mathbf{grad}_{x_1}[t, h-H] & h \in [H, 2H)  
    \end{cases}  
  $$  
  
  $$  
    \mathbf{grad}_x[t, :] = \mathbf{grad}_x[t, :] \cdot \mathbb{I}(t < \text{trunc})  
  $$  

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/两段式接口.md)，必须先调用"aclnnSwigluGroupQuantGradGetWorkspaceSize"接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用"aclnnSwigluGroupQuantGrad"接口执行计算。

```Cpp
aclnnStatus aclnnSwigluGroupQuantGradGetWorkspaceSize(
    const aclTensor   *gradY,
    const aclTensor   *x,
    const aclTensor   *weightOptional,
    const aclTensor   *yOriginOptional,
    const aclIntArray *groupIndexOptional,
    double             clampLimit,
    const aclTensor   *gradXOut,
    const aclTensor   *gradWeightOutOptional,
    uint64_t          *workspaceSize,
    aclOpExecutor     **executor)
```

```Cpp
aclnnStatus aclnnSwigluGroupQuantGrad(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnSwigluGroupQuantGradGetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 200px">
  <col style="width: 120px">
  <col style="width: 250px">
  <col style="width: 330px">
  <col style="width: 212px">
  <col style="width: 100px">
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
      <td>gradY（aclTensor*）</td>
      <td>输入</td>
      <td>梯度输出张量，来自下游层的梯度。</td>
      <td><ul><li>shape=[T, H]或[B, S, H]。</li><li>T为token数量，B为batch size，S为sequence length，H为hidden size。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>x（aclTensor*）</td>
      <td>输入</td>
      <td>前向传播的输入张量。</td>
      <td><ul><li>shape=[T, 2H]或[B, S, 2H]，最后一维必须为偶数。</li><li>最后一维的H与gradY的H对应。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>weightOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>MoE权重张量。</td>
      <td><ul><li>shape=[T, 1]或[B, S, 1]，需与gradY的第一维一致。</li><li>当提供weight时，必须同时提供yOrigin才能计算gradWeight。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>yOriginOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>SwiGLU前向传播的原始激活值输出。</td>
      <td><ul><li>shape=[T, H]或[B, S, H]，需与gradY的shape一致。</li><li>当提供weight时，必须同时提供yOrigin才能计算gradWeight。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>groupIndexOptional（aclTensor*）</td>
      <td>输入（可选）</td>
      <td>GroupIndex张量，动态核分配。</td>
      <td><ul><li>shape=[G]，dtype=INT64。</li><li>G为MoE专家分组数。</li><li>groupIndex内元素要求为非递减。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>√</td>
    </tr>
    <tr>
      <td>clampLimit（float）</td>
      <td>输入</td>
      <td>Clamp阈值。</td>
      <td><ul><li>取值范围≥0.0。</li><li>clampLimit=0表示不启用Clamp反向传播掩码。</li></ul></td>
      <td>FLOAT</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradXOut（aclTensor*）</td>
      <td>输出</td>
      <td>输入梯度张量。</td>
      <td><ul><li>shape=[T, 2H]或[B, S, 2H]，与x一致。</li><li>数据类型与gradY/x保持一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16、FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>gradWeightOutOptional（aclTensor*）</td>
      <td>输出（可选）</td>
      <td>权重梯度张量。</td>
      <td><ul><li>当提供weight时输出，shape=[T, 1]或[B, S, 1]。</li><li>数据类型为FLOAT。</li></ul></td>
      <td>FLOAT</td>
      <td>ND</td>
      <td>2-3</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn返回码.md)。

  第一段接口会完成入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed; width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>必选参数gradY/x/gradXOut为nullptr。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>161002</td>
      <td>gradY、x、weight等输入变量的数据类型和数据格式不在支持的范围内。</td>
    </tr>
    <tr>
      <td>ACLNN_ERR_INNER_TILING_ERROR</td>
      <td>561002</td>
      <td>多个输入tensor之间的shape信息不匹配、输入属性不在取值范围（详见参数说明）。</td>
    </tr>
  </tbody></table>

## aclnnSwigluGroupQuantGrad

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1149px"><colgroup>
  <col style="width: 173px">
  <col style="width: 124px">
  <col style="width: 852px">
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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnSwigluGroupQuantGradGetWorkspaceSize获取。</td>
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
  - aclnnSwigluGroupQuantGrad默认确定性实现。

- 输入shape约束：
  - x最后一维必须为偶数（$2H$）
  - gradY最后一维为 $H$，与x最后一维的一半对应
  - gradY与x的前n-1维shape必须一致

- 可选参数约束：
  - weight提供时，必须同时提供yOrigin才能计算gradWeight
  - weight的shape需与gradY的第一维一致
  - yOrigin的shape需与gradY一致

- 数据类型约束：
  - gradY、x、yOrigin、gradX数据类型必须一致（FLOAT、FLOAT16或BFLOAT16）
  - weight、gradWeight必须为FLOAT类型
  - groupIndex必须为INT64类型

- Clamp约束：
  - clampLimit必须 ≥ 0.0
  - clampLimit=0表示不启用Clamp反向传播掩码

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/编译与运行样例.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_swiglu_group_quant_grad.h"

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
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

template <typename T>
int CreateAclTensorWithValue(const std::vector<int64_t>& shape, void** deviceAddr,
                              aclDataType dataType, aclTensor** tensor, T value) {
  int64_t shapeSize = GetShapeSize(shape);
  std::vector<T> hostData(shapeSize, value);
  return CreateAclTensor(hostData, shape, deviceAddr, dataType, tensor);
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> gradYShape = {512, 512};
  std::vector<int64_t> xShape = {512, 1024};
  std::vector<int64_t> weightShape = {512, 1};
  std::vector<int64_t> yOriginShape = {512, 512};
  std::vector<int64_t> groupIndexShape = {256};
  std::vector<int64_t> gradXShape = {512, 1024};
  std::vector<int64_t> gradWeightShape = {512, 1};

  void* gradYDeviceAddr = nullptr;
  void* xDeviceAddr = nullptr;
  void* weightDeviceAddr = nullptr;
  void* yOriginDeviceAddr = nullptr;
  void* groupIndexDeviceAddr = nullptr;
  void* gradXDeviceAddr = nullptr;
  void* gradWeightDeviceAddr = nullptr;

  aclTensor* gradYTensor = nullptr;
  aclTensor* xTensor = nullptr;
  aclTensor* weightTensor = nullptr;
  aclTensor* yOriginTensor = nullptr;
  aclIntArray* groupIndexArray = nullptr;
  aclTensor* gradXTensor = nullptr;
  aclTensor* gradWeightTensor = nullptr;

  int64_t gradYSize = GetShapeSize(gradYShape);
  std::vector<float> gradYHostData(gradYSize, 1.0f);
  for (int64_t i = 0; i < gradYSize; i++) {
    gradYHostData[i] = static_cast<float>(i % 10) * 0.1f;
  }

  int64_t xSize = GetShapeSize(xShape);
  std::vector<float> xHostData(xSize, 1.0f);
  for (int64_t i = 0; i < xSize; i++) {
    xHostData[i] = static_cast<float>((i % 20) - 10) * 0.5f;
  }

  int64_t weightSize = GetShapeSize(weightShape);
  std::vector<float> weightHostData(weightSize, 1.0f);
  for (int64_t i = 0; i < weightSize; i++) {
    weightHostData[i] = static_cast<float>((i % 5) + 1) * 0.2f;
  }

  int64_t yOriginSize = GetShapeSize(yOriginShape);
  std::vector<float> yOriginHostData(yOriginSize, 1.0f);
  for (int64_t i = 0; i < yOriginSize; i++) {
    yOriginHostData[i] = static_cast<float>((i % 8) + 1) * 0.3f;
  }

  int64_t groupIndexSize = GetShapeSize(groupIndexShape);
  std::vector<int64_t> groupIndexHostData(groupIndexSize, 0);
  int64_t groupStride = 512 / 256;
  for (int64_t i = 0; i < groupIndexSize; i++) {
    groupIndexHostData[i] = i * groupStride;
  }

  ret = CreateAclTensor(gradYHostData, gradYShape, &gradYDeviceAddr, aclDataType::ACL_FLOAT16, &gradYTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(xHostData, xShape, &xDeviceAddr, aclDataType::ACL_FLOAT16, &xTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(weightHostData, weightShape, &weightDeviceAddr, aclDataType::ACL_FLOAT, &weightTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensor(yOriginHostData, yOriginShape, &yOriginDeviceAddr, aclDataType::ACL_FLOAT16, &yOriginTensor);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  std::vector<int64_t> groupArray = {256, 256};
  groupIndexArray = aclCreateIntArray(groupArray.data(), groupArray.size());
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensorWithValue<float>(gradXShape, &gradXDeviceAddr, aclDataType::ACL_FLOAT16, &gradXTensor, 0.0f);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  ret = CreateAclTensorWithValue<float>(gradWeightShape, &gradWeightDeviceAddr, aclDataType::ACL_FLOAT, &gradWeightTensor, 0.0f);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  float clampLimit = 1.0f;

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;

  ret = aclnnSwigluGroupQuantGradGetWorkspaceSize(gradYTensor, xTensor, weightTensor, yOriginTensor,
                                                    groupIndexArray, clampLimit, gradXTensor, gradWeightTensor,
                                                    &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGradGetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSwigluGroupQuantGrad(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSwigluGroupQuantGrad failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  auto gradXResultSize = GetShapeSize(gradXShape);
  std::vector<float> gradXResultData(gradXResultSize, 0);
  ret = aclrtMemcpy(gradXResultData.data(), gradXResultData.size() * sizeof(float),
                    gradXDeviceAddr, gradXResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradX result from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("gradX output (first 10 elements):\n");
  for (int64_t i = 0; i < 10 && i < gradXResultSize; i++) {
    LOG_PRINT("gradX[%ld] = %f\n", i, gradXResultData[i]);
  }

  auto gradWeightResultSize = GetShapeSize(gradWeightShape);
  std::vector<float> gradWeightResultData(gradWeightResultSize, 0);
  ret = aclrtMemcpy(gradWeightResultData.data(), gradWeightResultData.size() * sizeof(float),
                    gradWeightDeviceAddr, gradWeightResultSize * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradWeight result from device to host failed. ERROR: %d\n", ret); return ret);

  LOG_PRINT("gradWeight output (first 10 elements):\n");
  for (int64_t i = 0; i < 10 && i < gradWeightResultSize; i++) {
    LOG_PRINT("gradWeight[%ld] = %f\n", i, gradWeightResultData[i]);
  }

  aclDestroyTensor(gradYTensor);
  aclDestroyTensor(xTensor);
  aclDestroyTensor(weightTensor);
  aclDestroyTensor(yOriginTensor);
  aclDestroyTensor(gradXTensor);
  aclDestroyTensor(gradWeightTensor);

  aclrtFree(gradYDeviceAddr);
  aclrtFree(xDeviceAddr);
  aclrtFree(weightDeviceAddr);
  aclrtFree(yOriginDeviceAddr);
  aclrtFree(groupIndexDeviceAddr);
  aclrtFree(gradXDeviceAddr);
  aclrtFree(gradWeightDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }

  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();

  return 0;
}
```