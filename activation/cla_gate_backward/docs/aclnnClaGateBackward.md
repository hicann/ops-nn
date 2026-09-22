# aclnnClaGateBackward

[📄 查看源码](https://gitcode.com/cann/ops-nn/tree/9.2.0/activation/cla_gate_backward)

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- <term>Atlas A3系列产品</term>：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- <term>Atlas A2系列产品</term>：不支持
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

- 接口功能：融合算子，实现CLA（Cross-Layer Attention）gate merge的反向梯度计算。输入上游梯度G = ∂L/∂O、两路原始 Attention输出O_g/O_l与两路 gate logits z_g/z_l，一次性计算并返回四个梯度：两路Attention输出梯度（dO_g、dO_l）与两路 gate logits 梯度（dz_g、dz_l）。本接口不做梯度量化：输出保持输入精度（BF16/FP16），不生成FP8梯度。反向算子不接收前向sigmoid中间结果，s_g/s_l由算子内重新计算。

- 计算公式：

  记本rank上token数为T，attention head数为N，value head dim为D。

  **阶段1：Sigmoid重算（算子内重算，FP32域）**

  - 两路head-wise Sigmoid门控及其导数项（z_g、z_l为两路gate logits）：

    $$
    s\_g = \sigma(z\_g), \qquad s\_l = \sigma(z\_l), \qquad \sigma(x) = \frac{1}{1 + e^{-x}}
    $$

  **阶段2：Attention输出梯度（逐元素链式法则，s_g/s_l沿D维广播）**

  $$
  dO\_g = G \odot s\_g, \qquad dO\_l = G \odot s\_l
  $$

  **阶段3：gate logits梯度（链式法则 + Sigmoid导数，sum_d仅沿head_dim D归约，不跨token或head）**

  $$
  dz\_g = \left( \sum_{d=0}^{D-1} G \odot O\_g \right) \odot s\_g \odot (1 - s\_g), \qquad
  dz\_l = \left( \sum_{d=0}^{D-1} G \odot O\_l \right) \odot s\_l \odot (1 - s\_l)
  $$

  - G⊙O_g/G⊙O_l的乘积与沿D的归约均在 FP32域完成（一次pass内完成，无额外内存往返），以降低归约累积误差。
  - 无除法/指数溢出路径；NaN/Inf 输入按IEEE语义传播（无特殊保护需求）。

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnClaGateBackwardGetWorkspaceSize”接口获取计算所需workspace大小以及包含了算子计算流程的执行器，再调用“aclnnClaGateBackward”接口执行计算。

```cpp
aclnnStatus aclnnClaGateBackwardGetWorkspaceSize(
    const aclTensor *gradMerged,
    const aclTensor *globalAttn,
    const aclTensor *localAttn,
    const aclTensor *globalGateLogits,
    const aclTensor *localGateLogits,
    const char      *inputAttnLayout,
    const aclTensor *gradGlobalAttnOut,
    const aclTensor *gradLocalAttnOut,
    const aclTensor *gradGlobalGateLogitsOut,
    const aclTensor *gradLocalGateLogitsOut,
    uint64_t        *workspaceSize,
    aclOpExecutor   **executor)
```

```cpp
aclnnStatus aclnnClaGateBackward(
    void          *workspace,
    uint64_t       workspaceSize,
    aclOpExecutor *executor,
    aclrtStream    stream)
```

## aclnnClaGateBackwardGetWorkspaceSize

- **参数说明**

  <table style="undefined;table-layout: fixed; width: 1547px"><colgroup>
  <col style="width: 240px">
  <col style="width: 120px">
  <col style="width: 280px">
  <col style="width: 330px">
  <col style="width: 172px">
  <col style="width: 100px">
  <col style="width: 100px">
  <col style="width: 105px">
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
      <td>gradMerged（aclTensor*）</td>
      <td>输入</td>
      <td>输出投影反传到gate merge的梯度，公式中的G。</td>
      <td><ul><li>shape=[T, N, D]，D仅支持128或256。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>globalAttn（aclTensor*）</td>
      <td>输入</td>
      <td>前向Global/CLA分支Attention输出，公式中的O_g。</td>
      <td><ul><li>shape与gradMerged一致，为[T, N, D]。</li><li>数据类型与gradMerged一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>localAttn（aclTensor*）</td>
      <td>输入</td>
      <td>前向Local/SWA分支Attention输出，公式中的O_l。</td>
      <td><ul><li>shape与gradMerged一致，为[T, N, D]。</li><li>数据类型与gradMerged一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>√</td>
    </tr>
    <tr>
      <td>globalGateLogits（aclTensor*）</td>
      <td>输入</td>
      <td>前向Global gate logits，公式中的z_g。</td>
      <td><ul><li>shape=[T, N]，T、N与gradMerged一致。</li><li>数据类型与gradMerged一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>localGateLogits（aclTensor*）</td>
      <td>输入</td>
      <td>前向Local gate logits，公式中的z_l。</td>
      <td><ul><li>shape与globalGateLogits一致，为[T, N]。</li><li>数据类型与gradMerged一致。</li><li>不支持空Tensor。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>√</td>
    </tr>
    <tr>
      <td>inputAttnLayout（const char*）</td>
      <td>输入</td>
      <td>输入tensor的排布格式字符串。</td>
      <td><ul><li>当前仅支持"TND"。</li><li>默认值为"TND"，传空指针时按默认值处理。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradGlobalAttnOut（aclTensor*）</td>
      <td>输出</td>
      <td>Global Attention分支梯度，公式中的dO_g。</td>
      <td><ul><li>shape=[T, N, D]，与globalAttn一致。</li><li>数据类型与globalAttn一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradLocalAttnOut（aclTensor*）</td>
      <td>输出</td>
      <td>Local Attention分支梯度，公式中的dO_l。</td>
      <td><ul><li>shape=[T, N, D]，与localAttn一致。</li><li>数据类型与localAttn一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>3</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradGlobalGateLogitsOut（aclTensor*）</td>
      <td>输出</td>
      <td>Global gate logits梯度，公式中的dz_g。</td>
      <td><ul><li>shape=[T, N]，与globalGateLogits一致。</li><li>数据类型与globalGateLogits一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
      <td>-</td>
    </tr>
    <tr>
      <td>gradLocalGateLogitsOut（aclTensor*）</td>
      <td>输出</td>
      <td>Local gate logits梯度，公式中的dz_l。</td>
      <td><ul><li>shape=[T, N]，与localGateLogits一致。</li><li>数据类型与localGateLogits一致。</li></ul></td>
      <td>BFLOAT16、FLOAT16</td>
      <td>ND</td>
      <td>2</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

  第一段接口会完成下列入参校验，出现以下场景时报错：

  <table style="undefined;table-layout: fixed;width: 1155px"><colgroup>
  <col style="width: 253px">
  <col style="width: 140px">
  <col style="width: 762px">
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
      <td>gradMerged、globalAttn、localAttn、globalGateLogits、localGateLogits、gradGlobalAttnOut、gradLocalAttnOut、gradGlobalGateLogitsOut、gradLocalGateLogitsOut存在空指针。</td>
    </tr>
    <tr>
      <td rowspan="3">ACLNN_ERR_PARAM_INVALID</td>
      <td rowspan="3">161002</td>
      <td>5路输入的数据类型不在支持的范围之内。</td>
    </tr>
    <tr>
      <td>5路输入数据类型不一致，或4路输出数据类型与对应输入不一致。</td>
    </tr>
    <tr>
      <td>gradMerged、globalAttn、localAttn、globalGateLogits、localGateLogits的shape维度数不满足要求，或同类输入shape不一致，或4路输出shape与对应输入不一致。</td>
    </tr>
  </tbody></table>

## aclnnClaGateBackward

- **参数说明**

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnClaGateBackwardGetWorkspaceSize获取。</td>
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

- **返回值**

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 确定性计算：aclnnClaGateBackward默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

- <term>Ascend 950PR&950DT系列产品</term>：

  ```cpp
  #include <cstdint>
  #include <cstring>
  #include <iostream>
  #include <vector>
  #include "acl/acl.h"
  #include "aclnnop/aclnn_cla_gate_backward.h"

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

  int64_t GetShapeSize(const std::vector<int64_t>& shape)
  {
      int64_t shapeSize = 1;
      for (auto dim : shape) {
          shapeSize *= dim;
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
      return ACL_SUCCESS;
  }

  void Finalize(int32_t deviceId, aclrtStream stream)
  {
      (void)aclrtDestroyStream(stream);
      (void)aclrtResetDevice(deviceId);
      (void)aclFinalize();
  }

  bool CheckHardwareSupport()
  {
      const char* socName = aclrtGetSocName();
      if (socName == nullptr) {
          LOG_PRINT("Warning: Cannot get SOC name, skip hardware check\n");
          return true;
      }

      LOG_PRINT("Current SOC: %s\n", socName);
      if (strstr(socName, "Ascend950") != nullptr || strstr(socName, "ascend950") != nullptr) {
          return true;
      }

      LOG_PRINT("Warning: ClaGateBackward only supports Ascend950, current SOC '%s' is not supported. Skip test.\n",
                socName);
      return false;
  }

  // 将 float 转换为 bfloat16 的 uint16_t 表示。
  uint16_t FloatToBf16(float f)
  {
      uint32_t bits;
      std::memcpy(&bits, &f, sizeof(uint32_t));
      return static_cast<uint16_t>(bits >> 16);
  }

  // 将 bfloat16 的 uint16_t 位模式还原为 float。
  float Bf16ToFloat(uint16_t bf16)
  {
      uint32_t bits = static_cast<uint32_t>(bf16) << 16;
      float f;
      std::memcpy(&f, &bits, sizeof(float));
      return f;
  }

  template <typename T>
  int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                      aclDataType dataType, aclTensor** tensor)
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

      *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                                shape.data(), shape.size(), *deviceAddr);
      return ACL_SUCCESS;
  }

  int main()
  {
      int32_t deviceId = 0;
      aclrtStream stream;
      auto ret = Init(deviceId, &stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

      if (!CheckHardwareSupport()) {
          LOG_PRINT("\n=== Test SKIPPED (hardware not supported) ===\n");
          Finalize(deviceId, stream);
          return ACL_SUCCESS;
      }

      // token=8, head=64, dim=256：三路 TND 输入 [T, N, D]，两路 gate logits [T, N]。
      int64_t tokenCount = 8;
      int64_t headNum = 64;
      int64_t headDim = 256;
      std::vector<int64_t> tndShape = {tokenCount, headNum, headDim};
      std::vector<int64_t> logitsShape = {tokenCount, headNum};

      void* gradMergedDeviceAddr = nullptr;
      void* globalAttnDeviceAddr = nullptr;
      void* localAttnDeviceAddr = nullptr;
      void* globalGateLogitsDeviceAddr = nullptr;
      void* localGateLogitsDeviceAddr = nullptr;
      void* gradGlobalAttnOutDeviceAddr = nullptr;
      void* gradLocalAttnOutDeviceAddr = nullptr;
      void* gradGlobalGateLogitsOutDeviceAddr = nullptr;
      void* gradLocalGateLogitsOutDeviceAddr = nullptr;

      aclTensor* gradMerged = nullptr;
      aclTensor* globalAttn = nullptr;
      aclTensor* localAttn = nullptr;
      aclTensor* globalGateLogits = nullptr;
      aclTensor* localGateLogits = nullptr;
      aclTensor* gradGlobalAttnOut = nullptr;
      aclTensor* gradLocalAttnOut = nullptr;
      aclTensor* gradGlobalGateLogitsOut = nullptr;
      aclTensor* gradLocalGateLogitsOut = nullptr;

      // 输入用 BF16 存储（uint16_t 承载）；输出 host 缓冲仅用于回读。
      std::vector<uint16_t> gradMergedHostData(GetShapeSize(tndShape), FloatToBf16(0.5f));
      std::vector<uint16_t> globalAttnHostData(GetShapeSize(tndShape), FloatToBf16(1.0f));
      std::vector<uint16_t> localAttnHostData(GetShapeSize(tndShape), FloatToBf16(0.5f));
      std::vector<uint16_t> globalGateLogitsHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
      std::vector<uint16_t> localGateLogitsHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
      std::vector<uint16_t> gradGlobalAttnOutHostData(GetShapeSize(tndShape), FloatToBf16(0.0f));
      std::vector<uint16_t> gradLocalAttnOutHostData(GetShapeSize(tndShape), FloatToBf16(0.0f));
      std::vector<uint16_t> gradGlobalGateLogitsOutHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));
      std::vector<uint16_t> gradLocalGateLogitsOutHostData(GetShapeSize(logitsShape), FloatToBf16(0.0f));

      ret = CreateAclTensor(gradMergedHostData, tndShape, &gradMergedDeviceAddr, ACL_BF16, &gradMerged);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(globalAttnHostData, tndShape, &globalAttnDeviceAddr, ACL_BF16, &globalAttn);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(localAttnHostData, tndShape, &localAttnDeviceAddr, ACL_BF16, &localAttn);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(globalGateLogitsHostData, logitsShape, &globalGateLogitsDeviceAddr, ACL_BF16,
                            &globalGateLogits);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(localGateLogitsHostData, logitsShape, &localGateLogitsDeviceAddr, ACL_BF16, &localGateLogits);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(gradGlobalAttnOutHostData, tndShape, &gradGlobalAttnOutDeviceAddr, ACL_BF16,
                            &gradGlobalAttnOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(gradLocalAttnOutHostData, tndShape, &gradLocalAttnOutDeviceAddr, ACL_BF16, &gradLocalAttnOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(gradGlobalGateLogitsOutHostData, logitsShape, &gradGlobalGateLogitsOutDeviceAddr, ACL_BF16,
                            &gradGlobalGateLogitsOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);
      ret = CreateAclTensor(gradLocalGateLogitsOutHostData, logitsShape, &gradLocalGateLogitsOutDeviceAddr, ACL_BF16,
                            &gradLocalGateLogitsOut);
      CHECK_RET(ret == ACL_SUCCESS, return ret);

      // input_attn_layout：当前仅支持 "TND"。
      const char* inputAttnLayout = "TND";

      uint64_t workspaceSize = 0;
      aclOpExecutor* executor = nullptr;
      ret = aclnnClaGateBackwardGetWorkspaceSize(
          gradMerged, globalAttn, localAttn, globalGateLogits, localGateLogits, inputAttnLayout, gradGlobalAttnOut,
          gradLocalAttnOut, gradGlobalGateLogitsOut, gradLocalGateLogitsOut, &workspaceSize, &executor);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateBackwardGetWorkspaceSize failed. ERROR: %d\n", ret);
                return ret);

      void* workspaceAddr = nullptr;
      if (workspaceSize > 0) {
          ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
          CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
      }

      ret = aclnnClaGateBackward(workspaceAddr, workspaceSize, executor, stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnClaGateBackward failed. ERROR: %d\n", ret); return ret);

      ret = aclrtSynchronizeStream(stream);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

      auto tndSize = GetShapeSize(tndShape);
      auto logitsSize = GetShapeSize(logitsShape);
      std::vector<uint16_t> gradGlobalAttnOutData(tndSize, 0);
      std::vector<uint16_t> gradLocalAttnOutData(tndSize, 0);
      std::vector<uint16_t> gradGlobalGateLogitsOutData(logitsSize, 0);
      std::vector<uint16_t> gradLocalGateLogitsOutData(logitsSize, 0);

      ret = aclrtMemcpy(gradGlobalAttnOutData.data(), tndSize * sizeof(uint16_t), gradGlobalAttnOutDeviceAddr,
                        tndSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradGlobalAttnOut failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(gradLocalAttnOutData.data(), tndSize * sizeof(uint16_t), gradLocalAttnOutDeviceAddr,
                        tndSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradLocalAttnOut failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(gradGlobalGateLogitsOutData.data(), logitsSize * sizeof(uint16_t),
                        gradGlobalGateLogitsOutDeviceAddr, logitsSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradGlobalGateLogitsOut failed. ERROR: %d\n", ret); return ret);
      ret = aclrtMemcpy(gradLocalGateLogitsOutData.data(), logitsSize * sizeof(uint16_t),
                        gradLocalGateLogitsOutDeviceAddr, logitsSize * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy gradLocalGateLogitsOut failed. ERROR: %d\n", ret); return ret);

      for (int64_t i = 0; i < 8 && i < tndSize; i++) {
          LOG_PRINT("gradGlobalAttnOut[%ld] is: %f\n", i, static_cast<double>(Bf16ToFloat(gradGlobalAttnOutData[i])));
      }
      for (int64_t i = 0; i < 8 && i < logitsSize; i++) {
          LOG_PRINT("gradGlobalGateLogitsOut[%ld] is: %f\n", i,
                    static_cast<double>(Bf16ToFloat(gradGlobalGateLogitsOutData[i])));
      }

      aclDestroyTensor(gradMerged);
      aclDestroyTensor(globalAttn);
      aclDestroyTensor(localAttn);
      aclDestroyTensor(globalGateLogits);
      aclDestroyTensor(localGateLogits);
      aclDestroyTensor(gradGlobalAttnOut);
      aclDestroyTensor(gradLocalAttnOut);
      aclDestroyTensor(gradGlobalGateLogitsOut);
      aclDestroyTensor(gradLocalGateLogitsOut);

      aclrtFree(gradMergedDeviceAddr);
      aclrtFree(globalAttnDeviceAddr);
      aclrtFree(localAttnDeviceAddr);
      aclrtFree(globalGateLogitsDeviceAddr);
      aclrtFree(localGateLogitsDeviceAddr);
      aclrtFree(gradGlobalAttnOutDeviceAddr);
      aclrtFree(gradLocalAttnOutDeviceAddr);
      aclrtFree(gradGlobalGateLogitsOutDeviceAddr);
      aclrtFree(gradLocalGateLogitsOutDeviceAddr);

      if (workspaceSize > 0) {
          aclrtFree(workspaceAddr);
      }

      Finalize(deviceId, stream);
      return ACL_SUCCESS;
  }
  ```
