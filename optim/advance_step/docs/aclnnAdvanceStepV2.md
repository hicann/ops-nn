# aclnnAdvanceStepV2

## 产品支持情况

<!-- npu="950" id1 -->
- <term>Ascend 950PR&950DT系列产品</term>：支持
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

- 接口功能：

  vLLM是一个高性能的LLM推理和服务框架，专注于优化大规模语言模型的推理效率。它的核心特点包括PageAttention和高效内存管理。advance_step算子的主要作用是推进推理步骤，即在每个生成步骤中更新模型的状态并生成新的inputTokens、inputPositions、seqLens和slotMapping，为vLLM的推理提升效率。

- 计算公式：

  V2投机采样路径（specToken与acceptedNum同时传入，numSeqs == numQueries，specNum为specToken第二维长度，tokenEachReqs = specNum + 1）。

  $$
  blockTablesStride = blockTables.stride(0)
  $$

  记序列序号 i ∈ [0, numSeqs)、序列内token序号 j ∈ [0, tokenEachReqs)，更新后位置 nextPos_{i,j} = inputPositions[i * tokenEachReqs + j] + acceptedNum[i] + 1：

  $$
  inputTokens[i * tokenEachReqs] = lastToken_i
  $$

  其中lastToken_i取sampledTokenIds[i]行末补-1后首个-1的前一元素；行首为-1时 lastToken_i = -1。

  $$
  inputTokens[i * tokenEachReqs + 1 : (i+1) * tokenEachReqs] = specToken[i, :]
  $$

  $$
  inputPositions[i * tokenEachReqs + j] = nextPos_{i,j}
  $$

  $$
  seqLens[i * tokenEachReqs + j] = nextPos_{i,j} + 1
  $$

  $$
slotMapping[i * tokenEachReqs + j] = blockTablesFlat[⌊nextPos_{i,j} / blockSize⌋ + blockTablesStride * i] * blockSize + nextPos_{i,j} \% blockSize
  $$

## 函数原型

每个算子分为[两段式接口](../../../docs/zh/context/two_phase_api.md)，必须先调用“aclnnAdvanceStepV2GetWorkspaceSize”接口获取入参并根据计算流程计算所需workspace大小，再调用“aclnnAdvanceStepV2”接口执行计算。

```Cpp
aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(
  aclTensor *inputTokensRef,
  const aclTensor *sampledTokenIds,
  aclTensor *inputPositionsRef,
  aclTensor *seqLensRef,
  aclTensor *slotMappingRef,
  const aclTensor *blockTables,
  const aclTensor *specTokenOptional,
  const aclTensor *acceptedNumOptional,
  int64_t          numSeqs,
  int64_t          numQueries,
  int64_t          blockSize,
  uint64_t        *workspaceSize,
  aclOpExecutor  **executor)
```

```Cpp
aclnnStatus aclnnAdvanceStepV2(
  void            *workspace,
  uint64_t         workspaceSize,
  aclOpExecutor   *executor,
  aclrtStream      stream)
```

## aclnnAdvanceStepV2GetWorkspaceSize

- **参数说明：**

  <table style="undefined;table-layout: fixed; width: 1500px"><colgroup>
  <col style="width: 271px">
  <col style="width: 115px">
  <col style="width: 250px">
  <col style="width: 300px">
  <col style="width: 177px">
  <col style="width: 104px">
  <col style="width: 138px">
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
      <td>inputTokens（aclTensor*）</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出inputTokens，用于更新vLLM模型中的token值。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[numSeqs * (1+specNum)]，specNum为specToken第二维长度。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>sampledTokenIds（aclTensor*）</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于储存tokenID，公式中的输入sampledTokenIds。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape第一维长度与numSeqs一致，第二维长度为1+specNum。</li><li>取值范围为非负整数（0为合法值），-1为无接受token哨兵值。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
     <tr>
      <td>inputPositions（aclTensor*）</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出inputPositions，用于记录token的index。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[numSeqs * (1+specNum)]。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>seqLens（aclTensor*）</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，用于记录不同blockIdx下seq的长度，公式中的输入/输出seqLens。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[numSeqs * (1+specNum)]。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>slotMapping（aclTensor*）</td>
      <td>输入/输出</td>
      <td>待进行AdvanceStepV2计算的入参/出参，公式中的输出slotMapping，用于将token值在序列中的位置映射到物理位置。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape为[numSeqs * (1+specNum)]。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
    <tr>
      <td>blockTables（aclTensor*）</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录不同blockIdx下block的物理块编号，公式中的输入blockTables。</td>
      <td><ul><li>不支持空Tensor。</li><li>shape第一维长度与numSeqs一致，第二维长度blockTablesStride大于0，且须覆盖更新后位置的block索引：V2路径按每序列更新后位置（inputPositions值+acceptedNum[i]+1）计，须满足其小于blockTablesStride*blockSize；回落V1路径时按更新前seqLens最大值/blockSize计。</li><li>取值范围为非负整数（0为合法值）。</li><li>更新后位置越出blockTables覆盖范围属未定义输入域，kernel对查表下标越界做防御性检查并写入0。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>specTokenOptional（aclTensor*）</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录当前投机模型的token的index。</td>
      <td><ul><li>可选输入：传入空Tensor（numel==0）视为未传入，此时回落V1路径。</li><li>shape第一维长度与numSeqs一致，第二维长度为specNum。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>2</td>
      <td>×</td>
    </tr>
      <tr>
      <td>acceptedNumOptional（aclTensor*）</td>
      <td>输入</td>
      <td>待进行AdvanceStepV2计算的入参，用于记录每个request接受的投机的数量。</td>
      <td><ul><li>可选输入：传入空Tensor（numel==0）视为未传入，此时回落V1路径。</li><li>shape长度与numSeqs一致。</li><li>取值范围为非负整数（0为合法值）。</li></ul></td>
      <td>INT64</td>
      <td>ND</td>
      <td>1</td>
      <td>×</td>
    </tr>
      <tr>
      <td>numSeqs（int64_t）</td>
      <td>输入</td>
      <td>记录输入的seq数量，大小与seqLens的长度一致。</td>
      <td><ul><li>取值范围是大于0且小于200000000的正整数。</li><li>V2路径（specToken与acceptedNum同时传入）：numSeqs的值与输入numQueries的值一致。</li><li>可选输入未传入或仅传入其一时回落V1路径：此时要求numSeqs大于numQueries，inputTokens等1-D输入长度为numSeqs，sampledTokenIds为[numQueries, 1]，适用约束见aclnnAdvanceStep。</li></ul></td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
      <tr>
      <td>numQueries（int64_t）</td>
      <td>输入</td>
      <td>记录输入的Query的数量，大小与sampledTokenIds第一维的长度一致。</td>
      <td>取值范围是大于0的正整数。</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
      <td>-</td>
    </tr>
     <tr>
      <td>blockSize（int64_t）</td>
      <td>输入</td>
      <td>每个block的大小，对应公式中的blockSize。</td>
      <td>取值范围是大于0的正整数。</td>
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
  </tbody>
  </table>

- **返回值：**

aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

第一段接口会完成入参校验，出现以下场景时报错：

<table style="undefined;table-layout: fixed; width: 1048px"><colgroup>
<col style="width: 319px">
<col style="width: 108px">
<col style="width: 621px">
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
    <td>传入的inputTokens、sampledTokenIds、inputPositions、seqLens、slotMapping、blockTables是空指针。可选输入specTokenOptional、acceptedNumOptional允许传空指针（等价于未传入，走V1路径）。</td>
  </tr>
  <tr>
    <td>ACLNN_ERR_PARAM_INVALID</td>
    <td>161002</td>
    <td>inputTokens、sampledTokenIds、inputPositions、seqLens、slotMapping、blockTables、specToken、acceptedNum的数据类型不在支持的范围之内。</td>
  </tr>
  <tr>
    <td rowspan="8">aclnnAdvanceStepV2GetWorkspaceSize failed</td>
    <td rowspan="8">561002</td>
    <td>（V2路径）输入inputTokens、inputPositions、seqLens、slotMapping的shape长度不为numSeqs * (1 + specNum)。</td>
  </tr>
  <tr>
    <td>（V2路径）输入sampledTokenIds的shape的第一维长度与numSeqs不一致，或者shape的第二维长度不为1 + specNum。</td>
  </tr>
  <tr>
    <td>（V2路径）输入specToken非空时，shape的第一维长度与numSeqs不一致，或者shape的维度不为2。</td>
  </tr>
  <tr>
    <td>（V2路径）输入acceptedNum的shape长度与numSeqs不一致。</td>
  </tr>
  <tr>
    <td>输入blockTables的shape的第一维长度与numSeqs不一致，或者shape的第二维长度不为正数。</td>
  </tr>
  <tr>
    <td>（V2路径）输入numSeqs的值不等于输入numQueries的值。</td>
  </tr>
  <tr>
    <td>numSeqs、numQueries、blockSize不是大于0的正整数，或者numSeqs不小于200000000。</td>
  </tr>
  <tr>
    <td>（V2路径）specNum过大导致每序列输入缓冲超出可用UB预算（Invalid ubRequired）；specNum规模上限取决于芯片可用UB，须保证tokenEachReqs对应的每序列缓冲在UB预算内。</td>
  </tr>
 </tbody>
</table>

## aclnnAdvanceStepV2

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
      <td>在Device侧申请的workspace大小，由第一段接口aclnnAdvanceStepV2GetWorkspaceSize获取。</td>
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

  aclnnStatus：返回状态码，具体参见[aclnn返回码](../../../docs/zh/context/aclnn_return_code.md)。

## 约束说明

- 接口路径约束：
  - V2路径要求specToken与acceptedNum同时传入且均非空，specNum（specToken第二维长度）大于0，且numSeqs与numQueries相等。
  - 仅传入specToken或acceptedNum中的一个，或两者均未传入（含传入空Tensor；specNum为0时specToken为空Tensor）时，走V1路径；V1路径要求numSeqs大于numQueries，不满足时返回561002。
  - 无投机解码场景可评估使用aclnnAdvanceStep，但V1路径的numSeqs与numQueries约束与V2路径不同，不能直接等价替换。
- Host侧校验约束（tiling阶段校验，不满足时报错终止）：
  - 数据类型：所有输入、输出仅支持INT64。
  - 数据格式：所有输入仅支持ND。
  - 属性约束：numSeqs、numQueries、blockSize必须为大于0的整数，且numSeqs必须小于200000000；V2路径要求numSeqs等于numQueries。
  - 形状约束：inputTokens、inputPositions、seqLens、slotMapping的shape必须为[numSeqs * (1 + specNum)]（specNum为specToken第二维长度）；sampledTokenIds的shape必须为[numSeqs, (1 + specNum)]；specToken的shape必须为[numSeqs, specNum]；acceptedNum的shape必须为[numSeqs]；blockTables为2-D [numSeqs, blockTablesStride]，第二维长度必须大于0。
  - 可选输入约束：specToken与acceptedNum须同时传入或同时不传入；传入空Tensor视为未传入（回落V1路径）；仅传入其一时走V1路径，该输入不参与计算。
  - 所有必选输入不支持空Tensor。
- 用户保证约束（涉及数据内容或内存布局，Host侧与kernel侧均不校验数据值，由调用方保证，违反时可能导致计算错误或未定义行为）：
  - 所有输入Tensor必须连续，不支持非连续Tensor。
  - 入参默认值域：输入数据值须为符合业务语义的非负整数（token ID、序列长度、block编号、物理槽位等，0为合法值），token ID按INT64非负全域取值；sampledTokenIds另支持-1哨兵值（表示无接受token，行首为-1时lastToken为-1）；其中inputTokens、slotMapping为原地输出张量（输入值不参与计算，V2路径全量覆盖写回）。
  - blockTables查表覆盖约束：blockTables第二维须覆盖更新后位置的block索引，即满足 pos + acceptedNum + 1 < blockTablesStride * blockSize（pos为更新前的inputPositions值）。
  - 越界防护行为：V2路径对契约外查表列号（q ∉ [0, blockTablesStride)，含跨行平坦下标与负值）防御性写0——块号与余数同时屏蔽，不发生GM越界读；该输入域的计算语义未定义。
- V2路径规模约束：
  - specNum受每序列输入缓冲的UB预算限制：tokenEachReqs=specNum+1过大时tiling以Invalid ubRequired失败（561002拒绝），specNum规模上限取决于芯片可用UB。
- V2路径取值域约束：
  - blockSize与blockTablesStride须小于2^31（tiling校验，违反时报561002）。
  - 中间计算以int32参与运算：inputPositions、acceptedNum的值及slotMapping计算结果须在int32表示域内。token ID（sampledTokenIds、specToken）按INT64非负全域取值，仅-1哨兵检索经fp32（fp32不可精确表示的非-1值不会误判为哨兵，边界外取值已由用例验证）。
- 内存约束：本算子不使用workspace（GetWorkspaceSize返回的workspaceSize为0）。
- 确定性计算：
  - aclnnAdvanceStepV2默认确定性实现。

## 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../../docs/zh/context/compile_and_run_sample.md)。

```Cpp
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_advance_step_v2.h"
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

void PrintOutResult(std::vector<int64_t> &shape, void** deviceAddr) {
    auto size = GetShapeSize(shape);
    std::vector<int64_t> resultData(size, 0);
    auto ret = aclrtMemcpy(resultData.data(), resultData.size() * sizeof(resultData[0]),
                        *deviceAddr, size * sizeof(resultData[0]), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", ret); return);
    for (int64_t i = 0; i < size; i++) {
    LOG_PRINT("advanceStepV2 result[%ld] is: %ld\n", i, resultData[i]);
    }
}

int Init(int64_t deviceId, aclrtStream* stream) {
    // 固定写法，资源初始化
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
    LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret);
    (void)aclFinalize();
    return ret;
    }
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

    // 调用aclCreateTensor接口创建aclTensor
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int main() {
    // 1.（固定写法）device/stream初始化，参考acl API手册
    // 根据自己的实际device填写deviceId
    int32_t deviceId = 0;
    aclrtStream stream;
    auto ret = Init(deviceId, &stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

    // 2. 构造输入与输出，需要根据API的接口自定义构造
    std::vector<int64_t> input1Shape = {72};
    std::vector<int64_t> input2Shape = {8,9};
    std::vector<int64_t> input3Shape = {72};
    std::vector<int64_t> input4Shape = {72};
    std::vector<int64_t> input5Shape = {72};
    std::vector<int64_t> input6Shape = {8,1000};
    std::vector<int64_t> input7Shape = {8,8};
    std::vector<int64_t> input8Shape = {8};
    std::vector<int64_t> input1HostData(8*9, 1);
    std::vector<int64_t> input2HostData(8*9, 1);
    std::vector<int64_t> input3HostData(8*9, 1);
    std::vector<int64_t> input4HostData(8*9, 8);
    std::vector<int64_t> input5HostData(8*9, 1);
    std::vector<int64_t> input6HostData(8*1000, 1);
    std::vector<int64_t> input7HostData(8*8, 1);
    std::vector<int64_t> input8HostData(8, 1);

    void* input1DeviceAddr = nullptr;
    aclTensor* input1 = nullptr;
    void* input2DeviceAddr = nullptr;
    aclTensor* input2 = nullptr;
    void* input3DeviceAddr = nullptr;
    aclTensor* input3 = nullptr;
    void* input4DeviceAddr = nullptr;
    aclTensor* input4 = nullptr;
    void* input5DeviceAddr = nullptr;
    aclTensor* input5 = nullptr;
    void* input6DeviceAddr = nullptr;
    aclTensor* input6 = nullptr;
    void* input7DeviceAddr = nullptr;
    aclTensor* input7 = nullptr;
    void* input8DeviceAddr = nullptr;
    aclTensor* input8 = nullptr;
    // 创建input aclTensor
    ret = CreateAclTensor(input1HostData, input1Shape, &input1DeviceAddr, aclDataType::ACL_INT64, &input1);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input2HostData, input2Shape, &input2DeviceAddr, aclDataType::ACL_INT64, &input2);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input3HostData, input3Shape, &input3DeviceAddr, aclDataType::ACL_INT64, &input3);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input4HostData, input4Shape, &input4DeviceAddr, aclDataType::ACL_INT64, &input4);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input5HostData, input5Shape, &input5DeviceAddr, aclDataType::ACL_INT64, &input5);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input6HostData, input6Shape, &input6DeviceAddr, aclDataType::ACL_INT64, &input6);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input7HostData, input7Shape, &input7DeviceAddr, aclDataType::ACL_INT64, &input7);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = CreateAclTensor(input8HostData, input8Shape, &input8DeviceAddr, aclDataType::ACL_INT64, &input8);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    int64_t numseq = 8;
    int64_t specnum = 8;
    int64_t blocksize = 8;

    // 3. 调用CANN算子库API，需要修改为具体的API名称
    uint64_t workspaceSize = 16 * 1024 * 1024;
    aclOpExecutor* executor;

    // 调用aclnnAdvanceStepV2第一段接口
    ret = aclnnAdvanceStepV2GetWorkspaceSize(
    input1,input2,input3,input4,input5,input6,input7,input8,
    numseq,numseq,blocksize,
    &workspaceSize,
    &executor);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

    // 根据第一段接口计算出的workspaceSize申请device内存
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
    }

    // 调用aclnnAdvanceStepV2第二段接口
    ret = aclnnAdvanceStepV2(
    workspaceAddr,
    workspaceSize,
    executor,
    stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnAdvanceStepV2 failed. ERROR: %d\n", ret); return ret);

    // 4.（固定写法）同步等待任务执行结束
    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

    // 5. 获取输出的值，将device侧内存上的结果复制至host侧，需要根据具体API的接口定义修改
    PrintOutResult(input1Shape, &input1DeviceAddr);
    PrintOutResult(input2Shape, &input2DeviceAddr);
    PrintOutResult(input3Shape, &input3DeviceAddr);
    PrintOutResult(input4Shape, &input4DeviceAddr);
    PrintOutResult(input5Shape, &input5DeviceAddr);

    // 6. 释放aclTensor，需要根据具体API的接口定义修改
    aclDestroyTensor(input1);
    aclDestroyTensor(input2);
    aclDestroyTensor(input3);
    aclDestroyTensor(input4);
    aclDestroyTensor(input5);
    aclDestroyTensor(input6);
    aclDestroyTensor(input7);
    aclDestroyTensor(input8);

    // 7.释放device资源，需要根据具体API的接口定义修改
    aclrtFree(input1DeviceAddr);
    aclrtFree(input2DeviceAddr);
    aclrtFree(input3DeviceAddr);
    aclrtFree(input4DeviceAddr);
    aclrtFree(input5DeviceAddr);
    aclrtFree(input6DeviceAddr);
    aclrtFree(input7DeviceAddr);
    aclrtFree(input8DeviceAddr);

    if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
    }
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();

    return 0;
}
```
