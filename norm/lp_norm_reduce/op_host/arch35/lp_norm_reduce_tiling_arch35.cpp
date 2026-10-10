/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_host/arch35/lp_norm_reduce_tiling_arch35.cpp
// =============================================================================
//
// ROLE: Host-side tiling implementation for LpNormReduce on arch35 (Ascend 950).
//   本文件为公共 Tiling 实现，按 docs/LpNormReduce/design/HostTiling.md
//   （公共 tiling 量计算，「Tiling 整体结构」节主入口源码）与
//   docs/LpNormReduce/design/BranchRoute.md（分支判定与优先级）落码：
//   - TilingFuncLpNormReduce 编排全链：平台/输入/attr 解析 →
//     ±inf 空 R 拒绝 + 空 tensor 三分类（EMPTY_A / EMPTY_R → tilingKey=2 短路）→
//     合轴四步（DropSizeOneAxes → FuseAxis → PadLeadingOneA → PadRIfPureA）→
//     UB 切分三步（ComputeAUbFactor → ComputeRUbFactor → ExpandAIfRFullyLoaded）→
//     A 多核均分（ComputeFusedALoopSplit）+ R 迭代数（ComputeRLoopCnt）→
//     Group 判定（ShouldUseGroup：aLoopCntTotal ≤ coreNum/2 且 rLoopCntTotal ≥ 2
//     → ComputeGroupSplit + SetScheduleMode(1)）→ UB 尺寸（ComputeUbSizes）→
//     FillAndLogTilingData 全字段填充 → SetTilingKey / SetBlockDim / SetWorkspaceSize。
//   - TilingData 写 docs/LpNormReduce/design/TilingData.md 约定的
//     LpNormReduceTilingData（base/group 共用）与 LpNormReduceEmptyTilingData（empty 独立）。
//   - tilingKey 经 GET_TPL_TILING_KEY 位编码（arch35/lp_norm_reduce_tiling_key.h：
//     isGroup 占 bit0、isEmptyTensor 占 bit1）：base=0 / group=1 / empty=2；
//     判定顺序与 BranchRoute.md「分支优先级与互斥」一致。
//   - TilingPrepareForLpNormReduce 与 IMPL_OP_OPTILING(LpNormReduce)
//     .Tiling(...).TilingParse<LpNormReduceCompileInfo>(...) 注册链原样保留
//     （平台信息 coreNum/ubSize 编译期载入 CompileInfo）。
//
//   事实源：spec.yaml（公式 / dtype / determinism）、REQUIREMENTS.md（canndev 原生
//   约束：p=±inf 哨兵不得对空维度归约）、TilingData.md（字段集）、TilingKey.md
//   （tilingKey 集 {0,1,2} 与位编码）、BranchRoute.md（路由判定与优先级）、
//   InferShapeDtype.md（axes / keepdim 语义）。
//
// =============================================================================

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "register/op_def_registry.h"                          // IMPL_OP_OPTILING
#include "op_common/log/log.h"                                 // OP_LOGE, OP_LOGI, OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/math_util.h"                  // CeilDiv / CeilAlign / FloorAlign
#include "op_common/op_host/util/platform_util.h"              // platform_ascendc::PlatformAscendC
#include "graph/utils/type_utils.h"                            // ge::TypeUtils::DataTypeToSerialString
#include "securec.h"                                           // memset_s, EOK
#include "../../op_kernel/arch35/lp_norm_reduce_tiling_data.h" // TilingData 汇总结构体（base/group + empty）
#include "../../op_kernel/arch35/lp_norm_reduce_tiling_key.h"  // ASCENDC_TPL 声明 + GET_TPL_TILING_KEY
#include "lp_norm_reduce_tiling_arch35.h"                      // LpNormReduceCompileInfo

namespace optiling {

// ---------------------------------------------------------------------------
// 常量（与 docs/LpNormReduce/design/HostTiling.md、Interface.md OpDef 声明序一致）
// ---------------------------------------------------------------------------
constexpr int64_t INPUT_X_IDX = 0;       // 唯一数据输入 x 的下标
constexpr int64_t ATTR_P_IDX = 0;        // attr p（Int，默认 2）
constexpr int64_t ATTR_AXES_IDX = 1;     // attr axes（ListInt，默认 {} = 全归约）
constexpr int64_t ATTR_KEEP_DIM_IDX = 2; // attr keepdim（Bool，默认 false）
constexpr int64_t ATTR_EPSILON_IDX = 3;  // attr epsilon（Float，默认 1e-12，reduce 段不参与计算）

// UB 预算系数（HostTiling.md「UB 划分预分析（六步法）」：
// P_pre(2) + P_pre_ext(1) + cacheBuf(1) + P_post(1) = 5 个 UB buffer）
constexpr int64_t P_PRE = 2;                   // preReducePhase 存活 buffer 数（preInBuf + preReduceResult）
constexpr int64_t P_PRE_EXT = 1;               // preReduceResultTail（Phase A 尾块 + 兼 sharedTmpBuffer）
constexpr int64_t P_POST = 1;                  // postReducePhase outBuf
constexpr int64_t CACHE_BUF_BYTES = 16 * 1024; // 二分缓存树 buffer，恒 16KB
constexpr int64_t FP32_BYTES = 4;              // fp32 中间累加使 b16 也按 4 字节预算 UB

// A/R 规整不变量（偶位 A / 奇位 R，A 起头严格交替）
constexpr int64_t AXIS_INTERVAL = 2;
constexpr int32_t MIN_AXIS_NUM = 2;                // 合轴后最少轴数（AR / RA）
constexpr int32_t MAX_AXIS_NUM = MAX_PATTERN_RANK; // 9（TilingData.md）

// Group 触发判定（BranchRoute.md：aLoopCntTotal ≤ coreNum/GROUP_CORE_RATIO 且 rLoopCntTotal ≥ 2）
constexpr int64_t GROUP_CORE_RATIO = 2;

// p 值域哨兵（spec.yaml：p ∈ [0, 2147483647] ∪ {-2147483648}，±inf 整型哨兵特例）
constexpr int64_t INF_POS_SENTINEL = 2147483647;
constexpr int64_t INF_NEG_SENTINEL = -2147483648LL;

// ---------------------------------------------------------------------------
// Tiling 工作状态（HostTiling.md「硬件参数获取」节 XxxCtx；一次 TilingFunc 调用内的
// 全部中间量，按生产阶段分区）
// ---------------------------------------------------------------------------
enum class LpNormReduceEmptyKind { NORMAL, EMPTY_A, EMPTY_R };

struct LpNormReduceCtx {
    // ─── 平台参数（5 个硬件参数全走 platform 接口，禁止写死）───
    int64_t coreNum = 0;       // 可用 AIV 核数（GetCoreNumAiv）
    int64_t ubSize = 0;        // 单核 UB 字节数（GetCoreMemSize(UB)）
    int64_t blockSize = 0;     // UB block 字节数（GetUbBlockSize，950 = 32）
    int64_t cacheLineSize = 0; // cache line 字节数（GetCacheLineSize，950 = 256）
    int64_t vectorSize = 0;    // 向量寄存器字节数（GetVRegSize）

    // ─── 输入信息 ───
    ge::DataType xDtype = ge::DT_FLOAT;
    int64_t dtypeSize = 0;    // sizeof(D_T)（元素数取整用）
    int64_t maxDtypeSize = 0; // max(sizeof(D_T), 4)（字节容量预算用；fp32 累加）
    std::vector<int64_t> xShape;
    std::vector<int64_t> reduceAxes; // 归一化后的归约轴（升序）
    int64_t pOrder = 2;              // 范数阶数 p 原值（TilingData 唯一算子自定义字段）
    bool keepDim = false;

    // ─── 空 tensor 分类 ───
    LpNormReduceEmptyKind emptyKind = LpNormReduceEmptyKind::NORMAL;
    int64_t aTotalEmpty = 0; // EMPTY_R 的输出元素总数（∏ 非零 A 轴）

    // ─── A/R 规整模式 ───
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduceAxis;
    int32_t axisNum = 0;
    bool isTailR = false; // 偶数轴 → tail-R（最内轴 R），奇数轴 → tail-A

    // ─── UB 切分参数 ───
    int32_t aSplitIdx = 0;
    int32_t rSplitIdx = 0;
    int64_t aUbFactor = 0;       // valid：A 维实际元素数
    int64_t rUbFactor = 0;       // valid：R 维实际元素数
    int64_t rUbFactorAlign = 0;  // padded：UB 行 stride
    int64_t innerAProdAlign = 0; // padded：含最内 burst-tail A 的 CeilAlign 积
    int64_t innerRProdAlign = 0; // padded：含最内 burst-tail R 的 CeilAlign 积

    // ─── A 方向多核参数 ───
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ─── R 方向迭代数 ───
    int64_t rLoopCntTotal = 0;

    // ─── UB buffer 尺寸（字节）───
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;

    // ─── Group 2D 分核参数 ───
    bool isGroup = false;
    int64_t rGroupCnt = 0;
};

// ---------------------------------------------------------------------------
// 硬件参数获取（范式 §1.1：5 个硬件参数全走 platform 接口、禁止写死，每个返回值
// 非 0 校验，报错用 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON）
// ---------------------------------------------------------------------------
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ctx.coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(ctx.coreNum == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.coreNum",
                                                         "GetCoreNumAiv returned 0"),
                return ge::GRAPH_FAILED);

    uint64_t ub = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
    ctx.ubSize = static_cast<int64_t>(ub);
    OP_CHECK_IF(ctx.ubSize == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.ubSize",
                                                         "GetCoreMemSize(UB) returned 0"),
                return ge::GRAPH_FAILED);

    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));       // 不要写 32
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context)); // 不要写 256/512
    ctx.vectorSize = static_cast<int64_t>(Ops::Base::GetVRegSize(context));
    OP_CHECK_IF(
        ctx.blockSize == 0 || ctx.cacheLineSize == 0 || ctx.vectorSize == 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.blockSize/cacheLineSize/vectorSize",
                                                 "BlockSize=" + std::to_string(ctx.blockSize) +
                                                     ", cacheLineSize=" + std::to_string(ctx.cacheLineSize) +
                                                     " or vectorSize=" + std::to_string(ctx.vectorSize) + " is 0"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 输入输出获取及校验（范式 §1.2）：rank ∈ [0,8]、单轴长度 ≥ 0（负 dim 为动态 shape
// 防御）、dtype ∈ {DT_FLOAT16, DT_FLOAT, DT_BF16}；dtypeSize 取整用 sizeof(D_T)、
// maxDtypeSize = max(sizeof(D_T), 4)（fp32 中间累加使 b16 也按 4 字节预算 UB）。
// ---------------------------------------------------------------------------
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xS = xShapePtr->GetStorageShape();
    const size_t rank = xS.GetDimNum();
    // 本算子 rank ∈ [0,8]（批 b-5 对齐 canndev/changwei rank 0-8：rank-0 标量 +
    // 空 axes = 单元素全归约，A/R 规整第 4 步自动补 [A1,R1]；rank-0 + 非空 axes
    // 由 axes 值域校验拒绝——[-0,0) 为空值域）
    OP_CHECK_IF(
        rank > 8,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "x", std::to_string(static_cast<int64_t>(rank)), "rank of x must be in [0, 8]"),
        return ge::GRAPH_FAILED);
    ctx.xShape.clear();
    for (size_t i = 0; i < rank; ++i) {
        OP_CHECK_IF(
            xS.GetDim(i) < 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "x", std::to_string(xS.GetDim(i)),
                "dim[" + std::to_string(i) + "] is negative (dynamic shape must be instantiated before tiling)"),
            return ge::GRAPH_FAILED);
        ctx.xShape.push_back(xS.GetDim(i));
    }

    // numel 溢出总闸（codex 审查缺陷 #2；对齐 changwei IsConcreteShape 除法式上界
    // 保护）：非零维连乘超 int64 表示域 → 干净拒绝。该乘积是全部子集乘积（aTotal/
    // TotalAProd/stride 前缀积/outerAProd/outerR/FuseAxis 融合积）的上界，杜绝
    // signed overflow UB 流入 stride/循环数/workspace/GM 偏移计算。必须同时约束
    // 字节偏移：kernel 的 GM stride 会乘 sizeof(DT)，最宽 DT 为 fp32。
    // 仅按元素数留余量仍会放行 [2, (INT64_MAX-4096)/2]，使 stride*4 溢出。
    // 除以最大元素字节数也为下游 CeilDiv/CeilAlign/尾块端点留下足够余量。
    constexpr int64_t kNumelLimit = (std::numeric_limits<int64_t>::max() - 4096) / FP32_BYTES;
    int64_t nonZeroDimProduct = 1;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = ctx.xShape[i];
        if (dim == 0) {
            continue; // 零维不放大子集乘积（changwei 同口径）
        }
        OP_CHECK_IF(nonZeroDimProduct > kNumelLimit / dim,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(dim),
                        "dim[" + std::to_string(i) +
                            "] makes the nonzero-dimension byte span exceed the int64 range (tensor too large)"),
                    return ge::GRAPH_FAILED);
        nonZeroDimProduct *= dim;
    }

    auto xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();
    // dtype 三档（批 b-6 对齐 canndev：fp16/fp32/bf16；bf16 与 fp16 同为 2 字节，
    // 走同一 B16 搬运/对齐路径，kernel 侧 fp32 累加）
    const std::set<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16};
    OP_CHECK_IF(supportedDtypes.count(ctx.xDtype) == 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(),
                                                      "only DT_FLOAT16/DT_FLOAT/DT_BF16 are supported on ascend950"),
                return ge::GRAPH_FAILED);
    ctx.dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(ctx.xDtype));
    ctx.maxDtypeSize = std::max(ctx.dtypeSize, FP32_BYTES); // fp32 中间累加使 b16 也按 4 字节预算 UB

    std::string shapeStr = "[";
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(ctx.xShape[i]);
    }
    shapeStr += "]";
    OP_LOGI(context, "input: xShape=%s xDtype=%d dtypeSize=%ld", shapeStr.c_str(), static_cast<int>(ctx.xDtype),
            ctx.dtypeSize);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// axes 归一化（属性版，范式 §2 六步）：空 axes（nullptr 或 size==0）→ full reduce
// 展开 [0..rank-1]；负索引 +rank 归一到 [0, rank)；越界（转正前校验）报错，重复轴按首次出现去重。
// ---------------------------------------------------------------------------
static bool PushOneAxis(const char* nodeName, int64_t v, int64_t idx, int64_t xRank, std::set<int64_t>& seen,
                        std::vector<int64_t>& axesOut)
{
    if (v < -xRank || v >= xRank) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "axes", std::to_string(v),
                                              "axes[" + std::to_string(idx) + "] out of range [-" +
                                                  std::to_string(xRank) + ", " + std::to_string(xRank) + ")");
        return false;
    }
    const int64_t norm = (v < 0) ? (v + xRank) : v;
    if (!seen.insert(norm).second) {
        // 重复轴按首次出现去重（批 b-2，对齐 canndev/changwei 语义）：转正后同轴
        // 与首次出现等价，静默去重不再报错（codex 审查契约缩窄项）。
        return true;
    }
    axesOut.push_back(norm);
    return true;
}

// 解析 axes 属性（ListInt，恒 int64）+ p/keepdim/epsilon 属性到 ctx：
// p 值域校验 [0, 2147483647] ∪ {-2147483648}（±inf 整型哨兵特例）；
// 末尾 sort 升序保证合轴确定性（bitwise_reproducible: true）。
static ge::graphStatus ParseAttrs(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    auto attrs = context->GetAttrs(); // 注意：非 GetInputTensor——axes 为属性
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t xRank = static_cast<int64_t>(ctx.xShape.size());
    ctx.reduceAxes.clear();

    // ── attr p（idx 0，默认 2）──
    const int64_t* attrP = attrs->GetAttrPointer<int64_t>(ATTR_P_IDX);
    const int64_t p = (attrP == nullptr) ? 2 : (*attrP);
    // p 值域对齐 canndev 公共契约（p >= 0 完整非负 int64；changwei tiling.cpp:317 同口径）：
    // 有限 p ∈ [0, INT64_MAX]，其中 2147483647 保留为 +inf 哨兵（max|x|）、
    // -2147483648 为 -inf 哨兵（min|x|）；有限大 p 由 kernel 二进制快速幂承担（≤63 轮）。
    OP_CHECK_IF(
        p < 0 && p != INF_NEG_SENTINEL,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "p", std::to_string(p),
                                              "p must be non-negative (full int64) or -2147483648 (-inf sentinel); "
                                              "2147483647 is reserved as the +inf sentinel"),
        return ge::GRAPH_FAILED);
    ctx.pOrder = p; // TilingData 唯一算子自定义字段（kernel 按 pOrder 分发五个计算分支）

    // ── attr axes（idx 1，默认 {} = 全归约）──
    const gert::TypedContinuousVector<int64_t>* axesVec = attrs->GetListInt(ATTR_AXES_IDX);
    const int64_t axesNum = (axesVec == nullptr) ? 0 : static_cast<int64_t>(axesVec->GetSize());
    if (axesNum == 0) {
        ctx.reduceAxes.resize(static_cast<size_t>(xRank));
        for (int64_t i = 0; i < xRank; ++i) {
            ctx.reduceAxes[static_cast<size_t>(i)] = i;
        }
    } else {
        const int64_t* data = axesVec->GetData();
        OP_CHECK_IF(data == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "axes",
                                                             "axes ListInt attr data ptr is null"),
                    return ge::GRAPH_FAILED);
        std::set<int64_t> seen;
        for (int64_t i = 0; i < axesNum; ++i) {
            if (!PushOneAxis(context->GetNodeName(), data[i], i, xRank, seen, ctx.reduceAxes)) {
                return ge::GRAPH_FAILED; // 失败细节已在 PushOneAxis 内打 ERROR
            }
        }
        std::sort(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
    }

    // ── attr keepdim（idx 2，默认 false）/ epsilon（idx 3，默认 1e-12，reduce 段不参与计算）──
    const bool* attrKeepDim = attrs->GetAttrPointer<bool>(ATTR_KEEP_DIM_IDX);
    ctx.keepDim = (attrKeepDim == nullptr) ? false : (*attrKeepDim);
    // epsilon 按 OpDef Float 存储（AppendAttr(float) 落 4 字节槽），读 float 防 8 字节越界读；
    // 值不参与切分（spec.yaml attributes.epsilon 语义），仅日志透传。
    const float* attrEpsilon = attrs->GetAttrPointer<float>(ATTR_EPSILON_IDX);
    const double epsilon = (attrEpsilon == nullptr) ? 1.0e-12 : static_cast<double>(*attrEpsilon);

    OP_LOGI(context, "attrs: p=%ld axesNum=%ld keepDim=%d epsilon=%.3e", static_cast<long>(ctx.pOrder),
            static_cast<long>(axesNum), static_cast<int>(ctx.keepDim), epsilon);
    return ge::GRAPH_SUCCESS;
}

// 初始化轴列表：axisShape 拷贝 xShape，按 reduceAxes 给各轴打 R（归约轴）标记，其余为 A。
// 在 ClassifyEmptyTensor 之前执行，使空分类能在原始轴语义上判零维。
// ---------------------------------------------------------------------------
// 输出 desc 校验（批 b-4，对齐 changwei tiling.cpp GetAndCheckDtypes /
// CheckOutputShape 模式，codex 审查缺陷 #6）：y.dtype = x.dtype、y format ND、
// y origin/storage shape 与「axes + keepdim 推导的期望归约 shape」一致
// （rank-0 标量按 GE 物化约定接受 [] 或 [1]）。防御非推导链（GEIR 直调 /
// kernel 直调）传入错误输出描述直达 kernel 的越界写风险。
// ---------------------------------------------------------------------------
static ge::graphStatus CheckOutputDesc(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    auto yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_IF(yDesc->GetDataType() != ctx.xDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                    "output dtype must be identical to input dtype (y.dtype = x.dtype, "
                    "no cross-dtype promotion)"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yDesc->GetFormat().GetStorageFormat() != ge::FORMAT_ND,
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    context->GetNodeName(), "y",
                    ge::TypeUtils::FormatToSerialString(yDesc->GetFormat().GetStorageFormat()).c_str(),
                    "the storage format of y must be ND"),
                return ge::GRAPH_FAILED);

    // 期望 shape：非归约维保留原值；归约维 keepdim=true 置 1、false 删除
    std::vector<int64_t> expected;
    {
        const std::set<int64_t> reduced(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
        for (size_t i = 0; i < ctx.xShape.size(); ++i) {
            if (reduced.count(static_cast<int64_t>(i)) > 0) {
                if (ctx.keepDim) {
                    expected.push_back(1);
                }
            } else {
                expected.push_back(ctx.xShape[i]);
            }
        }
    }
    // 标量物化约定（changwei 同款）：全归约 keepdim=false 期望空时接受 [] 或 [1]
    const auto shapeMatches = [&expected](const gert::Shape& s) {
        if (expected.empty()) {
            return s.GetDimNum() == 0 || (s.GetDimNum() == 1 && s.GetDim(0) == 1);
        }
        if (s.GetDimNum() != expected.size()) {
            return false;
        }
        for (size_t i = 0; i < expected.size(); ++i) {
            if (s.GetDim(i) != expected[i]) {
                return false;
            }
        }
        return true;
    };

    auto yShapePtr = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    OP_CHECK_IF(!shapeMatches(yShapePtr->GetOriginShape()),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "y", "",
                    "the logical shape of y must match x, axes and keepdim; a scalar may use [] or [1]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !shapeMatches(yShapePtr->GetStorageShape()),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "y", "",
                                              "the storage shape of y must be concrete and match x, axes and keepdim; "
                                              "a scalar may use [] or [1]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static void BuildInitialAxisList(LpNormReduceCtx& ctx)
{
    const size_t rank = ctx.xShape.size();
    ctx.axisShape = ctx.xShape;
    ctx.isReduceAxis.assign(rank, false);
    for (int64_t reduceAxis : ctx.reduceAxes) {
        ctx.isReduceAxis[static_cast<size_t>(reduceAxis)] = true;
    }
}

// ---------------------------------------------------------------------------
// canndev 约束 2：p=±inf 哨兵且存在 size=0 的 R 轴（含空 axes 全量归约遇任一 0 维）→
// 拒绝：空集上 max/min 无定义。EMPTY_A（仅 A 轴为 0、R 轴全 >0）不触发——输出为空
// 张量、无空集 max/min。
// ---------------------------------------------------------------------------
static ge::graphStatus RejectInfSentinelOnEmptyReduceAxis(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    const bool isInfSentinel = (ctx.pOrder == INF_POS_SENTINEL || ctx.pOrder == INF_NEG_SENTINEL);
    if (!isInfSentinel) {
        return ge::GRAPH_SUCCESS;
    }
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        if (ctx.isReduceAxis[i] && ctx.xShape[i] == 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "p", std::to_string(ctx.pOrder),
                "p=+-inf sentinel cannot reduce over the whole (empty) tensor or an empty dim "
                "(canndev constraint, spec.yaml shape_constraints)");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 空 tensor 三分类（在原始 shape 上扫描零维，先于规整执行以免合轴/预算退化）：
// A 轴含 0 → EMPTY_A（输出 0 元素）；仅 R 轴含 0 → EMPTY_R（空归约：p=0/1/≥2 输出 0、
// p=±inf 已被上函数拒绝）；否则 NORMAL 走主 tiling 路径。
static void ClassifyEmptyTensor(LpNormReduceCtx& ctx)
{
    bool hasZeroA = false;
    bool hasZeroR = false;
    int64_t aTotal = 1;
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        const bool isR = ctx.isReduceAxis[i];
        const int64_t sz = ctx.xShape[i];
        if (!isR) {
            if (sz == 0) {
                hasZeroA = true;
            } else {
                aTotal *= sz;
            }
        } else {
            if (sz == 0) {
                hasZeroR = true;
            }
        }
    }
    if (hasZeroA) {
        ctx.emptyKind = LpNormReduceEmptyKind::EMPTY_A;
        ctx.aTotalEmpty = 0;
    } else if (hasZeroR) {
        ctx.emptyKind = LpNormReduceEmptyKind::EMPTY_R;
        ctx.aTotalEmpty = aTotal;
    } else {
        ctx.emptyKind = LpNormReduceEmptyKind::NORMAL;
        ctx.aTotalEmpty = 0;
    }
}

// ---------------------------------------------------------------------------
// 四步合轴（PreprocessPattern 总装，四步顺序不可换）
// ---------------------------------------------------------------------------

// A/R 规整第 1 步（共 4 步，顺序不可换）：删除所有 size-1 轴（对归约数值无贡献；
// A=1 不影响输出、R=1 退化为 |x|）；全删空则保留 [A1] 占位。
static void DropSizeOneAxes(LpNormReduceCtx& ctx)
{
    std::vector<int64_t> newShape;
    std::vector<bool> newIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (ctx.axisShape[i] != 1) {
            newShape.push_back(ctx.axisShape[i]);
            newIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    if (newShape.empty()) {
        newShape.push_back(1);
        newIsR.push_back(false);
    }
    ctx.axisShape = std::move(newShape);
    ctx.isReduceAxis = std::move(newIsR);
}

// A/R 规整第 2 步：相邻同类型轴（同 A 或同 R）乘积合并——A 合并 = 输出维乘积、
// R 合并 = 归约长度乘积（ND 连续布局下数值等价），压缩为最短交替模式。
static void FuseAxis(LpNormReduceCtx& ctx)
{
    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (!fusedShape.empty() && fusedIsR.back() == ctx.isReduceAxis[i]) {
            fusedShape.back() *= ctx.axisShape[i];
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduceAxis = std::move(fusedIsR);
}

// A/R 规整第 3 步：首轴为 R 时在最前补一根 size-1 的 A 轴，保证模式以 A 起头
// （偶位 A / 奇位 R 下标不变量的前提）。合成 A=1 的 stride = ∏所有轴 size，不是 1。
static void PadLeadingOneA(LpNormReduceCtx& ctx)
{
    if (!ctx.axisShape.empty() && ctx.isReduceAxis.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
    }
}

// A/R 规整第 4 步：纯 A 退化（无 R 轴）时补 R——全 1 退化 [A1] 末尾补 R=1（成 tail-R）；
// 其余前置 [A1, R1]（成 tail-A，保持最内轴为 A 的内存连续性——禁止末尾补 R=1，
// 否则最内轴变成长度 1 的 R，每 A 元素独占 1 元素 burst，搬运效率崩塌）。
static void PadRIfPureA(LpNormReduceCtx& ctx)
{
    bool hasR = false;
    for (bool isReduce : ctx.isReduceAxis) {
        if (isReduce) {
            hasR = true;
            break;
        }
    }
    if (!hasR) {
        if (ctx.axisShape.size() == 1 && ctx.axisShape[0] == 1) {
            ctx.axisShape.push_back(1); // A=1 → 末尾补 [R=1]，pattern 由 A 变 AR（tail-R）
            ctx.isReduceAxis.push_back(true);
        } else {
            ctx.axisShape.insert(ctx.axisShape.begin(), {1, 1});              // A>1 → 前置 [A=1, R=1]，
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), {false, true}); // pattern 由 A 变 ARA（tail-A）
        }
    }
}

// A/R 规整总装（仅 NORMAL 路径）：依次执行去 1 → 合轴 → 补 leading A → 补 R 增广，
// 校验 axisNum ∈ [MIN_AXIS_NUM, MAX_AXIS_NUM] 与「偶位 A / 奇位 R」不变量（防御性
// 自检），按轴数奇偶定 isTailR：偶数轴 → tail-R（最内轴为 R），奇数轴 → tail-A。
static ge::graphStatus PreprocessPattern(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    DropSizeOneAxes(ctx);
    FuseAxis(ctx);
    PadLeadingOneA(ctx);
    PadRIfPureA(ctx);

    ctx.axisNum = static_cast<int32_t>(ctx.axisShape.size());
    OP_CHECK_IF(ctx.axisNum < MIN_AXIS_NUM || ctx.axisNum > MAX_AXIS_NUM,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "axisNum",
                    "after A/R pattern regularization, axisNum=" + std::to_string(ctx.axisNum) + " out of [" +
                        std::to_string(MIN_AXIS_NUM) + ", " + std::to_string(MAX_AXIS_NUM) + "]"),
                return ge::GRAPH_FAILED);

    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        const bool wantR = (i % AXIS_INTERVAL == 1);
        OP_CHECK_IF(ctx.isReduceAxis[static_cast<size_t>(i)] != wantR,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "isReduceAxis",
                                                             "axis[" + std::to_string(i) +
                                                                 "] type mismatch after A/R regularization "
                                                                 "(expected alternating [A,R] pattern)"),
                    return ge::GRAPH_FAILED);
    }
    ctx.isTailR = (ctx.axisNum % AXIS_INTERVAL == 0);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 辅助函数（A/R 规整后偶位 A、奇位 R，AXIS_INTERVAL = 2）
// ---------------------------------------------------------------------------

// 计算规整后各轴在输入 GM 上的步长：stride[i] = ∏_{j>i} axisShape[j]
// （ND 连续布局，最内轴 stride=1），供 kernel 侧 Unravel 解码 GM 偏移。
static void ComputeAxisStrides(const LpNormReduceCtx& ctx, int64_t outStride[])
{
    int64_t strideAcc = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        outStride[i] = strideAcc;
        strideAcc *= ctx.axisShape[static_cast<size_t>(i)];
    }
}

// 返回最后一个 A 轴（最大偶下标）的位置；规整后恒存在，兜底返回 0（防御）。
static int32_t LastAAxisIdx(const LpNormReduceCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

// 返回最后一个 R 轴（最大奇下标）的位置；规整后恒存在，兜底返回 1（防御）。
static int32_t LastRAxisIdx(const LpNormReduceCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// 计算全部 A 轴长度乘积 = 输出元素总数 aTotal（Group workspace 列数 / 扩 A 上限）。
static int64_t TotalAProd(const LpNormReduceCtx& ctx)
{
    int64_t total = 1;
    for (int32_t i = 0; i < ctx.axisNum; i += AXIS_INTERVAL) {
        total *= ctx.axisShape[static_cast<size_t>(i)];
    }
    return total;
}

// 计算 aSplitIdx 内侧 A 轴的对齐乘积 innerAProdAlign：仅 tail-A 的最内 A 轴（搬运
// burst 轴）按 32B 元素数（bsElem）向上对齐，其余轴原值累乘；aSplit 变化后需重算。
static void ComputeInnerAProdAlign(LpNormReduceCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProdAlign = 1;
    for (int32_t k = ctx.aSplitIdx + AXIS_INTERVAL; k < ctx.axisNum; k += AXIS_INTERVAL) {
        if (k == ctx.axisNum - 1 && !ctx.isTailR) {
            ctx.innerAProdAlign *= Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(k)], bsElem);
        } else {
            ctx.innerAProdAlign *= ctx.axisShape[static_cast<size_t>(k)];
        }
    }
}

// ---------------------------------------------------------------------------
// UB 切分三步算法（范式 reduction-binary-base-tiling.md §0 / §6）
// ⛔ dtype 参数适用边界（强制）：bsElem = blockSize / sizeof(D_T)（元素数取整——搬运/
//    对齐针对输入数据，burst 按 D_T 计）；maxDtypeSize = max(sizeof(D_T), sizeof(float))
//    （字节容量——buffer 跨 dtype 复用按最宽分配）。两参数禁止混用。
// ---------------------------------------------------------------------------

// Step 1：定 aSplitIdx 与 aUbFactor（ComputeAUbFactor）——以 1 条 cache line 为初始
// 预算确定 A 切分，使 A 单元（aUnit = aUbFactor × innerAProdAlign）驻留单条 cache line
// （Reduce 主循环热数据局部性）。联合爬坡从最内轴向外累乘（尾轴无论 A/R 一律 CeilAlign
// 到 block 对齐，终止比较用 `>` 而非 `>=`：tail-A 下当 CeilAlign(LastA) == cachelineTmp
// 时仍吸入 LastA，保证 aUnit 一定对齐）：停在 A 轴 → 按剩余预算切该轴；停在 R 轴 →
// 退到其左侧 A 轴、因子 1；全部装得下 → 不切（aSplitIdx=0）。
static void ComputeAUbFactor(LpNormReduceCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize; // ⛔ 元素数取整用 sizeof(D_T)
    const int64_t maxInnerAInitElem = ctx.cacheLineSize / ctx.dtypeSize;
    const int64_t cachelineTmp = (maxInnerAInitElem / bsElem) * bsElem; // block 对齐

    int64_t product = 1;
    int32_t idx = ctx.axisNum - 1;
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1) ? Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)],
                                                                           bsElem) : // 尾轴 burst 方向 block 对齐
                                                      ctx.axisShape[static_cast<size_t>(idx)];
        if (axisSize > cachelineTmp / product) { // 除法式比较：product*axisSize 可溢出 int64（合法大维度）
            break;
        }
        product *= axisSize;
        idx--;
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0; // 所有轴全驻，无轴可切
        ctx.aUbFactor = ctx.axisShape[0];
    } else if (idx % AXIS_INTERVAL == 0) { // 停在 A 轴：直接切
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(cachelineTmp / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    } else { // 停在 R 轴：R 不能作 A 切分点，左移一根 A，每 chunk 取 1
        ctx.aSplitIdx = idx - 1;
        ctx.aUbFactor = 1;
    }

    ComputeInnerAProdAlign(ctx);
}

// 反解单次 R 迭代可用元素上限 rIMax（UB 预算不等式的逆）：
// (ubSize − 16KB cacheBuf − outBuf) / (3 个 pre buffer × aUnit × maxDtypeSize)；
// aUnit 非法或预算不足时返回 -1（由调用方报 TILING_FAIL）。
static int64_t ComputeRiMax(const LpNormReduceCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    const int64_t postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
    const int64_t aOnlyBytes = P_POST * postBufSize;
    const int64_t bytesPerRElem = (P_PRE + P_PRE_EXT) * aUnit * ctx.maxDtypeSize; // ⛔ 字节容量用 maxDtypeSize
    if (aUnit <= 0 || bytesPerRElem <= 0) {
        return -1;
    }
    const int64_t numer = ubAvailable - aOnlyBytes;
    if (numer <= 0) {
        return -1;
    }
    return numer / bytesPerRElem;
}

// Step 2：定 rSplitIdx 与 rUbFactor（ComputeRUbFactor）——先算 postBufSize，UB 预算
// 反解 r_i_max，R 端 Climb 从最内 R 轴向外吸入 innerRProdAlign，再定 rUbFactor /
// rUbFactorAlign（tail-R 且切最内 R 轴时做 32B burst 对齐）：
// ⚠ 无条件 FloorAlign 会误判 TILING_FAIL：整轴全载时先 CeilAlign 校验预算，超预算才
//    回退 FloorAlign；R < blockSize 全载时跳过 FloorAlign（FloorAlign(R, bsElem)=0 是误判）。
static ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    const int64_t rIMax = ComputeRiMax(ctx);
    OP_CHECK_IF(rIMax < 1,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "rIMax",
                    "R_i_max=" + std::to_string(rIMax) + " < 1 (aUbFactor=" + std::to_string(ctx.aUbFactor) +
                        ", innerAProdAlign=" + std::to_string(ctx.innerAProdAlign) + ")"),
                return ge::GRAPH_FAILED);

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerRProdAlign = 1;
    ctx.rSplitIdx = lastR;
    while (ctx.rSplitIdx > 1) {
        int64_t axisSize = (ctx.rSplitIdx == lastR && ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)],
                                                    bsElem) : // 尾轴 R 先对齐再比较
                               ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
        if (axisSize > rIMax / ctx.innerRProdAlign) { // 除法式比较：乘积可溢出 int64（合法大维度）
            break;                                    // 吸入后会超预算则停
        }
        ctx.innerRProdAlign *= axisSize;
        ctx.rSplitIdx -= AXIS_INTERVAL; // 向外移动一根 R 轴（跳过中间的 A 轴）
    }

    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);

    const bool isBurstTailR = (ctx.isTailR && ctx.rSplitIdx == lastR);
    if (isBurstTailR) {
        if (ctx.rUbFactor < rAxisSize) {
            // 切多 chunk：burst 尾轴方向需 block 对齐
            ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
            OP_CHECK_IF(
                ctx.rUbFactor == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rUbFactor",
                                                         "rUbFactor floor-aligned to 0 (tail-R burst alignment)"),
                return ge::GRAPH_FAILED);
            ctx.rUbFactorAlign = ctx.rUbFactor;
        } else {
            // 整轴（含 R < blockSize 全载场景）：CeilAlign 后校验不超预算——
            // 不直接 FloorAlign（R < bsElem 时 FloorAlign 到 0 会误判 TILING_FAIL）
            ctx.rUbFactorAlign = Ops::Base::CeilAlign(ctx.rUbFactor, bsElem);
            if (ctx.rUbFactorAlign * ctx.innerRProdAlign > rIMax) {
                // 放大后超预算，退回切多 chunk
                ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
                OP_CHECK_IF(ctx.rUbFactor == 0,
                            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                                context->GetNodeName(), "rUbFactor",
                                "rUbFactor floor-aligned to 0 (retry after CeilAlign overflow)"),
                            return ge::GRAPH_FAILED);
                ctx.rUbFactorAlign = ctx.rUbFactor;
            }
        }
    } else {
        // 非 burst 尾轴方向，不需要对齐
        ctx.rUbFactorAlign = ctx.rUbFactor;
    }
    return ge::GRAPH_SUCCESS;
}

// 判定整个 R 空间是否单次迭代全载：rUbFactor 占满 rSplit 轴长度，且 rSplitIdx 外侧
// 无 R 轴（即 rLoopCntTotal 将为 1）——全载时才值得把剩余 UB 让给 A。
static bool RIsFullyLoaded(const LpNormReduceCtx& ctx)
{
    if (ctx.rUbFactor != ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)]) {
        return false;
    }
    for (int32_t i = ctx.rSplitIdx - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return false;
        }
    }
    return true;
}

// R 全载时把 UB 预算反解为 A 上限：每 A lane 耗 3×rPadded 个 pre 元素 + 1 个 out 元素；
// 预算非法返回 -1。
static int64_t SolveAUnitMax(const LpNormReduceCtx& ctx)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    const int64_t coeff = (P_PRE + P_PRE_EXT) * rPaddedElems * ctx.maxDtypeSize + P_POST * ctx.maxDtypeSize;
    if (coeff <= 0) {
        return -1;
    }
    return ubAvailable / coeff;
}

// Step 3：所有 R 全驻后扩 A（ExpandAIfRFullyLoaded）——R 全载时用剩余 UB 扩张 A 单元，
// 减少 A 迭代次数。aUnitMax 受三重约束：UB 反解值 / totalA / cacheBuf 行宽上限
// （16KB÷4 = 4096，cacheBuf 硬约束——不钳制会 cacheBuf 溢出）；tail-A 时按 32B 元素数
// 向下对齐（最内 A 轴是 burst 轴）；从 lastA 向内重新累积定 aSplitIdx / aUbFactor 并
// 重算 innerAProdAlign；无利可图（≤ 当前值）则保持不变。
static ge::graphStatus ExpandAIfRFullyLoaded(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    if (!RIsFullyLoaded(ctx)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    const int64_t cacheLaneLimit = CACHE_BUF_BYTES / FP32_BYTES; // 16KB / 4B = 4096
    int64_t totalA = TotalAProd(ctx);

    int64_t aUnitMax = std::min(SolveAUnitMax(ctx), totalA);
    aUnitMax = std::min(aUnitMax, cacheLaneLimit); // ★ cacheBuf 硬约束：R 全驻即 cacheCount=1
    if (!ctx.isTailR) {
        // ⚠ 必须在 min(∏A) 钳制之后 FloorAlign：∏A 可能非对齐，aUnitMax 就可能非对齐
        aUnitMax = Ops::Base::FloorAlign(aUnitMax, bsElem); // tail-A 下 aUnit 是 burst 尾轴方向，需 block 对齐
    }

    const int64_t curAUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    if (aUnitMax <= curAUnit) {
        return ge::GRAPH_SUCCESS; // R 全载后 UB 剩余空间不足，扩 A 无效，保持 Step 1 结果
    }

    int64_t product = 1;
    int32_t idx = LastAAxisIdx(ctx);
    while (idx >= 0) {
        int64_t axisSize = (idx == ctx.axisNum - 1 && !ctx.isTailR) ?
                               Ops::Base::CeilAlign(ctx.axisShape[static_cast<size_t>(idx)],
                                                    bsElem) : // tail-A 尾轴 burst 方向 block 对齐
                               ctx.axisShape[static_cast<size_t>(idx)];
        if (axisSize > aUnitMax / product) { // 除法式比较：product*axisSize 可溢出 int64（合法大维度）
            break;
        }
        product *= axisSize;
        idx -= AXIS_INTERVAL; // 向外移动一根 A 轴（跳过中间的 R 轴）
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else {
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(aUnitMax / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    }

    ComputeInnerAProdAlign(ctx);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 多核切分（范式 §2.1）：外层 A loop 全部 fuse 成一根线性计数，多核按此瓜分。
// 大小核均衡：前 aLoopCntTotal%coreNum 个核多干 1 轮；迭代-核映射固定，保证
// bitwise_reproducible。核数计算全部动态（coreNum 来自 GetPlatformInfo，禁止硬编码）。
// ---------------------------------------------------------------------------

// A 方向多核切分：aLoopCntTotal = aSplitIdx 外侧 A 轴乘积 × 切分轴 chunk 数（外层 A
// 轴视作一个混合进制循环，即 "fused"）。
static void ComputeFusedALoopSplit(LpNormReduceCtx& ctx)
{
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitIdx; i += AXIS_INTERVAL) {
        outerAProd *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t aSplitAxisSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitIdx)];
    ctx.aSplitChunkCnt = Ops::Base::CeilDiv(aSplitAxisSize, ctx.aUbFactor);
    ctx.aLoopCntTotal = outerAProd * ctx.aSplitChunkCnt;

    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum);
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, 1);
}

// R 方向总迭代数：rLoopCntTotal = rSplitIdx 外侧 R 轴乘积 × CeilDiv(切分轴长, rUbFactor)
// （注意用 valid 的 rUbFactor，不是 padded 的 rUbFactorAlign）；供 kernel Init 推导二分
// 树参数（bisectionPos/Tail/cacheCount）与 Group 触发判定。
static void ComputeRLoopCnt(LpNormReduceCtx& ctx)
{
    int64_t outerR = 1;
    for (int32_t i = 1; i < ctx.rSplitIdx; i += AXIS_INTERVAL) {
        outerR *= ctx.axisShape[static_cast<size_t>(i)];
    }
    const int64_t rChunks = Ops::Base::CeilDiv(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], ctx.rUbFactor);
    ctx.rLoopCntTotal = outerR * rChunks;
}

// ---------------------------------------------------------------------------
// workspace 申报（范式 [7.4]）：Group 时用户 workspace = rGroupCnt × aTotal ×
// sizeof(float)（[rGroupCnt 行, aTotal 列] fp32 行优先 dense 部分和矩阵），叠加系统
// workspace（GetLibApiWorkSpaceSize）；normal（base）/ empty 仅需系统部分。
// 必须显式设置，即使不使用（ascendc 要求）。
// ---------------------------------------------------------------------------
static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    size_t* ws = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, ws);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t usrSize = 0;
    if (ctx.isGroup) {
        int64_t aTotal = TotalAProd(ctx);
        // workspace 连乘上界守卫（codex 审查缺陷 #2 末点）：rGroupCnt × aTotal ×
        // sizeof(float) 超 size_t 表示域 → 干净拒绝，杜绝 wrap 后的小尺寸申报。
        // 上界再扣 sysWorkspaceSize，保证 ws[0] = usr + sys 不回绕。
        const size_t usrLimit = std::numeric_limits<size_t>::max() - sysWorkspaceSize;
        OP_CHECK_IF(ctx.rGroupCnt <= 0 ||
                        static_cast<size_t>(aTotal) > usrLimit / static_cast<size_t>(ctx.rGroupCnt) / sizeof(float),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(aTotal),
                        "group workspace size (rGroupCnt * aTotal * sizeof(float)) exceeds the size_t range"),
                    return ge::GRAPH_FAILED);
        usrSize = static_cast<size_t>(ctx.rGroupCnt) * static_cast<size_t>(aTotal) * sizeof(float);
    }

    ws[0] = usrSize + sysWorkspaceSize;
    OP_LOGI(context, "Set ws size:%lu, usrSize:%lu", static_cast<unsigned long>(ws[0]),
            static_cast<unsigned long>(usrSize));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 空 tensor 快速路径（empty 专用：UB 切分 + 多核切分 + workspace，范式 [5.2]~[5.4]）
// ---------------------------------------------------------------------------

// EMPTY_R 输出填充切分（无需读输入，仅向 y 写 aTotal 个 0）：
// aUbFactor = clamp(max(每核 ≥4KB 防碎片, CeilDiv(aTotal, coreNum)), 单 buf ≤64KB 上限,
// aTotal)；输出区间按大小核协议均分；postBufSize = CeilAlign(max(aUbFactor×maxDtypeSize,
// 32B), 32B)；preBufSize = 0（不申请任何输入侧 buffer）。
static void ComputeEmptyRTiling(LpNormReduceCtx& ctx)
{
    constexpr int64_t maxSingleUbBytes = 64 * 1024;
    const int64_t maxUbFactor = std::min(maxSingleUbBytes / ctx.maxDtypeSize, ctx.ubSize / P_POST / ctx.maxDtypeSize);

    constexpr int64_t minBytesPerCore = 4096;
    const int64_t minAPerCore = Ops::Base::CeilDiv(minBytesPerCore, ctx.maxDtypeSize);
    const int64_t aTotal = ctx.aTotalEmpty;

    int64_t aUbFactor = std::max(minAPerCore, Ops::Base::CeilDiv(aTotal, ctx.coreNum));
    aUbFactor = std::min(aUbFactor, maxUbFactor);
    aUbFactor = std::min(aUbFactor, aTotal);
    aUbFactor = std::max<int64_t>(aUbFactor, 1);

    const int64_t aLoopCntTotal = Ops::Base::CeilDiv(aTotal, aUbFactor);
    const int64_t aSmallCoreLoopCnt = aLoopCntTotal / ctx.coreNum;
    const int32_t aBigCoreCnt = static_cast<int32_t>(aLoopCntTotal % ctx.coreNum);
    const int64_t aBigCoreLoopCnt = aSmallCoreLoopCnt + (aBigCoreCnt > 0 ? 1 : 0);
    const int32_t usedCoreNum = (aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : std::max(aBigCoreCnt, 1);

    ctx.aUbFactor = aUbFactor;
    ctx.aLoopCntTotal = aLoopCntTotal;
    ctx.aSplitChunkCnt = aLoopCntTotal;
    ctx.aSmallCoreLoopCnt = aSmallCoreLoopCnt;
    ctx.aBigCoreLoopCnt = aBigCoreLoopCnt;
    ctx.aBigCoreCnt = aBigCoreCnt;
    ctx.usedCoreNum = usedCoreNum;

    int64_t outRaw = aUbFactor * ctx.maxDtypeSize;
    outRaw = std::max(outRaw, ctx.blockSize);
    ctx.postBufSize = Ops::Base::CeilAlign(outRaw, ctx.blockSize);
    ctx.preBufSize = 0;
}

// 填充 LpNormReduceEmptyTilingData：memset 清零后，EMPTY_A 全零直接返回（kernel 全核
// 早退）；EMPTY_R 填输出切分 7 字段（usedCoreNum / aTotal / aUbFactor / 大小核三参 /
// postBufSize）。
static ge::graphStatus FillEmptyTilingData(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    LpNormReduceEmptyTilingData* td = context->GetTilingData<LpNormReduceEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(LpNormReduceEmptyTilingData), 0, sizeof(LpNormReduceEmptyTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    if (ctx.emptyKind == LpNormReduceEmptyKind::EMPTY_A) {
        OP_LOGI(context, "EMPTY_A: dtype=%d usedCoreNum=0", static_cast<int>(ctx.xDtype));
        return ge::GRAPH_SUCCESS;
    }

    td->aTotal = ctx.aTotalEmpty;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aUbFactor = ctx.aUbFactor;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->postBufSize = ctx.postBufSize;

    OP_LOGI(context, "EMPTY_R: dtype=%d aTotal=%ld aUbFactor=%ld usedCoreNum=%d postBuf=%ld",
            static_cast<int>(ctx.xDtype), ctx.aTotalEmpty, td->aUbFactor, td->usedCoreNum, td->postBufSize);
    return ge::GRAPH_SUCCESS;
}

// 空 tensor 快速路径总装（主流程短路出口，跳过整套 normal 切分）：
// EMPTY_R → ComputeEmptyRTiling；EMPTY_A → usedCoreNum=0（kernel 全核零操作）→
// FillEmptyTilingData → 固定 tilingKey(isGroup=0, isEmptyTensor=1) →
// SetBlockDim(max(usedCoreNum,1))（EMPTY_A 启 1 核即退，严禁 SetBlockDim(0)）→
// SetWorkspaceSize（仅系统 ws，empty 不走 Group）。
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context, LpNormReduceCtx& ctx)
{
    if (ctx.emptyKind == LpNormReduceEmptyKind::EMPTY_R) {
        ComputeEmptyRTiling(ctx);
    } else {
        ctx.usedCoreNum = 0;
    }
    OP_CHECK_IF(FillEmptyTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(0U), static_cast<uint64_t>(1U));
    context->SetTilingKey(tilingKey);
    OP_LOGI(context, "SetTilingKey: isGroup=0 isEmptyTensor=1 tilingKey=%lu", static_cast<unsigned long>(tilingKey));
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// Group 触发判定与 2D 分核（范式 [7.3]）
// ---------------------------------------------------------------------------

// Group 2D 分核触发判定（BranchRoute.md：aLoopCntTotal ≤ coreNum/GROUP_CORE_RATIO
// （A 方向并行度吃不满半数核）且 R 迭代数 > 1（R 方向有多轮迭代可切）——半数核闲置
// 且 R 有并行度时才值得 2D 分核；布尔数值条件严格二分、无交集。
static bool ShouldUseGroup(const LpNormReduceCtx& ctx)
{
    if (ctx.aLoopCntTotal > ctx.coreNum / GROUP_CORE_RATIO) {
        return false;
    }
    if (ctx.rLoopCntTotal <= 1) {
        return false;
    }
    return true;
}

// Group 2D 分核参数：totalOuter = aLoopCntTotal × rLoopCntTotal 按核均分得 numBlocks，
// 再把 numBlocks 对齐到 aLoopCntTotal 的整数倍（CeilAlign 超核数则 FloorAlign）——保证
// 每核 = 整数个完整 A chunk × 一段完整 R 区间（Phase1 核号分解 blockIdx = aChunkIdx×
// rGroupCnt + rChunkIdx 依赖此性质）；得 usedCoreNum 与 rGroupCnt（R 分组数 = Phase2
// workspace 行数）。aPerCore = 1 恒成立（numBlocks 是 aLoopCntTotal 的整数倍）。
static bool ComputeGroupSplit(LpNormReduceCtx& ctx)
{
    // totalOuter 连乘上界守卫（codex 审查缺陷 #2 派生点）：aLoopCntTotal 与
    // rLoopCntTotal 各自 ≤ numel，但二者乘积可溢出 int64；同时预留 coreNum 余量
    // 吸收 CeilDiv(totalOuter, coreNum) 的加法边界。溢出 → 返回 false 由调用方拒绝。
    constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();
    if (ctx.rLoopCntTotal <= 0 || ctx.aLoopCntTotal > (kInt64Max - ctx.coreNum) / ctx.rLoopCntTotal) {
        return false;
    }
    int64_t totalOuter = ctx.aLoopCntTotal * ctx.rLoopCntTotal;
    int64_t perCoreNum = Ops::Base::CeilDiv(totalOuter, ctx.coreNum);
    int64_t numBlocks = Ops::Base::CeilDiv(totalOuter, perCoreNum);

    if (Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal) <= ctx.coreNum) {
        numBlocks = Ops::Base::CeilAlign(numBlocks, ctx.aLoopCntTotal);
    } else {
        numBlocks = Ops::Base::FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }

    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal;
    ctx.isGroup = true;
    return true;
}

// ---------------------------------------------------------------------------
// Buffer 大小计算 + 填 TilingData（范式 §5.1 / [8.1]）
// ---------------------------------------------------------------------------

// Each binary-tree level stores one block-aligned A row in the fixed cache.
// Group phase 1 uses the largest local R range, not the global R loop count.
// Check metadata before launching kernels for shapes too large to allocate here.
static ge::graphStatus CheckCacheCapacity(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    const int64_t localRLoops = ctx.isGroup ? Ops::Base::CeilDiv(ctx.rLoopCntTotal, ctx.rGroupCnt) : ctx.rLoopCntTotal;
    int64_t cacheLevels = 1;
    for (int64_t remaining = localRLoops - 1; remaining > 1; remaining >>= 1) {
        ++cacheLevels;
    }
    const int64_t levelStride = Ops::Base::CeilAlign(ctx.aUbFactor * ctx.innerAProdAlign, ctx.blockSize / FP32_BYTES);
    OP_CHECK_IF(levelStride <= 0 || levelStride > CACHE_BUF_BYTES / FP32_BYTES / cacheLevels,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(localRLoops),
                                                         "reduction tree exceeds the fixed UB cache capacity"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// 计算 UB buffer 尺寸：preBufSize = aUnit × rPaddedElems × maxDtypeSize
// （preIn / preRes / preResTail 三个 buffer 同尺寸，统一按 maxDtypeSize 预算，b16 也
// 按 4B 计）；postBufSize = CeilAlign(aUnit × maxDtypeSize, blockSize)（1D，block 对齐）。
static void ComputeUbSizes(LpNormReduceCtx& ctx)
{
    const int64_t aUnit = ctx.aUbFactor * ctx.innerAProdAlign;
    const int64_t rPaddedElems = ctx.rUbFactorAlign * ctx.innerRProdAlign;
    ctx.preBufSize = aUnit * rPaddedElems * ctx.maxDtypeSize;
    ctx.postBufSize = Ops::Base::CeilAlign(aUnit * ctx.maxDtypeSize, ctx.blockSize);
}

// 把 ctx 全量写入 LpNormReduceTilingData：memset_s 清零后逐字段填充（未用轴槽
// axisShape=1 / axisStride=0，定长数组约定），cacheBufUbSize 直填固定 16KB；
// axisStride 现场由 ComputeAxisStrides 计算；本算子唯一追加字段 pOrder 直填；
// 随后打全量 INFO 日志（pattern / 切分 / 多核参数，含数组字段拼接打印）。
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const LpNormReduceCtx& ctx)
{
    LpNormReduceTilingData* td = context->GetTilingData<LpNormReduceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(LpNormReduceTilingData), 0, sizeof(LpNormReduceTilingData)) != EOK,
                OP_LOGE(context, "Memset tilingdata error"), return ge::GRAPH_FAILED);

    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    ComputeAxisStrides(ctx, axisStride);

    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = (i < ctx.axisNum) ? ctx.axisShape[static_cast<size_t>(i)] : 1;
        td->axisStride[i] = (i < ctx.axisNum) ? axisStride[i] : 0;
    }
    td->aLoopCntTotal = ctx.aLoopCntTotal;
    td->aSplitChunkCnt = ctx.aSplitChunkCnt;
    td->aBigCoreLoopCnt = ctx.aBigCoreLoopCnt;
    td->aSmallCoreLoopCnt = ctx.aSmallCoreLoopCnt;
    td->aBigCoreCnt = ctx.aBigCoreCnt;
    td->usedCoreNum = ctx.usedCoreNum;
    td->aSplitIdx = ctx.aSplitIdx;
    td->rSplitIdx = ctx.rSplitIdx;
    td->aUbFactor = ctx.aUbFactor;
    td->rUbFactor = ctx.rUbFactor;
    td->rUbFactorAlign = ctx.rUbFactorAlign;
    td->innerAProdAlign = ctx.innerAProdAlign;
    td->innerRProdAlign = ctx.innerRProdAlign;
    td->rLoopCntTotal = ctx.rLoopCntTotal;
    td->preBufSize = ctx.preBufSize;
    td->postBufSize = ctx.postBufSize;
    td->cacheBufUbSize = CACHE_BUF_BYTES;
    td->rGroupCnt = ctx.rGroupCnt;
    td->pOrder = ctx.pOrder; // 本算子唯一自定义字段：范数阶数 p 原值（含 ±inf 整型哨兵）

    OP_LOGI(context, "tiling: dtype=%d axisNum=%d isTailR=%d usedCoreNum=%d isGroup=%d", static_cast<int>(ctx.xDtype),
            ctx.axisNum, static_cast<int>(ctx.isTailR), ctx.usedCoreNum, static_cast<int>(ctx.isGroup));
    OP_LOGI(context, "  aSplitIdx=%d aUbFactor=%ld rSplitIdx=%d rUbFactor=%ld rUbFactorAlign=%ld", td->aSplitIdx,
            td->aUbFactor, td->rSplitIdx, td->rUbFactor, td->rUbFactorAlign);
    OP_LOGI(context, "  innerAProdAlign=%ld innerRProdAlign=%ld rLoopCntTotal=%ld", td->innerAProdAlign,
            td->innerRProdAlign, td->rLoopCntTotal);
    OP_LOGI(context, "  UB: preBuf=%ld postBuf=%ld cache=%ld", td->preBufSize, td->postBufSize, td->cacheBufUbSize);
    OP_LOGI(context, "  aLoopCntTotal=%ld aSplitChunkCnt=%ld aBigCoreLoopCnt=%ld aSmallCoreLoopCnt=%ld aBigCoreCnt=%d",
            td->aLoopCntTotal, td->aSplitChunkCnt, td->aBigCoreLoopCnt, td->aSmallCoreLoopCnt, td->aBigCoreCnt);
    OP_LOGI(context, "  rGroupCnt=%ld pOrder=%ld", td->rGroupCnt, td->pOrder);
    std::string shapeStr = "[";
    for (int32_t i = 0; i < td->axisNum; ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(td->axisShape[i]);
    }
    shapeStr += "]";
    OP_LOGI(context, "  axisShape=%s", shapeStr.c_str());
    std::string strideStr = "[";
    for (int32_t i = 0; i < td->axisNum; ++i) {
        if (i > 0) {
            strideStr += ", ";
        }
        strideStr += std::to_string(td->axisStride[i]);
    }
    strideStr += "]";
    OP_LOGI(context, "  axisStride=%s", strideStr.c_str());
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingFuncLpNormReduce(context) — tiling entry point called by CANN runtime
//
// TilingFunc 主入口（IMPL_OP_OPTILING 注册），编排全链（HostTiling.md「Tiling 整体
// 结构」节）：
// 平台/输入/attr 解析 → ±inf 空 R 拒绝 + 空分类（非 NORMAL 走 HandleEmptyTensor
//    直接返回）→ A/R 规整（合轴四步）→ A 切分 → R 切分 → R 全载扩 A（UB 三步
//    算法）→ A 多核均分 + R 迭代数 → Group 判定（触发则 ComputeGroupSplit +
//    SetScheduleMode(1) batch mode，kernel SyncAll 依赖）→ UB 尺寸 + 填 TilingData →
// SetTilingKey(isGroup, isEmptyTensor=0) + SetBlockDim + SetWorkspaceSize。
// 判定顺序与 BranchRoute.md「分支优先级与互斥」一致（入口校验 → ±inf 空 R 拒绝 →
// EMPTY_A → EMPTY_R → 合轴切分 → Group 判定）。
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncLpNormReduce(gert::TilingContext* context)
{
    OP_LOGD(context, "Begin LpNormReduceTilingFunc");
    LpNormReduceCtx ctx;

    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(GetShapeAndDtype(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // axes 属性版（非 ParseAxesTensor）
    OP_CHECK_IF(ParseAttrs(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputDesc(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    BuildInitialAxisList(ctx);
    OP_CHECK_IF(RejectInfSentinelOnEmptyReduceAxis(context, ctx) != ge::GRAPH_SUCCESS, ,
                return ge::GRAPH_FAILED); // canndev 约束 2：±inf 哨兵 + 空 R 轴 → 拒绝
    ClassifyEmptyTensor(ctx);

    if (ctx.emptyKind != LpNormReduceEmptyKind::NORMAL) {
        return HandleEmptyTensor(context, ctx); // EMPTY_A / EMPTY_R → tilingKey=2
    }

    OP_CHECK_IF(PreprocessPattern(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeAUbFactor(ctx);
    OP_LOGD(context->GetNodeName(), "ComputeAUbFactor: aSplitIdx=%d aUbFactor=%ld innerAProdAlign=%ld", ctx.aSplitIdx,
            ctx.aUbFactor, ctx.innerAProdAlign);
    OP_CHECK_IF(ComputeRUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExpandAIfRFullyLoaded(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    ComputeFusedALoopSplit(ctx);
    ComputeRLoopCnt(ctx);
    OP_LOGD(context->GetNodeName(), "MultiCore: aLoopCntTotal=%ld aSplitChunkCnt=%ld usedCoreNum=%d rLoopCntTotal=%ld",
            ctx.aLoopCntTotal, ctx.aSplitChunkCnt, ctx.usedCoreNum, ctx.rLoopCntTotal);
    if (ShouldUseGroup(ctx)) {
        OP_CHECK_IF(!ComputeGroupSplit(ctx),
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        context->GetNodeName(), "x", std::to_string(ctx.aLoopCntTotal),
                        "group total loop count (aLoopCntTotal * rLoopCntTotal) exceeds the int64 range"),
                    return ge::GRAPH_FAILED);
        OP_LOGD(context->GetNodeName(), "Group: rGroupCnt=%ld usedCoreNum=%d", ctx.rGroupCnt, ctx.usedCoreNum);
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF(CheckCacheCapacity(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    ComputeUbSizes(ctx);
    OP_LOGD(context->GetNodeName(), "UbSizes: preBufSize=%ld postBufSize=%ld", ctx.preBufSize, ctx.postBufSize);

    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // TilingKey 按 ASCENDC_TPL 双 bool 位编码（TilingKey.md）：isGroup 占 bit0、
    // isEmptyTensor 占 bit1 → base=0、group=1、empty=2；(1,1) 互斥为设计性空位
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(ctx.isGroup ? 1U : 0U),
                                                  static_cast<uint64_t>(0U));
    context->SetTilingKey(tilingKey);
    OP_LOGI(context, "SetTilingKey: isGroup=%d isEmptyTensor=0 tilingKey=%lu", static_cast<int>(ctx.isGroup),
            static_cast<unsigned long>(tilingKey));
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));

    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingPrepareForLpNormReduce(context) — compile-time preparation
//
// 编译期准备函数：把平台信息（AIV 核数 / UB 大小）写入 LpNormReduceCompileInfo，
// 供 tiling 运行期使用（本算子 TilingFunc 运行期另经 GetPlatformInfo 现查，两条
// 通路数据同源）。
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForLpNormReduce(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<LpNormReduceCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(LpNormReduce) — register tiling functions with CANN
//
// .Tiling(TilingFuncLpNormReduce)：
//   注册运行期 tiling 函数（见上）。
// .TilingParse<LpNormReduceCompileInfo>(TilingPrepareForLpNormReduce)：
//   注册编译期准备函数；LpNormReduceCompileInfo 承载平台信息（coreNum/ubSize）。
// 注意：无 .TilingInputsDataDependency —— axes 为 ListInt 属性而非 tensor 输入，
// 区别于范式参考实现（euclidean_norm：IMPL_OP_OPTILING(...).TilingInputsDataDependency({1})）。
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(LpNormReduce)
    .Tiling(TilingFuncLpNormReduce)
    .TilingParse<LpNormReduceCompileInfo>(TilingPrepareForLpNormReduce);

} // namespace optiling
