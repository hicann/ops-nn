/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_host/arch35/l2_normalize_tiling_arch35.cpp
// =============================================================================
//
// ROLE: Host-side common tiling for L2Normalize on arch35 (Ascend 950).
//   主要步骤：
//     §3.1 GetPlatformInfo      平台参数（coreNum/ubSize/blockSize/cacheLineSize，禁写死）
//     §3.2 NormalizeAxes        axes 归一化（空列表为 no-reduce → 越界校验 → 负转正 → 升序 → 集合化）
//     §3.3 四步合轴             DropSizeOneAxes → FuseAxis → PadLeadingOneA → PadRIfPureA
//     §3.4 HandleEmptyTensor    空 tensor 短路（EMPTY_A / EMPTY_R 均全核早退）
//     §3.5 异常值校验           dtype → format → 维度 → attr（tiling 入口硬门禁）
//     §4.1 ComputeAUbFactor     cachelineTmp 联合爬坡定 aSplitIdx / aUbFactor
//     §4.2 ComputeRUbFactor     postBufSize → r_i_max 反解 → R 端爬坡 + burst 尾轴对齐
//     §4.3 ExpandAIfRFullyLoaded 所有 R 全驻后扩 A（cacheBuf 4096 lane 钳制）
//     §5   ComputeFusedALoopSplit / ComputeRLoopCnt（大小核均衡 / R loop 扁平化）
//     §6.1 ShouldUseGroup / §6.2 ComputeGroupSplit / §6.3 ScheduleMode + Workspace
//     §7   ComputeUbSizes + FillAndLogTilingData + SetTilingKey/SetBlockDim
//   分支判定：校验 → empty(TPL_SEL_1)
//   → group(TPL_SEL_2, ShouldUseGroup) → base(TPL_SEL_0 兜底)；tilingKey 数值由框架宏
//   GET_TPL_TILING_KEY(isGroup, isEmptyTensor) 生成。
//   workspace 使用 padded 槽位精化布局
//   （base 两遍 = aLoopCntTotal × aUnit × 4B；group = partial 区 + denom 区 + tail-A 余量）。
// =============================================================================

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "register/op_def_registry.h"                          // IMPL_OP_OPTILING, OP_ADD
#include "op_common/log/log.h"                                 // OP_LOGE, OP_LOGI, OP_CHECK_*
#include "op_common/op_host/util/platform_util.h"              // PlatformAscendC, GetUbBlockSize/GetCacheLineSize
#include "op_common/op_host/util/math_util.h"                  // Ops::Base::CeilAlign/FloorAlign/CeilDiv
#include "graph/utils/type_utils.h"                            // TypeUtils::DataTypeToSerialString
#include "securec.h"                                           // memset_s
#include "../../op_kernel/arch35/l2_normalize_tiling_struct.h" // L2NormalizeTilingData, L2NormalizeEmptyTilingData
#include "../../op_kernel/arch35/l2_normalize_struct.h"        // TPL 声明（GET_TPL_TILING_KEY）
#include "l2_normalize_tiling_arch35.h"                        // L2NormalizeCompileInfo

namespace optiling {

namespace {
constexpr size_t INPUT_X_IDX = 0;
constexpr size_t OUTPUT_Y_IDX = 0;  // 输出 y（shape 一致性校验用）
constexpr size_t ATTR_AXIS_IDX = 0; // attr 按索引访问：axis=0，eps=1
constexpr size_t ATTR_EPS_IDX = 1;

constexpr int32_t MIN_AXIS_NUM = 2;                // 合轴后轴数下限（AR）
constexpr int32_t MAX_AXIS_NUM = MAX_PATTERN_RANK; // 合轴后轴数上限 = 9
constexpr int32_t MAX_INPUT_RANK = 8;              // 契约 1D–8D
constexpr int64_t CACHE_BUF_BYTES = 16 * 1024;     // cacheBuf（二分缓存树）固定 16KB
constexpr int64_t FP32_BYTES = 4;                  // fp32 单元素字节数（中间精度）
constexpr float DEFAULT_EPS = 1e-4f;               // attr eps 默认值（canndev IR 一致）

// UB 预算系数：pre 侧 3 份（preIn + preRes + preResTail = P_PRE + P_PRE_EXT），post 侧 1 份（denom）
constexpr int64_t P_PRE = 2;
constexpr int64_t P_PRE_EXT = 1;
constexpr int64_t P_POST = 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴的固定间距
constexpr int32_t AXIS_INTERVAL = 2;
// Group 触发阈值分母：aLoopCntTotal ≤ coreNum/2 且 rLoopCntTotal ≥ 2 时 2D 分核
constexpr int64_t GROUP_CORE_RATIO = 2;

bool CheckedMulInt64(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs)) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAddInt64(int64_t lhs, int64_t rhs, int64_t& result)
{
    if (lhs < 0 || rhs < 0 || rhs > std::numeric_limits<int64_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

bool CheckedAlignInt64(int64_t value, int64_t alignment, int64_t& result)
{
    if (value < 0 || alignment <= 0) {
        return false;
    }
    const int64_t remainder = value % alignment;
    return (remainder == 0) ? (result = value, true) : CheckedAddInt64(value, alignment - remainder, result);
}

bool CheckedCeilDivInt64(int64_t value, int64_t divisor, int64_t& result)
{
    if (value < 0 || divisor <= 0) {
        return false;
    }
    result = value / divisor + ((value % divisor) != 0 ? 1 : 0);
    return true;
}

bool CheckedMulSize(size_t lhs, size_t rhs, size_t& result)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool CheckedAddSize(size_t lhs, size_t rhs, size_t& result)
{
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        return false;
    }
    result = lhs + rhs;
    return true;
}

ge::graphStatus ReportIntegerOverflow(gert::TilingContext* context, const char* field)
{
    OP_LOGE(context->GetNodeName(), "integer overflow while computing %s", field);
    return ge::GRAPH_FAILED;
}
} // namespace

// tiling 全程工作状态：一次 TilingFunc 调用内的全部中间量，按生产阶段分区
struct L2NormalizeCtx {
    // ── 平台参数（GetPlatformInfo 填充）──
    int64_t coreNum = 0;
    int64_t ubSize = 0;
    int64_t blockSize = 0;
    int64_t cacheLineSize = 0;

    // ── 输入信息（GetShapeAndDtype / ReadAndValidateAttrs 填充）──
    ge::DataType xDtype = ge::DT_UNDEFINED;
    int64_t dtypeSize = 0;
    int64_t maxDtypeSize = 0;
    std::vector<int64_t> xShape;
    std::vector<int64_t> reduceAxes; // 归一化后（非负、升序、去重）
    float eps = DEFAULT_EPS;

    // ── A/R 规整模式（BuildInitialAxisList + PreprocessPattern 填充）──
    std::vector<int64_t> axisShape;
    std::vector<bool> isReduceAxis;
    int32_t axisNum = 0;
    bool isTailR = false;

    // ── UB 切分参数（ComputeAUbFactor / ComputeRUbFactor / ExpandAIfRFullyLoaded 填充）──
    int32_t aSplitIdx = 0;
    int32_t rSplitIdx = 0;
    int64_t aUbFactor = 0;
    int64_t rUbFactor = 0;
    int64_t rUbFactorAlign = 0;
    int64_t innerAProdAlign = 0;
    int64_t innerRProdAlign = 0;

    // ── A 方向多核参数（ComputeFusedALoopSplit 填充）──
    int64_t aLoopCntTotal = 0;
    int64_t aSplitChunkCnt = 0;
    int64_t aBigCoreLoopCnt = 0;
    int64_t aSmallCoreLoopCnt = 0;
    int32_t aBigCoreCnt = 0;
    int32_t usedCoreNum = 0;

    // ── R 方向迭代数（ComputeRLoopCnt 填充）──
    int64_t rLoopCntTotal = 0;

    // ── UB buffer 尺寸（ComputeUbSizes 填充）──
    int64_t preBufSize = 0;
    int64_t postBufSize = 0;

    // ── Group 2D 分核参数（ComputeGroupSplit 填充）──
    bool isGroup = false;
    int64_t rGroupCnt = 0;
};

// ---------------------------------------------------------------------------
// GetPlatformInfo — 平台硬件参数获取（§3.1）
//   coreNum / ubSize / blockSize / cacheLineSize 全走 platform 接口，逐个非 0
//   校验，禁止写死（范式 [3.2] 约束）。
// ---------------------------------------------------------------------------
static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, L2NormalizeCtx& ctx)
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

    ctx.blockSize = static_cast<int64_t>(Ops::Base::GetUbBlockSize(context));
    ctx.cacheLineSize = static_cast<int64_t>(Ops::Base::GetCacheLineSize(context));
    OP_CHECK_IF(ctx.blockSize <= 0 || ctx.cacheLineSize < ctx.blockSize || ctx.ubSize <= CACHE_BUF_BYTES,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "platform.blockSize/cacheLineSize",
                                                         "invalid UB geometry: ubSize=" + std::to_string(ctx.ubSize) +
                                                             ", blockSize=" + std::to_string(ctx.blockSize) +
                                                             ", cacheLineSize=" + std::to_string(ctx.cacheLineSize)),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// CheckOutputDesc — 输出 y 与输入 x 的 dtype / format / shape 一致性校验（host 侧硬门禁）
//   y.dtype = x.dtype、y.format = ND、y.shape = x.shape。若调用方提供的输出描述不一致，
//   缺该校验可能使 kernel 按输入描述落设备读写错误的缓冲，而非预期的 host 侧拒绝。
// ---------------------------------------------------------------------------
static ge::graphStatus CheckOutputDesc(gert::TilingContext* context, const L2NormalizeCtx& ctx)
{
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    OP_CHECK_IF(yDesc->GetDataType() != ctx.xDtype,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "y", ge::TypeUtils::DataTypeToSerialString(yDesc->GetDataType()).c_str(),
                    "y dtype must be the same as x dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        yDesc->GetStorageFormat() != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "y",
                                               std::to_string(static_cast<int64_t>(yDesc->GetStorageFormat())).c_str(),
                                               "only FORMAT_ND is supported"),
        return ge::GRAPH_FAILED);

    auto yShapePtr = context->GetOutputShape(OUTPUT_Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShapePtr);
    const gert::Shape& yS = yShapePtr->GetStorageShape();

    bool match = (yS.GetDimNum() == ctx.xShape.size());
    if (match) {
        for (size_t i = 0; i < ctx.xShape.size(); ++i) {
            if (yS.GetDim(i) != ctx.xShape[i]) {
                match = false;
                break;
            }
        }
    }
    if (!match) {
        std::string yStr = "[";
        for (size_t i = 0; i < yS.GetDimNum(); ++i) {
            if (i > 0) {
                yStr += ", ";
            }
            yStr += std::to_string(yS.GetDim(i));
        }
        yStr += "]";
        std::string xStr = "[";
        for (size_t i = 0; i < ctx.xShape.size(); ++i) {
            if (i > 0) {
                xStr += ", ";
            }
            xStr += std::to_string(ctx.xShape[i]);
        }
        xStr += "]";
        OP_LOGE_FOR_INVALID_SHAPE(context->GetNodeName(), "y", yStr.c_str(), xStr.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// GetShapeAndDtype — 输入 x 的 shape / dtype / format 读取与校验（§3.5 #1–#3/#5）
//   校验顺序 dtype → format → 维度；负 dim（动态 shape 防御）报错。
//   dtype 白名单 {DT_FLOAT16, DT_FLOAT}（y 与 x 一致由 InferShape 恒等复制和
//   CheckOutputDesc 校验承载，单输入无组合矩阵，§3.5 #5 N/A）。
// ---------------------------------------------------------------------------
static ge::graphStatus GetShapeAndDtype(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    auto xShapePtr = context->GetInputShape(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShapePtr);
    const gert::Shape& xS = xShapePtr->GetStorageShape();

    // dtype/format 必须走 GetInputDesc（GetInputTensor 在部分编译场景
    // 可能返回无效数据，仅可用于取 Shape 之外的信息时亦不例外）；shape 走 GetInputShape。
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(INPUT_X_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    ctx.xDtype = xDesc->GetDataType();

    // #1 dtype 校验：x ∈ {DT_FLOAT16, DT_FLOAT}
    const std::vector<ge::DataType> supportedDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT};
    OP_CHECK_IF(std::find(supportedDtypes.begin(), supportedDtypes.end(), ctx.xDtype) == supportedDtypes.end(),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(),
                                                      "only DT_FLOAT16/DT_FLOAT are supported"),
                return ge::GRAPH_FAILED);

    // #2 format 校验：ND（ge::FORMAT_ND，GetStorageFormat）
    OP_CHECK_IF(
        xDesc->GetStorageFormat() != ge::FORMAT_ND,
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), "x",
                                               std::to_string(static_cast<int64_t>(xDesc->GetStorageFormat())).c_str(),
                                               "only FORMAT_ND is supported"),
        return ge::GRAPH_FAILED);

    // #3 维度校验：1 ≤ rank ≤ 8；rank 0 标量拒绝（IsScalar 官方判定，勿手写 GetDimNum()==0）
    const size_t rank = xS.GetDimNum();
    OP_CHECK_IF(xS.IsScalar() || rank > MAX_INPUT_RANK,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "rank",
                                                         std::to_string(static_cast<int64_t>(rank)).c_str(),
                                                         "x rank must be in [1, 8] (rank-0 scalar is rejected)"),
                return ge::GRAPH_FAILED);

    ctx.xShape.clear();
    bool isEmpty = false;
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = xS.GetDim(i);
        OP_CHECK_IF(
            dim < 0,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "x", std::to_string(dim).c_str(),
                "dim[" + std::to_string(i) + "] is negative (dynamic shape must be instantiated before tiling)"),
            return ge::GRAPH_FAILED);
        ctx.xShape.push_back(dim);
        isEmpty = isEmpty || (dim == 0);
    }

    // dtype 派生量：dtypeSize 元素数取整专用 / maxDtypeSize 字节容量专用（⛔ 禁止混用，§4）
    ctx.dtypeSize = static_cast<int64_t>(ge::GetSizeByDataType(ctx.xDtype));
    ctx.maxDtypeSize = std::max(ctx.dtypeSize, FP32_BYTES);
    OP_CHECK_IF(ctx.dtypeSize <= 0 || ctx.blockSize < ctx.dtypeSize || ctx.blockSize % ctx.dtypeSize != 0,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), "x",
                                                      ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(),
                                                      "dtype size must be positive and divide the UB block size"),
                return ge::GRAPH_FAILED);

    // 非空输入必须能以 int64 element offset 和 byte offset 完整寻址。空 Tensor 不访问
    // 数据，允许未实例化为实际存储的超大其他维组合直接走 empty 短路。
    if (!isEmpty) {
        int64_t totalElements = 1;
        for (int64_t dim : ctx.xShape) {
            if (!CheckedMulInt64(totalElements, dim, totalElements)) {
                return ReportIntegerOverflow(context, "input element count");
            }
        }
        int64_t totalBytes = 0;
        if (!CheckedMulInt64(totalElements, ctx.dtypeSize, totalBytes)) {
            return ReportIntegerOverflow(context, "input byte size");
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ReadAndValidateAttrs — attr 读取 + 校验 + INFO 全参数日志（§3.2 / §3.5 #4 / §2）
//   axis：空列表按 no-reduce 处理 → 越界校验（负数转正之前，-rank-1 转正后变 -1 会漏检）→
//         负数转正 → 升序排序 → 集合化；
//   eps：可选 attr 空时用默认值 1e-4f（不报错）；REG_OP 公开契约为 Float，不额外收窄值域。
// ---------------------------------------------------------------------------
static ge::graphStatus ReadAndValidateAttrs(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    const auto* attrs = context->GetAttrs();
    const gert::TypedContinuousVector<int64_t>* axisAttr = (attrs != nullptr) ? attrs->GetListInt(ATTR_AXIS_IDX) :
                                                                                nullptr;
    const size_t axesNum = (axisAttr != nullptr) ? axisAttr->GetSize() : 0U;

    // eps（可选 attr 空时用默认值）
    const float* epsAttr = (attrs != nullptr) ? attrs->GetFloat(ATTR_EPS_IDX) : nullptr;
    ctx.eps = (epsAttr != nullptr) ? *epsAttr : DEFAULT_EPS;

    // 空 axis 是公开契约的默认值，表示不归约；后续 PadRIfPureA 会补 R=1 占位轴，
    // 复用同一 kernel 路径得到 y = x / sqrt(max(x*x, eps))。
    const int64_t xRank = static_cast<int64_t>(ctx.xShape.size());
    const int64_t* axisData = (axesNum > 0U) ? axisAttr->GetData() : nullptr;
    OP_CHECK_IF(axesNum > 0 && axisData == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", "nullptr",
                                                      "axis data must not be null when axis is non-empty"),
                return ge::GRAPH_FAILED);
    ctx.reduceAxes.clear();
    for (size_t i = 0; i < axesNum; ++i) {
        const int64_t v = axisData[i];
        OP_CHECK_IF(
            v < -xRank || v >= xRank,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "axis", std::to_string(v).c_str(),
                                                  "axis element must be in [-rank, rank) = [" + std::to_string(-xRank) +
                                                      ", " + std::to_string(xRank) + ")"),
            return ge::GRAPH_FAILED);
        ctx.reduceAxes.push_back(v < 0 ? v + xRank : v);
    }
    std::sort(ctx.reduceAxes.begin(), ctx.reduceAxes.end());
    ctx.reduceAxes.erase(std::unique(ctx.reduceAxes.begin(), ctx.reduceAxes.end()), ctx.reduceAxes.end());

    std::string axisStr = "[";
    for (size_t i = 0; i < ctx.reduceAxes.size(); ++i) {
        if (i > 0U) {
            axisStr += ", ";
        }
        axisStr += std::to_string(ctx.reduceAxes[i]);
    }
    axisStr += "]";

    // INFO 级日志打印每个输入参数/属性（范式 [3.1] 通用要求）
    std::string shapeStr = "[";
    for (size_t i = 0; i < ctx.xShape.size(); ++i) {
        if (i > 0) {
            shapeStr += ", ";
        }
        shapeStr += std::to_string(ctx.xShape[i]);
    }
    shapeStr += "]";
    OP_LOGI(context->GetNodeName(), "L2Normalize input: xShape=%s, xDtype=%s, format=ND, axis=%s, eps=%f",
            shapeStr.c_str(), ge::TypeUtils::DataTypeToSerialString(ctx.xDtype).c_str(), axisStr.c_str(), ctx.eps);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// HandleEmptyTensor — 空 tensor 短路（§3.4，EMPTY_A / EMPTY_R 合一）
//   本算子 y.shape = x.shape：A 含 0 或 R 含 0，输出均为同维空 tensor → 全核早退
//   （零计算零 IO）；范式 EMPTY_R 的 Duplicate 固化值 + CopyOut 路径与 aUbFactor 4
//   约束切分对本算子 N/A（Empty struct 字段仅保留范式布局一致性，kernel 侧不消费）。
// ---------------------------------------------------------------------------
static ge::graphStatus HandleEmptyTensor(gert::TilingContext* context)
{
    auto* tiling = context->GetTilingData<L2NormalizeEmptyTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(memset_s(tiling, sizeof(L2NormalizeEmptyTilingData), 0, sizeof(L2NormalizeEmptyTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "Memset tilingdata error"), return ge::GRAPH_FAILED);
    tiling->usedCoreNum = 0; // ⛔ 严禁 SetBlockDim(0)；kernel 侧按 usedCoreNum==0 短路
    OP_LOGI(context->GetNodeName(),
            "L2Normalize empty tensor short-circuit: usedCoreNum=0, aTotal=0, EMPTY_A/EMPTY_R early-exit");
    context->SetBlockDim(1);                                                       // 框架要求 ≥ 1
    context->SetTilingKey(GET_TPL_TILING_KEY(/*isGroup=*/0, /*isEmptyTensor=*/1)); // TPL_SEL_1
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0U; // 早退无系统/用户 workspace 需求，显式设置
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// 四步合轴（§3.3，顺序不可换）
// ---------------------------------------------------------------------------

// 1. 去 1 轴：无论 A/R，size==1 的整根轴删除；全 1 退化（∏shape==1）保留 1 根占位 A=1
static void DropSizeOneAxes(L2NormalizeCtx& ctx)
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

// 2. 合轴：连续 A 轴合并为 1 根 A 轴，连续 R 轴合并为 1 根 R 轴（乘积合并）
static ge::graphStatus FuseAxis(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    std::vector<int64_t> fusedShape;
    std::vector<bool> fusedIsR;
    for (size_t i = 0; i < ctx.axisShape.size(); ++i) {
        if (!fusedShape.empty() && fusedIsR.back() == ctx.isReduceAxis[i]) {
            if (!CheckedMulInt64(fusedShape.back(), ctx.axisShape[i], fusedShape.back())) {
                return ReportIntegerOverflow(context, "fused axis size");
            }
        } else {
            fusedShape.push_back(ctx.axisShape[i]);
            fusedIsR.push_back(ctx.isReduceAxis[i]);
        }
    }
    ctx.axisShape = std::move(fusedShape);
    ctx.isReduceAxis = std::move(fusedIsR);
    return ge::GRAPH_SUCCESS;
}

// 3. 补 leading A=1：若首轴为 R，前置一根 A=1
static void PadLeadingOneA(L2NormalizeCtx& ctx)
{
    if (!ctx.axisShape.empty() && ctx.isReduceAxis.front()) {
        ctx.axisShape.insert(ctx.axisShape.begin(), 1);
        ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
    }
}

// 4. 补 R 增广：纯 A 时——A=1（全 1 退化）末尾补 [R=1] 变 AR；A>1 前置 [A=1, R=1] 变 ARA
//    ⛔ A>1 绝不能末尾补 R=1 的 AR（tail R=1 且 A>1 会使 burst 长度塌底，性能崩）
static void PadRIfPureA(L2NormalizeCtx& ctx)
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
            ctx.axisShape.push_back(1);
            ctx.isReduceAxis.push_back(true);
        } else {
            ctx.axisShape.insert(ctx.axisShape.begin(), 1);
            ctx.axisShape.insert(ctx.axisShape.begin() + 1, 1);
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), true);
            ctx.isReduceAxis.insert(ctx.isReduceAxis.begin(), false);
        }
    }
}

// ---------------------------------------------------------------------------
// PreprocessPattern — 合轴总装 + 不变量自检（§3.3）
//   合轴后 pattern 必然 A 起头、A/R 严格交替、axisNum ∈ [2, 9]；
//   tail 类型由 axisNum 奇偶现算：偶 → tail-R，奇 → tail-A。
// ---------------------------------------------------------------------------
static ge::graphStatus PreprocessPattern(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    DropSizeOneAxes(ctx);
    OP_CHECK_IF(FuseAxis(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
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
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "isReduceAxis",
                        "axis[" + std::to_string(i) +
                            "] type mismatch after A/R regularization (expected alternating [A,R] pattern)"),
                    return ge::GRAPH_FAILED);
    }
    ctx.isTailR = (ctx.axisNum % AXIS_INTERVAL == 0);

    std::string axisStr = "[";
    for (int32_t i = 0; i < ctx.axisNum; ++i) {
        if (i > 0) {
            axisStr += ", ";
        }
        axisStr += std::to_string(ctx.axisShape[static_cast<size_t>(i)]);
        axisStr += ctx.isReduceAxis[static_cast<size_t>(i)] ? "(R)" : "(A)";
    }
    axisStr += "]";
    OP_LOGI(context->GetNodeName(), "L2Normalize tiling: axisNum=%d, isTailR=%d, axes=%s", ctx.axisNum,
            static_cast<int>(ctx.isTailR), axisStr.c_str());
    return ge::GRAPH_SUCCESS;
}

// 计算规整后各轴 GM 步长（element 计）：stride[i] = ∏_{j>i} axisShape[j]（§7.2）
static ge::graphStatus ComputeAxisStrides(gert::TilingContext* context, const L2NormalizeCtx& ctx, int64_t outStride[])
{
    int64_t strideAcc = 1;
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        outStride[i] = strideAcc;
        if (!CheckedMulInt64(strideAcc, ctx.axisShape[static_cast<size_t>(i)], strideAcc)) {
            return ReportIntegerOverflow(context, "axis stride");
        }
    }
    return ge::GRAPH_SUCCESS;
}

// 返回最后一个 A 轴（最大偶下标）；规整后恒存在，兜底返回 0（防御）
static int32_t LastAAxisIdx(const L2NormalizeCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

// 返回最后一个 R 轴（最大奇下标）；规整后恒存在，兜底返回 1（防御）
static int32_t LastRAxisIdx(const L2NormalizeCtx& ctx)
{
    for (int32_t i = ctx.axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// 计算全部 A 轴长度乘积 = 输出侧 A 元素总数 aTotal（group workspace partial 区列数 / 扩 A 上限）
static ge::graphStatus TotalAProd(gert::TilingContext* context, const L2NormalizeCtx& ctx, int64_t& total)
{
    total = 1;
    for (int32_t i = 0; i < ctx.axisNum; i += AXIS_INTERVAL) {
        if (!CheckedMulInt64(total, ctx.axisShape[static_cast<size_t>(i)], total)) {
            return ReportIntegerOverflow(context, "total A elements");
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeInnerAProdAlign — aSplitIdx 内侧 A 轴的对齐乘积（§4.1 (4)）
//   仅 tail-A 的最内 A 轴（搬运 burst 轴）按 bsElem 向上对齐，其余轴原值累乘。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeInnerAProdAlign(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    ctx.innerAProdAlign = 1;
    for (int32_t k = ctx.aSplitIdx + AXIS_INTERVAL; k < ctx.axisNum; k += AXIS_INTERVAL) {
        int64_t axisSize = ctx.axisShape[static_cast<size_t>(k)];
        if (k == ctx.axisNum - 1 && !ctx.isTailR && !CheckedAlignInt64(axisSize, bsElem, axisSize)) {
            return ReportIntegerOverflow(context, "aligned inner A axis");
        }
        if (!CheckedMulInt64(ctx.innerAProdAlign, axisSize, ctx.innerAProdAlign)) {
            return ReportIntegerOverflow(context, "aligned inner A product");
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeAUbFactor — UB 切分 Step 1（§4.1）
//   cachelineTmp 联合爬坡：A 轴与 R 轴一起从最内轴向外爬坡，终止于
//   product × axisSize > cachelineTmp（⛔ 用 > 而非 >=：tail-A 下 CeilAlign(LastA)
//   == cachelineTmp 时仍吸入 LastA，保证 aUnit 一定对齐）；停在 A 轴 → 按剩余预算
//   切该轴（向下取整，不保证 block 对齐）；停在 R 轴 → R 不能作 A 切分点，左移一根
//   A、每 chunk 取 1；全部装得下 → 不切（aSplitIdx=0）。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeAUbFactor(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    const int64_t maxInnerAInitElem = ctx.cacheLineSize / ctx.dtypeSize;
    const int64_t cachelineTmp = (maxInnerAInitElem / bsElem) * bsElem;

    int64_t product = 1;
    int32_t idx = ctx.axisNum - 1;
    while (idx >= 0) {
        int64_t axisSize = ctx.axisShape[static_cast<size_t>(idx)];
        if (idx == ctx.axisNum - 1 && !CheckedAlignInt64(axisSize, bsElem, axisSize)) {
            return ReportIntegerOverflow(context, "aligned innermost axis");
        }
        if (axisSize > cachelineTmp / product) {
            break;
        }
        if (!CheckedMulInt64(product, axisSize, product)) {
            return ReportIntegerOverflow(context, "initial UB axis product");
        }
        idx--;
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else if (idx % AXIS_INTERVAL == 0) {
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(cachelineTmp / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    } else {
        ctx.aSplitIdx = idx - 1;
        ctx.aUbFactor = 1;
    }

    OP_CHECK_IF(ComputeInnerAProdAlign(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeRiMax — 反解单次 R 迭代可用元素上限 r_i_max（§4.2 (1)）
//   (ubSize − 16KB cacheBuf − postBuf) / (3 个 pre buffer × aUnit × maxDtypeSize)；
//   aUnit 非法或预算不足时返回 -1（由调用方报 TILING_FAIL）。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeRiMax(gert::TilingContext* context, const L2NormalizeCtx& ctx, int64_t& rIMax)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    int64_t aUnit = 0;
    int64_t aUnitBytes = 0;
    int64_t postBufSize = 0;
    int64_t bytesPerRElem = 0;
    if (!CheckedMulInt64(ctx.aUbFactor, ctx.innerAProdAlign, aUnit) ||
        !CheckedMulInt64(aUnit, ctx.maxDtypeSize, aUnitBytes) ||
        !CheckedAlignInt64(aUnitBytes, ctx.blockSize, postBufSize) ||
        !CheckedMulInt64(P_PRE + P_PRE_EXT, aUnitBytes, bytesPerRElem)) {
        return ReportIntegerOverflow(context, "R UB budget");
    }
    if (aUnit <= 0 || bytesPerRElem <= 0) {
        rIMax = -1;
        return ge::GRAPH_SUCCESS;
    }
    const int64_t numer = ubAvailable - P_POST * postBufSize;
    if (numer <= 0) {
        rIMax = -1;
        return ge::GRAPH_SUCCESS;
    }
    rIMax = numer / bytesPerRElem;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeRUbFactor — UB 切分 Step 2（§4.2）
//   在 r_i_max 预算内从最内 R 轴向外吸收 R 轴（innerRProdAlign / rSplitIdx），再定
//   rUbFactor；切在最内 R 轴（isTailR 且 rSplitIdx==lastR，burst 尾轴方向）时做 32B
//   对齐得 rUbFactorAlign——部分切分 FloorAlign（对齐到 0 报 TILING_FAIL）/ 整轴
//   CeilAlign 后校验不超预算、超则退回 FloorAlign；非 burst 尾轴方向不对齐。
//   valid/padded 双字段：rUbFactor（valid）与 rUbFactorAlign（padded）分离。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeRUbFactor(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    const int32_t lastR = LastRAxisIdx(ctx);
    int64_t rIMax = 0;
    OP_CHECK_IF(ComputeRiMax(context, ctx, rIMax) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
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
        int64_t axisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
        if (ctx.rSplitIdx == lastR && ctx.isTailR && !CheckedAlignInt64(axisSize, bsElem, axisSize)) {
            return ReportIntegerOverflow(context, "aligned inner R axis");
        }
        if (axisSize > rIMax / ctx.innerRProdAlign) {
            break;
        }
        if (!CheckedMulInt64(ctx.innerRProdAlign, axisSize, ctx.innerRProdAlign)) {
            return ReportIntegerOverflow(context, "aligned inner R product");
        }
        ctx.rSplitIdx -= AXIS_INTERVAL;
    }

    const int64_t rAxisSize = ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)];
    ctx.rUbFactor = std::min(rIMax / ctx.innerRProdAlign, rAxisSize);
    ctx.rUbFactor = std::max<int64_t>(ctx.rUbFactor, 1);

    const bool isBurstTailR = (ctx.isTailR && ctx.rSplitIdx == lastR);
    if (isBurstTailR) {
        if (ctx.rUbFactor < rAxisSize) {
            // 切多 chunk：FloorAlign（⛔ 连一个 block 都装不下 → TILING_FAIL）
            ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
            OP_CHECK_IF(
                ctx.rUbFactor == 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rUbFactor",
                                                         "rUbFactor floor-aligned to 0 (tail-R burst alignment)"),
                return ge::GRAPH_FAILED);
            ctx.rUbFactorAlign = ctx.rUbFactor;
        } else {
            // 整轴：CeilAlign 后校验不超预算，超则退回切多 chunk
            if (!CheckedAlignInt64(ctx.rUbFactor, bsElem, ctx.rUbFactorAlign)) {
                return ReportIntegerOverflow(context, "aligned R UB factor");
            }
            if (ctx.rUbFactorAlign > rIMax / ctx.innerRProdAlign) {
                ctx.rUbFactor = Ops::Base::FloorAlign(ctx.rUbFactor, bsElem);
                ctx.rUbFactorAlign = ctx.rUbFactor;
            }
        }
    } else {
        ctx.rUbFactorAlign = ctx.rUbFactor; // 非 burst 尾轴方向不对齐
    }
    return ge::GRAPH_SUCCESS;
}

// 判定整个 R 空间是否单次迭代全载：rUbFactor 占满 rSplit 轴长度，且 rSplitIdx 外侧无 R 轴
static bool RIsFullyLoaded(const L2NormalizeCtx& ctx)
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

// R 全载时把 UB 预算反解为 A 上限（§4.3）：
//   aUnitMax = ubAvailable / (maxDtypeSize × (3 × rPaddedElems + 1))
static ge::graphStatus SolveAUnitMax(gert::TilingContext* context, const L2NormalizeCtx& ctx, int64_t& aUnitMax)
{
    const int64_t ubAvailable = ctx.ubSize - CACHE_BUF_BYTES;
    int64_t rPaddedElems = 0;
    int64_t weightedElems = 0;
    int64_t coeffElems = 0;
    int64_t coeff = 0;
    if (!CheckedMulInt64(ctx.rUbFactorAlign, ctx.innerRProdAlign, rPaddedElems) ||
        !CheckedMulInt64(P_PRE + P_PRE_EXT, rPaddedElems, weightedElems) ||
        !CheckedAddInt64(weightedElems, P_POST, coeffElems) || !CheckedMulInt64(coeffElems, ctx.maxDtypeSize, coeff)) {
        return ReportIntegerOverflow(context, "expanded A UB budget");
    }
    if (coeff <= 0) {
        aUnitMax = -1;
        return ge::GRAPH_SUCCESS;
    }
    aUnitMax = ubAvailable / coeff;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ExpandAIfRFullyLoaded — UB 切分 Step 3：所有 R 全驻后扩 A（§4.3）
//   aUnitMax 受三重约束：UB 反解值 / totalA / cacheBuf 行宽上限（16KB÷4 = 4096 lane，
//   R 全驻 cacheCount=1 时不钳制树溢出）；tail-A 时按 bsElem 向下对齐（⚠ 必须在
//   min(∏A) 钳制之后）；从 lastA 向内重新累积定 aSplitIdx / aUbFactor 并重算
//   innerAProdAlign；无利可图（≤ 当前值）则保持不变。
// ---------------------------------------------------------------------------
static ge::graphStatus ExpandAIfRFullyLoaded(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    if (!RIsFullyLoaded(ctx)) {
        return ge::GRAPH_SUCCESS;
    }

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    const int64_t cacheLaneLimit = CACHE_BUF_BYTES / FP32_BYTES;
    int64_t totalA = 0;
    OP_CHECK_IF(TotalAProd(context, ctx, totalA) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    int64_t aUnitMax = 0;
    OP_CHECK_IF(SolveAUnitMax(context, ctx, aUnitMax) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    aUnitMax = std::min(aUnitMax, totalA);
    aUnitMax = std::min(aUnitMax, cacheLaneLimit);
    if (aUnitMax <= 0) {
        return ge::GRAPH_SUCCESS;
    }
    if (!ctx.isTailR) {
        aUnitMax = Ops::Base::FloorAlign(aUnitMax, bsElem);
    }

    int64_t curAUnit = 0;
    if (!CheckedMulInt64(ctx.aUbFactor, ctx.innerAProdAlign, curAUnit)) {
        return ReportIntegerOverflow(context, "current A unit");
    }
    if (aUnitMax <= curAUnit) {
        return ge::GRAPH_SUCCESS;
    }

    // 用 aUnitMax 替换目标重做 A 端爬坡（只爬 A 轴，R 已全驻不参与）
    int64_t product = 1;
    int32_t idx = LastAAxisIdx(ctx);
    while (idx >= 0) {
        int64_t axisSize = ctx.axisShape[static_cast<size_t>(idx)];
        if (idx == ctx.axisNum - 1 && !ctx.isTailR && !CheckedAlignInt64(axisSize, bsElem, axisSize)) {
            return ReportIntegerOverflow(context, "aligned expanded A axis");
        }
        if (axisSize > aUnitMax / product) {
            break;
        }
        if (!CheckedMulInt64(product, axisSize, product)) {
            return ReportIntegerOverflow(context, "expanded A product");
        }
        idx -= AXIS_INTERVAL;
    }

    if (idx < 0) {
        ctx.aSplitIdx = 0;
        ctx.aUbFactor = ctx.axisShape[0];
    } else {
        ctx.aSplitIdx = idx;
        ctx.aUbFactor = std::min(aUnitMax / product, ctx.axisShape[static_cast<size_t>(idx)]);
        ctx.aUbFactor = std::max<int64_t>(ctx.aUbFactor, 1);
    }

    OP_CHECK_IF(ComputeInnerAProdAlign(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(),
            "L2Normalize tiling: expand A (R fully loaded): aSplitIdx=%d, aUbFactor=%ld, "
            "innerAProdAlign=%ld",
            ctx.aSplitIdx, ctx.aUbFactor, ctx.innerAProdAlign);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeFusedALoopSplit — A 方向多核切分（§5）
//   aSplitIdx 切分轴的 chunk 数与外层 A 轴整根一起 fuse 成线性计数（row-major，
//   chunk 在最内）；按核数做大小核均分（负载差 ≤ 1，尾块由 chunk 划分自然形成）。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeFusedALoopSplit(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    int64_t outerAProd = 1;
    for (int32_t i = 0; i < ctx.aSplitIdx; i += AXIS_INTERVAL) {
        if (!CheckedMulInt64(outerAProd, ctx.axisShape[static_cast<size_t>(i)], outerAProd)) {
            return ReportIntegerOverflow(context, "outer A loop count");
        }
    }
    const int64_t aSplitAxisSize = ctx.axisShape[static_cast<size_t>(ctx.aSplitIdx)];
    if (!CheckedCeilDivInt64(aSplitAxisSize, ctx.aUbFactor, ctx.aSplitChunkCnt) ||
        !CheckedMulInt64(outerAProd, ctx.aSplitChunkCnt, ctx.aLoopCntTotal)) {
        return ReportIntegerOverflow(context, "fused A loop count");
    }

    ctx.aSmallCoreLoopCnt = ctx.aLoopCntTotal / ctx.coreNum;
    ctx.aBigCoreCnt = static_cast<int32_t>(ctx.aLoopCntTotal % ctx.coreNum);
    ctx.aBigCoreLoopCnt = ctx.aSmallCoreLoopCnt + (ctx.aBigCoreCnt > 0 ? 1 : 0);
    ctx.usedCoreNum = (ctx.aSmallCoreLoopCnt > 0) ? static_cast<int32_t>(ctx.coreNum) : ctx.aBigCoreCnt;
    ctx.usedCoreNum = std::max(ctx.usedCoreNum, 1);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeRLoopCnt — R 方向总迭代数（§5）
//   rLoopCntTotal = ∏(外层 R 轴 size) × CeilDiv(axisShape[rSplitIdx], rUbFactor)
//   （=1 即 R 全载单遍路径，>1 即 R 切分两遍路径）
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeRLoopCnt(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    int64_t outerR = 1;
    for (int32_t i = 1; i < ctx.rSplitIdx; i += AXIS_INTERVAL) {
        if (!CheckedMulInt64(outerR, ctx.axisShape[static_cast<size_t>(i)], outerR)) {
            return ReportIntegerOverflow(context, "outer R loop count");
        }
    }
    int64_t rChunks = 0;
    if (!CheckedCeilDivInt64(ctx.axisShape[static_cast<size_t>(ctx.rSplitIdx)], ctx.rUbFactor, rChunks) ||
        !CheckedMulInt64(outerR, rChunks, ctx.rLoopCntTotal)) {
        return ReportIntegerOverflow(context, "total R loop count");
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ShouldUseGroup — Group 触发判定（§6.1）
//   A 用不满核（aLoopCntTotal ≤ coreNum/2）且 R 有并行度（rLoopCntTotal ≥ 2）→ group。
//   阈值取 coreNum/2：A×R 2D 分核时 A 切片不可再拆（UB 切分后最小调度单元），
//   aLoopCntTotal > coreNum/2 时即使 rGroupCnt=2 也会 usedCoreNum > coreNum，
//   NPU 不允许，SyncAll() 会挂死。
// ---------------------------------------------------------------------------
static bool ShouldUseGroup(const L2NormalizeCtx& ctx)
{
    if (ctx.aLoopCntTotal > ctx.coreNum / GROUP_CORE_RATIO) {
        return false;
    }
    if (ctx.rLoopCntTotal <= 1) {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// ComputeGroupSplit — A×R 2D 分核（§6.2）
//   totalOuter = aLoopCntTotal × rLoopCntTotal 按核均分得 numBlocks，再对齐到
//   aLoopCntTotal 的整数倍（CeilAlign 超核数则 FloorAlign）——保证每核 = 整数个完整
//   A chunk × 一段完整 R 区间；得 usedCoreNum 与 rGroupCnt（aPerCore=1 恒成立）。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeGroupSplit(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    int64_t totalOuter = 0;
    int64_t perCoreNum = 0;
    int64_t numBlocks = 0;
    if (!CheckedMulInt64(ctx.aLoopCntTotal, ctx.rLoopCntTotal, totalOuter) ||
        !CheckedCeilDivInt64(totalOuter, ctx.coreNum, perCoreNum) ||
        !CheckedCeilDivInt64(totalOuter, perCoreNum, numBlocks)) {
        return ReportIntegerOverflow(context, "group split");
    }

    int64_t alignedBlocks = 0;
    if (!CheckedAlignInt64(numBlocks, ctx.aLoopCntTotal, alignedBlocks)) {
        return ReportIntegerOverflow(context, "aligned group block count");
    }
    if (alignedBlocks <= ctx.coreNum) {
        numBlocks = alignedBlocks;
    } else {
        numBlocks = Ops::Base::FloorAlign(numBlocks, ctx.aLoopCntTotal);
    }

    OP_CHECK_IF(numBlocks <= 0 || numBlocks > ctx.coreNum,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "group blocks",
                                                         "computed group block count is out of range"),
                return ge::GRAPH_FAILED);
    ctx.usedCoreNum = static_cast<int32_t>(numBlocks);
    ctx.rGroupCnt = numBlocks / ctx.aLoopCntTotal;
    OP_CHECK_IF(ctx.rGroupCnt <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "rGroupCnt",
                                                         "computed rGroupCnt must be positive"),
                return ge::GRAPH_FAILED);
    ctx.isGroup = true;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeUbSizes — UB buffer 尺寸（§7.1）
//   preBufSize = aUnit × rPaddedElems × maxDtypeSize（pre 阶段 2D dense，burst 尾轴
//   方向已由 rUbFactorAlign / innerAProdAlign 保证对齐，按乘积直算）；
//   postBufSize = CeilAlign(aUnit × maxDtypeSize, blockSize)（post 阶段 1D 必须
//   block 对齐）；cacheBufUbSize 恒定 16KB。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeUbSizes(gert::TilingContext* context, L2NormalizeCtx& ctx)
{
    int64_t aUnit = 0;
    int64_t rPaddedElems = 0;
    int64_t tileElems = 0;
    int64_t aBytes = 0;
    int64_t preBuffersBytes = 0;
    int64_t ubUsed = 0;
    if (!CheckedMulInt64(ctx.aUbFactor, ctx.innerAProdAlign, aUnit) ||
        !CheckedMulInt64(ctx.rUbFactorAlign, ctx.innerRProdAlign, rPaddedElems) ||
        !CheckedMulInt64(aUnit, rPaddedElems, tileElems) ||
        !CheckedMulInt64(tileElems, ctx.maxDtypeSize, ctx.preBufSize) ||
        !CheckedMulInt64(aUnit, ctx.maxDtypeSize, aBytes) ||
        !CheckedAlignInt64(aBytes, ctx.blockSize, ctx.postBufSize) ||
        !CheckedMulInt64(P_PRE + P_PRE_EXT, ctx.preBufSize, preBuffersBytes) ||
        !CheckedAddInt64(preBuffersBytes, ctx.postBufSize, ubUsed) ||
        !CheckedAddInt64(ubUsed, CACHE_BUF_BYTES, ubUsed)) {
        return ReportIntegerOverflow(context, "UB buffer sizes");
    }
    // The split factors above are solved from the same three-pre-buffer, aligned-post-buffer and cache-buffer
    // budget. Keep this check as a defensive invariant and overflow/addressability guard; it is not a late
    // fallback tiling policy.
    OP_CHECK_IF(ubUsed > ctx.ubSize || ctx.preBufSize > std::numeric_limits<uint32_t>::max() ||
                    ctx.postBufSize > std::numeric_limits<uint32_t>::max(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "UB buffer sizes",
                                                         "computed buffers exceed UB or uint32 addressable range"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// ComputeGroupP2 — group Phase 2 A 重切分参数（host 侧与 kernel 同式现算，
//   仅用于 workspace denom 区分配，不进 TilingData）
//   aUbFactorP2 = preInElems / rGroupCnt（floor，R 全载 rGroupCnt 行优先）→
//   FloorAlign 到 bsFp32（≥ bsFp32 时）→ min(postBufSize/4)（B3 fp32 denom 行容量，
//   fp32 denom 行容量）→ min(aTotal)；slotStride：tail-A = CeilAlign(aUbFactorP2, bsElem)
//   / tail-R = aUbFactorP2。
// ---------------------------------------------------------------------------
static ge::graphStatus ComputeGroupP2(gert::TilingContext* context, const L2NormalizeCtx& ctx, int64_t aTotal,
                                      int64_t& aUbFactorP2, int64_t& slotStride, int64_t& aSplitChunkCntP2)
{
    constexpr int64_t BS_FP32 = 8; // 32B / 4B
    const int64_t preInElems = ctx.preBufSize / FP32_BYTES;
    aUbFactorP2 = preInElems / ctx.rGroupCnt;
    if (aUbFactorP2 >= BS_FP32) {
        aUbFactorP2 = Ops::Base::FloorAlign(aUbFactorP2, BS_FP32);
    }
    aUbFactorP2 = std::min(aUbFactorP2, ctx.postBufSize / FP32_BYTES);
    aUbFactorP2 = std::min(aUbFactorP2, aTotal);
    aUbFactorP2 = std::max<int64_t>(aUbFactorP2, 1); // 防御（preInElems < rGroupCnt 病态形态）

    const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
    if (ctx.isTailR) {
        slotStride = aUbFactorP2;
    } else if (!CheckedAlignInt64(aUbFactorP2, bsElem, slotStride)) {
        return ReportIntegerOverflow(context, "group denom slot stride");
    }
    if (!CheckedCeilDivInt64(aTotal, aUbFactorP2, aSplitChunkCntP2)) {
        return ReportIntegerOverflow(context, "group Phase 2 chunk count");
    }
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// SetWorkspaceSize — workspace 申报（padded 槽位精化）
//   group：partial 区 rGroupCnt × aTotal × 4B（valid dense）+ denom 区
//          aSplitChunkCntP2 × slotStride × 4B（padded 槽位）+ tail-A 段读余量
//          (bsElem-1) × 4B；
//   base R 切分两遍：denom 区 aLoopCntTotal × aUnit × 4B（逐 aLoop padded 槽位，
//          ≥ aTotal × 4B）；
//   base R 全载单遍：denom 驻 UB，无用户 workspace；
//   总大小恒 = sysWorkspaceSize + 用户区（ascendc 要求显式设置）。
// ---------------------------------------------------------------------------
static ge::graphStatus SetWorkspaceSize(gert::TilingContext* context, const L2NormalizeCtx& ctx)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);

    size_t usrSize = 0;
    int64_t aUnit = 0;
    if (!CheckedMulInt64(ctx.aUbFactor, ctx.innerAProdAlign, aUnit)) {
        return ReportIntegerOverflow(context, "workspace A unit");
    }
    if (ctx.isGroup) {
        int64_t aTotal = 0;
        OP_CHECK_IF(TotalAProd(context, ctx, aTotal) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        int64_t aUbFactorP2 = 0;
        int64_t slotStride = 0;
        int64_t aSplitChunkCntP2 = 0;
        OP_CHECK_IF(
            ComputeGroupP2(context, ctx, aTotal, aUbFactorP2, slotStride, aSplitChunkCntP2) != ge::GRAPH_SUCCESS, ,
            return ge::GRAPH_FAILED);
        const int64_t bsElem = ctx.blockSize / ctx.dtypeSize;
        const int64_t padMargin = ctx.isTailR ? 0 : (bsElem - 1);
        size_t partialElems = 0;
        size_t partialBytes = 0;
        size_t denomElems = 0;
        size_t denomBytes = 0;
        size_t padBytes = 0;
        size_t subtotal = 0;
        if (!CheckedMulSize(static_cast<size_t>(ctx.rGroupCnt), static_cast<size_t>(aTotal), partialElems) ||
            !CheckedMulSize(partialElems, sizeof(float), partialBytes) ||
            !CheckedMulSize(static_cast<size_t>(aSplitChunkCntP2), static_cast<size_t>(slotStride), denomElems) ||
            !CheckedMulSize(denomElems, sizeof(float), denomBytes) ||
            !CheckedMulSize(static_cast<size_t>(padMargin), sizeof(float), padBytes) ||
            !CheckedAddSize(partialBytes, denomBytes, subtotal) || !CheckedAddSize(subtotal, padBytes, usrSize)) {
            return ReportIntegerOverflow(context, "group workspace size");
        }
    } else if (ctx.rLoopCntTotal > 1) {
        // base R 切分两遍路径（reduce→denom 中转→除法）：denom 区逐 aLoop padded 槽位
        size_t denomElems = 0;
        if (!CheckedMulSize(static_cast<size_t>(ctx.aLoopCntTotal), static_cast<size_t>(aUnit), denomElems) ||
            !CheckedMulSize(denomElems, sizeof(float), usrSize)) {
            return ReportIntegerOverflow(context, "base workspace size");
        }
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const size_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    if (!CheckedAddSize(usrSize, sysWorkspaceSize, currentWorkspace[0])) {
        return ReportIntegerOverflow(context, "total workspace size");
    }
    OP_LOGI(context->GetNodeName(),
            "L2Normalize workspace: isGroup=%d, rGroupCnt=%ld, aLoopCntTotal=%ld, usrSize=%zu, ws[0]=%zu",
            static_cast<int>(ctx.isGroup), ctx.rGroupCnt, ctx.aLoopCntTotal, usrSize, currentWorkspace[0]);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// FillAndLogTilingData — 填 TilingData（全字段赋值 + OP_LOGI 全量打印）
//   未用轴槽清零（定长数组约定）；eps 经 attr→TilingData 透传（NumericalStable：
//   不硬编码、不进 TilingKey）。
// ---------------------------------------------------------------------------
static ge::graphStatus FillAndLogTilingData(gert::TilingContext* context, const L2NormalizeCtx& ctx)
{
    auto* td = context->GetTilingData<L2NormalizeTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, td);
    OP_CHECK_IF(memset_s(td, sizeof(L2NormalizeTilingData), 0, sizeof(L2NormalizeTilingData)) != EOK,
                OP_LOGE(context->GetNodeName(), "Memset tilingdata error"), return ge::GRAPH_FAILED);

    int64_t axisStride[MAX_PATTERN_RANK] = {0};
    OP_CHECK_IF(ComputeAxisStrides(context, ctx, axisStride) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    td->axisNum = ctx.axisNum;
    for (int32_t i = 0; i < MAX_PATTERN_RANK; ++i) {
        td->axisShape[i] = (i < ctx.axisNum) ? ctx.axisShape[static_cast<size_t>(i)] : 0;
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
    td->rGroupCnt = ctx.rGroupCnt; // base 路径填 0
    td->eps = ctx.eps;

    // OP_LOGI 全量打印（范式约束：每个字段必须打印；数组字段拼接为字符串）
    std::string shapeStr = "[";
    std::string strideStr = "[";
    for (int32_t i = 0; i < td->axisNum; ++i) {
        if (i > 0) {
            shapeStr += ", ";
            strideStr += ", ";
        }
        shapeStr += std::to_string(td->axisShape[i]);
        strideStr += std::to_string(td->axisStride[i]);
    }
    shapeStr += "]";
    strideStr += "]";
    OP_LOGI(context->GetNodeName(), "L2Normalize tiling: axisNum=%d, axisShape=%s, axisStride=%s", td->axisNum,
            shapeStr.c_str(), strideStr.c_str());
    OP_LOGI(context->GetNodeName(),
            "L2Normalize tiling: aLoopCntTotal=%ld, aSplitChunkCnt=%ld, aBigCoreCnt=%d, aBigCoreLoopCnt=%ld, "
            "aSmallCoreLoopCnt=%ld, usedCoreNum=%d",
            td->aLoopCntTotal, td->aSplitChunkCnt, td->aBigCoreCnt, td->aBigCoreLoopCnt, td->aSmallCoreLoopCnt,
            td->usedCoreNum);
    OP_LOGI(context->GetNodeName(),
            "L2Normalize tiling: aSplitIdx=%d, rSplitIdx=%d, aUbFactor=%ld, rUbFactor=%ld, rUbFactorAlign=%ld, "
            "innerAProdAlign=%ld, innerRProdAlign=%ld",
            td->aSplitIdx, td->rSplitIdx, td->aUbFactor, td->rUbFactor, td->rUbFactorAlign, td->innerAProdAlign,
            td->innerRProdAlign);
    OP_LOGI(context->GetNodeName(),
            "L2Normalize tiling: rLoopCntTotal=%ld, preBufSize=%ld, postBufSize=%ld, cacheBufUbSize=%ld, "
            "rGroupCnt=%ld, eps=%f",
            td->rLoopCntTotal, td->preBufSize, td->postBufSize, td->cacheBufUbSize, td->rGroupCnt, td->eps);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingFuncL2Normalize — tiling 主入口（§9 Tiling 整体结构）
//   编排顺序：平台参数 → 异常值校验（dtype → format → 维度 → attr）→ 空 tensor 短路
//   （TPL_SEL_1）→ 合轴四步 → UB 切分三步 → 多核切分 → Group 判定（TPL_SEL_2 /
//   TPL_SEL_0）→ buffer 尺寸 → 填 TilingData → SetTilingKey + SetBlockDim →
//   SetWorkspaceSize。
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncL2Normalize(gert::TilingContext* context)
{
    L2NormalizeCtx ctx;

    // ── 0) 硬件参数（§3.1）──
    OP_CHECK_IF(GetPlatformInfo(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // ── ★ 异常值校验（§3.5；顺序 dtype → format → 维度 → attr）──
    OP_CHECK_IF(GetShapeAndDtype(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ReadAndValidateAttrs(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    // ── ★ 输出 y 描述一致性（host 侧硬门禁，缺省可能落设备读写错误的 out）──
    OP_CHECK_IF(CheckOutputDesc(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── ★ 空 tensor 短路（合轴四步之前，§3.4；EMPTY_A / EMPTY_R 均全核早退）──
    for (int64_t dim : ctx.xShape) {
        if (dim == 0) {
            return HandleEmptyTensor(context);
        }
    }

    // ── 合轴预处理（§3.3 四步，顺序不可换）──
    ctx.axisShape = ctx.xShape;
    ctx.isReduceAxis.assign(ctx.xShape.size(), false);
    for (int64_t reduceAxis : ctx.reduceAxes) {
        ctx.isReduceAxis[static_cast<size_t>(reduceAxis)] = true;
    }
    OP_CHECK_IF(PreprocessPattern(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── 1) UB 切分（§4 三步）──
    OP_CHECK_IF(ComputeAUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeRUbFactor(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ExpandAIfRFullyLoaded(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── 2) 多核切分（§5）──
    OP_CHECK_IF(ComputeFusedALoopSplit(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(ComputeRLoopCnt(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── 3) Group 判定（§6；base / group 二选一，对非空输入完备划分）──
    if (ShouldUseGroup(ctx)) {
        OP_CHECK_IF(ComputeGroupSplit(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
        // group 模板使用 SyncAll 做 Phase 1→Phase 2 全核同步，host 必须设置（防多核同步挂死）
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "Failed to set ScheduleMode!"), return ge::GRAPH_FAILED);
    }

    // ── 4) 切分后处理（§7 / §6.3）──
    OP_CHECK_IF(ComputeUbSizes(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(FillAndLogTilingData(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);
    OP_CHECK_IF(SetWorkspaceSize(context, ctx) != ge::GRAPH_SUCCESS, , return ge::GRAPH_FAILED);

    // ── SetTilingKey（框架宏按模板实参位序生成，§7.3）+ SetBlockDim（⛔ ≥ 1，严禁超物理核数）──
    const uint64_t tilingKey = GET_TPL_TILING_KEY(static_cast<uint64_t>(ctx.isGroup ? 1U : 0U),
                                                  static_cast<uint64_t>(0U));
    context->SetTilingKey(tilingKey);
    OP_LOGI(context->GetNodeName(), "L2Normalize tiling: isGroup=%d, isEmptyTensor=0", static_cast<int>(ctx.isGroup));
    context->SetBlockDim(static_cast<uint32_t>(std::max(ctx.usedCoreNum, 1)));
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingParseForL2Normalize — 编译期准备（§8）
//   CompileInfo 按范式取空结构（本算子无跨次编译缓存信息，平台参数每次经 §3.1
//   GetPlatformInfo 现取）；恒返回成功。后续若做 binary 复用，缓存字段挂
//   CompileInfo 并在此解析。
// ---------------------------------------------------------------------------
static ge::graphStatus TilingParseForL2Normalize([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(L2Normalize) — Host 侧注册（§8）
//   无 .TilingInputsDataDependency：axis / eps 均为 attr，无 tensor 值依赖输入。
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(L2Normalize)
    .Tiling(TilingFuncL2Normalize)
    .TilingParse<L2NormalizeCompileInfo>(TilingParseForL2Normalize);

} // namespace optiling
