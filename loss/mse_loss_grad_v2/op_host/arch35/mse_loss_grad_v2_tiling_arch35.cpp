/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mse_loss_grad_v2_tiling_arch35.cpp
 * \brief arch35 host tiling for MseLossGradV2:
 *        ReadInputs -> ReadPlatform -> validation (dtype -> format -> raw rank <= 8 -> attr)
 *        -> empty-output short-circuit (SetBlockDim(1) + zeroed TilingData) -> PadAndSqueeze
 *        -> CheckBroadcastShape -> negative-dim guard -> platform zero guards
 *        -> PrecomputeStrides + cof folding -> effective-rank routing (RANK 4/8)
 *        -> DoTilingAndSet<R> -> SetTilingKey; workspaces[0] = 0 on every SUCCESS path.
 */

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "securec.h"                                            // memset_s, EOK
#include "register/op_def_registry.h"                           // IMPL_OP_OPTILING
#include "op_common/log/log.h"                                  // OP_LOGE/OP_LOGI/OP_CHECK_*
#include "op_common/op_host/util/platform_util.h"               // PlatformAscendC, CoreMemType
#include "../../op_kernel/arch35/mse_loss_grad_v2_tiling_key.h" // MSE_LOSS_GRAD_V2_RANK_4/8, GET_TPL_TILING_KEY
#include "mse_loss_grad_v2_tiling_arch35.h"                     // MseLossGradV2CompileInfo + TilingData struct

namespace optiling {
namespace {

// ---------------------------------------------------------------------------
// Constants (design/TilingData.md §1, design/HostTiling.md §3/§4/§7)
// ---------------------------------------------------------------------------
constexpr int64_t NUM_INPUTS = 3;          // predict/label/dout (OpDef declaration order)
constexpr int64_t NUM_OUTPUTS = 1;         // y
constexpr int64_t UB_ALIGN_MASK = ~31LL;   // 32B down-alignment (ONE_BLK_SIZE = 32)
constexpr int64_t FP32_SIZE = 4;           // FP32 budget regardless of input dtype (HostTiling.md §1.2/§4)
constexpr size_t ATTR_INDEX_REDUCTION = 0; // reduction: OPTIONAL String, default "mean" (Interface.md)

// ---------------------------------------------------------------------------
// Log helpers (工程约束 ⑨: error logs carry concrete values)
// ---------------------------------------------------------------------------
std::string DTypeToString(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16:
            return "float16";
        case ge::DT_FLOAT:
            return "float32";
        case ge::DT_BF16:
            return "bfloat16";
        case ge::DT_INT32:
            return "int32";
        case ge::DT_INT64:
            return "int64";
        case ge::DT_DOUBLE:
            return "float64";
        default:
            return "dtype(" + std::to_string(static_cast<int32_t>(dt)) + ")";
    }
}

std::string FormatToString(ge::Format f)
{
    switch (f) {
        case ge::FORMAT_ND:
            return "ND";
        case ge::FORMAT_FRACTAL_NZ:
            return "FRACTAL_NZ";
        case ge::FORMAT_NCHW:
            return "NCHW";
        case ge::FORMAT_NHWC:
            return "NHWC";
        default:
            return "format(" + std::to_string(static_cast<int32_t>(f)) + ")";
    }
}

// Arr2String — format an int64 array as "[a,b,c,...]" for the 维测日志
// (HostTiling.md §7: print from the struct, padding dims included).
std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) {
            oss << ",";
        }
        oss << arr[i];
    }
    oss << "]";
    return oss.str();
}

// ---------------------------------------------------------------------------
// PadAndSqueeze (HostTiling.md §3(1), verbatim):
//   补 1 (front-pad shapes to max rank) -> 去 1 (drop dims that are 1 in every
//   tensor) -> 归一 (all-scalar case normalizes to (1,)). Returns bool, always
//   true; broadcast compatibility is checked separately (CheckBroadcastShape).
// ---------------------------------------------------------------------------
bool PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                   const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                   std::vector<std::vector<int64_t>>& normalInputShapes,
                   std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    const int64_t numInputs = static_cast<int64_t>(inputShapes.size());
    const int64_t numOutputs = static_cast<int64_t>(outputShapes.size());
    int64_t maxRank = 0;
    for (const auto& s : inputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }
    for (const auto& s : outputShapes) {
        maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
    }

    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(static_cast<size_t>(maxRank - static_cast<int64_t>(s.size())), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> paddedIn(static_cast<size_t>(numInputs));
    std::vector<std::vector<int64_t>> paddedOut(static_cast<size_t>(numOutputs));
    for (int64_t i = 0; i < numInputs; i++) {
        paddedIn[static_cast<size_t>(i)] = pad(inputShapes[static_cast<size_t>(i)]);
    }
    for (int64_t i = 0; i < numOutputs; i++) {
        paddedOut[static_cast<size_t>(i)] = pad(outputShapes[static_cast<size_t>(i)]);
    }

    maximumBroShape.clear();
    normalInputShapes.assign(static_cast<size_t>(numInputs), std::vector<int64_t>());
    normalOutputShapes.assign(static_cast<size_t>(numOutputs), std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            if (paddedIn[static_cast<size_t>(i)][static_cast<size_t>(d)] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedIn[static_cast<size_t>(i)][static_cast<size_t>(d)]);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            if (paddedOut[static_cast<size_t>(i)][static_cast<size_t>(d)] != 1) {
                allOne = false;
            }
            maxDim = std::max(maxDim, paddedOut[static_cast<size_t>(i)][static_cast<size_t>(d)]);
        }
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs; i++) {
                normalInputShapes[static_cast<size_t>(i)].push_back(
                    paddedIn[static_cast<size_t>(i)][static_cast<size_t>(d)]);
            }
            for (int64_t i = 0; i < numOutputs; i++) {
                normalOutputShapes[static_cast<size_t>(i)].push_back(
                    paddedOut[static_cast<size_t>(i)][static_cast<size_t>(d)]);
            }
        }
    }
    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs; i++) {
            normalInputShapes[static_cast<size_t>(i)].push_back(1);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            normalOutputShapes[static_cast<size_t>(i)].push_back(1);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CheckBroadcastShape (HostTiling.md §3(2), verbatim): per dim, every non-1
// size must agree across all inputs and outputs. Empty dims (0) are legal
// here (empty tensors are diverted by the short-circuit upstream, and a 0
// against a non-0 size is a broadcast conflict).
// ---------------------------------------------------------------------------
bool CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                         const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][static_cast<size_t>(d)] != 1) {
                if (ref == -1) {
                    ref = paddedIn[i][static_cast<size_t>(d)];
                } else if (paddedIn[i][static_cast<size_t>(d)] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "CheckBroadcastShape", "input size and ref size",
                        (std::string("dim ") + std::to_string(d) + " input[" + std::to_string(i) + "] size " +
                         std::to_string(paddedIn[i][static_cast<size_t>(d)]) + " and " + std::to_string(ref))
                            .c_str(),
                        "broadcast incompatible: input sizes must be equal or 1");
                    return false;
                }
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][static_cast<size_t>(d)] != 1) {
                if (ref == -1) {
                    ref = paddedOut[i][static_cast<size_t>(d)];
                } else if (paddedOut[i][static_cast<size_t>(d)] != ref) {
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        "CheckBroadcastShape", "output size and ref size",
                        (std::string("dim ") + std::to_string(d) + " output[" + std::to_string(i) + "] size " +
                         std::to_string(paddedOut[i][static_cast<size_t>(d)]) + " and " + std::to_string(ref))
                            .c_str(),
                        "broadcast incompatible: output sizes must be equal or 1");
                    return false;
                }
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// PrecomputeStrides (HostTiling.md §3(3), verbatim): a dim where this tensor
// has size 1 but the broadcast shape is larger is a broadcast axis (GM stride
// 0, NDDMA expands in-flight); every other dim gets the row-major contiguous
// stride (product of this tensor's later dims). Returns ELEMENT counts.
// Note: for the all-scalar normalized (1,) vs maxBro (1,) the single dim is
// NOT a broadcast axis, so its stride is 1 (not 0).
// ---------------------------------------------------------------------------
void PrecomputeStrides(const std::vector<std::vector<int64_t>>& normalShapes, const std::vector<int64_t>& maxBroShape,
                       std::vector<std::vector<int64_t>>& strides)
{
    const int64_t rank = static_cast<int64_t>(maxBroShape.size());
    strides.assign(normalShapes.size(), std::vector<int64_t>(static_cast<size_t>(rank), 0));
    for (size_t i = 0; i < normalShapes.size(); i++) {
        int64_t contiguous = 1; // 尾侧各轴连乘（连续 ND 布局）
        for (int64_t d = rank - 1; d >= 0; d--) {
            const bool isBrcAxis = (normalShapes[i][static_cast<size_t>(d)] == 1) &&
                                   (maxBroShape[static_cast<size_t>(d)] != 1);
            strides[i][static_cast<size_t>(d)] = isBrcAxis ? 0 : contiguous;
            contiguous *= normalShapes[i][static_cast<size_t>(d)];
        }
    }
}

// ---------------------------------------------------------------------------
// FindSplitAxis (HostTiling.md §4 / DESIGN-BRANCH-0/1 §2, verbatim): scan from
// the innermost dim outward accumulating `inner` (product of inner dims); the
// first dim whose d_k * inner exceeds perBufElems becomes the split axis.
// dtypeSize is FIXED at sizeof(float) = 4 (FP32 budget regardless of dtype).
// perBufBytes > 0 is guaranteed by the caller (platform guards).
// ---------------------------------------------------------------------------
bool FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t dtypeSize, int64_t ubPerCore, int64_t physNodes,
                   SplitResult& out)
{
    const int64_t perBufBytes = (ubPerCore / physNodes) & UB_ALIGN_MASK; // UB_ALIGN_MASK = ~31
    const int64_t perBufElems = perBufBytes / dtypeSize;
    const int64_t rank = static_cast<int64_t>(maxBroShape.size());
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maxBroShape[static_cast<size_t>(k)] * inner > perBufElems) {
            out.aI = perBufElems / inner;                                         // 内轴 tile 大小（元素数）
            out.aO = (maxBroShape[static_cast<size_t>(k)] + out.aI - 1) / out.aI; // CeilDiv：外轴 tile 数
            const int64_t rem = maxBroShape[static_cast<size_t>(k)] % out.aI;
            out.aITail = (rem == 0) ? out.aI : rem; // 末块大小
            out.axis = k;
            return true;
        }
        if (k == 0) {
            // 全量装得下：整 tensor 单 tile
            out.axis = 0;
            out.aI = maxBroShape[0];
            out.aO = 1;
            out.aITail = out.aI;
            return true;
        }
        inner *= maxBroShape[static_cast<size_t>(k)];
    }
    return true;
}

// ---------------------------------------------------------------------------
// MultiCoreSplit (HostTiling.md §5 / DESIGN-BRANCH-0/1 §2, verbatim): tiles
// are the unit of core distribution; numCores = min(totalTiles, coreNum)
// (tile-starved shapes open only as many cores as there are tiles); big-little
// balance via tilesMain/coresTail (first coresTail cores take one extra tile).
// maxCores >= 1 and totalTiles >= 1 are guaranteed by the callers.
// ---------------------------------------------------------------------------
bool MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                    MultiCoreResult& out)
{
    const int64_t k = ubSplit.axis;
    int64_t outerProd = 1;
    for (int64_t j = 0; j < k; j++) {
        outerProd *= maxBroShape[static_cast<size_t>(j)]; // UB 外轴全量
    }
    out.totalTiles = outerProd * ubSplit.aO;

    out.numCores = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    out.tilesMain = out.totalTiles / out.numCores;
    out.coresTail = out.totalTiles % out.numCores;
    // coresTail: 前 coresTail 个核各多处理 1 个 tile（tilesMain+1），其余核各处理 tilesMain
    return true;
}

// ===========================================================================
// class MseLossGradV2Tiling — main tiling orchestrator (HostTiling.md §9)
// ===========================================================================
class MseLossGradV2Tiling {
public:
    explicit MseLossGradV2Tiling(gert::TilingContext* ctx) : ctx_(ctx) {}

    ge::graphStatus RunTiling();

private:
    // Read STORAGE shapes (what TilingFunc reads per design/tiling facts),
    // dtypes, formats and the reduction attr into members.
    ge::graphStatus ReadInputs();

    // Platform facts: consume CompileInfo (TilingParse cache, design §8) when
    // present; fall back to the live PlatformInfo + PlatformAscendC otherwise
    // (aclnn path never calls TilingParse).
    ge::graphStatus ReadPlatform();

    // Validation chain (§3(4), fixed order): dtype -> format -> raw rank -> attr.
    ge::graphStatus CheckDtypeSupport();
    ge::graphStatus CheckFormatSupport();
    ge::graphStatus CheckMaxDimensions();
    ge::graphStatus CheckAttrValueRange();

    // Defensive guards: negative dims (-1/-2 unknown markers) must never reach
    // the numel/split formulas; platform zero guards (coreNum / ubSize /
    // perBufBytes) must fire before the divisions in FindSplitAxis.
    ge::graphStatus CheckNegativeDims();
    ge::graphStatus CheckPlatformParams();

    // cof folding (§3(5)): reduction="mean" -> 2/numel(predict) with predict's
    // OWN raw storage shape (OpDef input 0); none/sum -> 2.0; host fp32 fold.
    void FoldCof();

    template <int64_t R>
    ge::graphStatus DoTilingAndSet();

    template <int64_t R, int64_t MaxSlots>
    static void FillSlots(int64_t (&shapes)[MaxSlots][R], int64_t (&strides)[MaxSlots][R],
                          const std::vector<std::vector<int64_t>>& norm,
                          const std::vector<std::vector<int64_t>>& normStrides, int64_t delta);

    // OP_LOGE_FOR_INVALID_* build std::string(entityName) — never pass null.
    const char* NodeName() const
    {
        const ge::char_t* name = ctx_->GetNodeName();
        return (name != nullptr) ? name : "MseLossGradV2";
    }

    static std::vector<int64_t> ShapeToVector(const gert::Shape& s)
    {
        std::vector<int64_t> dims;
        for (size_t d = 0; d < s.GetDimNum(); d++) {
            dims.push_back(s.GetDim(d));
        }
        return dims;
    }

    // --- members (read during ReadInputs / ReadPlatform) ---
    gert::TilingContext* ctx_ = nullptr;
    std::vector<std::vector<int64_t>> raw_input_shapes_;  // predict/label/dout STORAGE shapes
    std::vector<std::vector<int64_t>> raw_output_shapes_; // y STORAGE shape
    std::vector<ge::DataType> in_dtypes_;
    std::vector<ge::Format> in_formats_;
    ge::DataType out_dtype_ = ge::DT_FLOAT;
    ge::Format out_format_ = ge::FORMAT_ND;
    std::string reduction_ = "mean"; // OPTIONAL default "mean"
    uint64_t core_num_ = 0;          // AIV core count
    uint64_t ub_size_ = 0;           // UB bytes per core

    // --- members (computed after normalization) ---
    std::vector<int64_t> max_bro_shape_;                     // maximumBroShape (effective frame)
    std::vector<std::vector<int64_t>> normal_input_shapes_;  // normalized input shapes
    std::vector<std::vector<int64_t>> normal_output_shapes_; // normalized output shapes
    std::vector<std::vector<int64_t>> input_strides_;        // GM strides (broadcast axis = 0)
    std::vector<std::vector<int64_t>> output_strides_;
    int64_t rank_ = 0; // effective rank (1..8)
    float cof_ = 2.0f; // host-folded fp32 coefficient
};

ge::graphStatus MseLossGradV2Tiling::ReadInputs()
{
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        const gert::StorageShape* shp = ctx_->GetInputShape(static_cast<size_t>(i));
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, shp);
        raw_input_shapes_.push_back(ShapeToVector(shp->GetStorageShape()));
        // TOPK-2: dtype/format must come from the compile-time TensorDesc
        // (GetInputDesc), never from GetInputTensor (which may return invalid
        // data and, for the output, would require out-of-range indexing).
        const gert::CompileTimeTensorDesc* inDesc = ctx_->GetInputDesc(static_cast<size_t>(i));
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, inDesc);
        in_dtypes_.push_back(inDesc->GetDataType());
        in_formats_.push_back(inDesc->GetStorageFormat());
    }
    const gert::StorageShape* outShp = ctx_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, outShp);
    raw_output_shapes_.push_back(ShapeToVector(outShp->GetStorageShape()));
    // Output dtype/format come from GetOutputDesc(0) (TOPK-2), not from a
    // layout-dependent GetInputTensor(numInputs + 0) accessor.
    const gert::CompileTimeTensorDesc* outDesc = ctx_->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, outDesc);
    out_dtype_ = outDesc->GetDataType();
    out_format_ = outDesc->GetStorageFormat();
    // reduction attr (OPTIONAL String; typed getter, nullptr -> default "mean",
    // never fail on the unset attr — HostTiling.md §3(5))
    const gert::RuntimeAttrs* attrs = ctx_->GetAttrs();
    const char* reductionPtr = (attrs != nullptr) ? attrs->GetStr(ATTR_INDEX_REDUCTION) : nullptr;
    reduction_ = (reductionPtr != nullptr) ? reductionPtr : "mean";
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::ReadPlatform()
{
    // Platform facts come from the LIVE platform info first (valid in both the
    // graph-compile flow and the direct op_tiling flow), falling back to the
    // TilingParse CompileInfo cache only when the platform query is
    // unavailable. ⛔ Never trust a non-null CompileInfo blindly: the direct
    // op_tiling flow hands back an uninitialized buffer (garbage coreNum /
    // ubSize 0) when TilingParse never ran for this process.
    fe::PlatFormInfos* platformInfo = ctx_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ap = platform_ascendc::PlatformAscendC(platformInfo);
        core_num_ = ap.GetCoreNumAiv();
        ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size_);
        if (core_num_ != 0 && ub_size_ != 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    const MseLossGradV2CompileInfo* compileInfo = ctx_->GetCompileInfo<MseLossGradV2CompileInfo>();
    if (compileInfo != nullptr) {
        core_num_ = compileInfo->coreNum;
        ub_size_ = compileInfo->ubSize;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckDtypeSupport()
{
    // spec dtype_policy: same-combination rows only; supported set {fp16, fp32, bf16};
    // all four dtypes (predict/label/dout/y) must be identical.
    auto supported = [](ge::DataType dt) { return dt == ge::DT_FLOAT16 || dt == ge::DT_FLOAT || dt == ge::DT_BF16; };
    if (!supported(out_dtype_)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            NodeName(), "predict/label/dout/y", DTypeToString(out_dtype_).c_str(),
            "dtype must be one of float16/float32/bfloat16 and identical across predict/label/dout/y");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        const size_t idx = static_cast<size_t>(i);
        if (!supported(in_dtypes_[idx]) || in_dtypes_[idx] != out_dtype_) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                NodeName(), "predict/label/dout/y", DTypeToString(in_dtypes_[idx]).c_str(),
                "dtype must be one of float16/float32/bfloat16 and identical across predict/label/dout/y");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckFormatSupport()
{
    // Interface.md「数据 Format 支持」: every input and output must be FORMAT_ND.
    if (out_format_ != ge::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(NodeName(), "predict/label/dout/y", FormatToString(out_format_).c_str(),
                                               "only FORMAT_ND is supported for every input and output");
        return ge::GRAPH_FAILED;
    }
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        const size_t idx = static_cast<size_t>(i);
        if (in_formats_[idx] != ge::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(NodeName(), "predict/label/dout/y",
                                                   FormatToString(in_formats_[idx]).c_str(),
                                                   "only FORMAT_ND is supported for every input and output");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckMaxDimensions()
{
    // Interface.md「约束与限制」: rank > 8 (any input or the broadcast result
    // exceeding MAX_DIM_LEN) -> shape_mismatch. RAW rank check BEFORE
    // PadAndSqueeze (HostTiling.md §9) so all-1 rank-9 cannot normalize away;
    // also guards delta = R - rank_ against negative array indexing.
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        if (static_cast<int64_t>(raw_input_shapes_[static_cast<size_t>(i)].size()) > MSE_LOSS_GRAD_V2_RANK_8) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                NodeName(), "input",
                ("rank " + std::to_string(raw_input_shapes_[static_cast<size_t>(i)].size()) + " exceeds limit").c_str(),
                "rank must be <= 8");
            return ge::GRAPH_FAILED;
        }
    }
    if (static_cast<int64_t>(raw_output_shapes_[0].size()) > MSE_LOSS_GRAD_V2_RANK_8) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            NodeName(), "output", ("rank " + std::to_string(raw_output_shapes_[0].size()) + " exceeds limit").c_str(),
            "rank must be <= 8");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckAttrValueRange()
{
    // spec attributes.machine_constraint: reduction ∈ {none, mean, sum};
    // OPTIONAL — unset falls back to "mean" in ReadInputs and never fails here.
    if (reduction_ != "none" && reduction_ != "mean" && reduction_ != "sum") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(NodeName(), "reduction", reduction_.c_str(),
                                              "reduction must be one of none/mean/sum");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckNegativeDims()
{
    // -1/-2 unknown-dim markers with all inputs agreeing would pass the
    // broadcast check and corrupt every downstream formula — reject before
    // FindSplitAxis so garbage TilingData is never emitted.
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        for (int64_t d : raw_input_shapes_[static_cast<size_t>(i)]) {
            if (d < 0) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    NodeName(), "predict/label/dout", (std::string("negative dim ") + std::to_string(d)).c_str(),
                    "dim values must be >= 0; unknown dims must be resolved before tiling");
                return ge::GRAPH_FAILED;
            }
        }
    }
    for (int64_t d : raw_output_shapes_[0]) {
        if (d < 0) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                NodeName(), "y", (std::string("negative dim ") + std::to_string(d)).c_str(),
                "dim values must be >= 0; unknown dims must be resolved before tiling");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MseLossGradV2Tiling::CheckPlatformParams()
{
    // §8 zero-value guards. The aclnn/UT path never calls TilingParse, so the
    // guards fire here: numCores = min(totalTiles, 0) would divide by zero in
    // MultiCoreSplit; perBufBytes == 0 (ubSize < 128) would divide by zero in
    // FindSplitAxis (aI = 0/inner then CeilDiv by 0).
    if (core_num_ == 0) {
        OP_LOGE(NodeName(), "coreNum is 0");
        return ge::GRAPH_FAILED;
    }
    if (ub_size_ == 0) {
        OP_LOGE(NodeName(), "ubSize is 0");
        return ge::GRAPH_FAILED;
    }
    const int64_t perBufBytes = (static_cast<int64_t>(ub_size_) / PHYS_NODES) & UB_ALIGN_MASK;
    if (perBufBytes == 0) {
        OP_LOGE(NodeName(), "ubSize(%lu) too small: perBufBytes is 0 after (ubSize / PHYS_NODES) & ~31", ub_size_);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

void MseLossGradV2Tiling::FoldCof()
{
    // §3(5): N = numel(predict)（非 broadcast 后）with predict's OWN raw storage
    // shape (OpDef input 0); host fp32 fold; kernel has no fp division.
    cof_ = 2.0f; // "none"/"sum"
    if (reduction_ == "mean") {
        int64_t numelPredict = 1;
        for (int64_t d : raw_input_shapes_[0]) {
            numelPredict *= d;
        }
        cof_ = 2.0f / static_cast<float>(numelPredict);
    }
}

ge::graphStatus MseLossGradV2Tiling::RunTiling()
{
    // ===== GetShapeInfo: 平台信息 + shape/dtype/attr 读取 (§9) =====
    ge::graphStatus ret = ReadInputs();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = ReadPlatform();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== 异常值校验（§3(4)，顺序固定 dtype → format → 维度上限 → attr）=====
    ret = CheckDtypeSupport();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckFormatSupport();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckMaxDimensions();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttrValueRange();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== 空 tensor 兜底（§3(6)）：output numel == 0 → 清零 TilingData + SetTilingKey
    // + SetBlockDim(1) 直接返回（严禁 SetBlockDim(0)；kernel 读到全零 multicore
    // → GetCoreRange 空区间 → 不下发任何 VEC 指令）=====
    int64_t totalOut = 1;
    for (int64_t d : raw_output_shapes_[0]) {
        totalOut *= d;
    }
    if (totalOut == 0) {
        // 短路发生在 DoTilingAndSet 之前，TilingData 缓冲仍是未初始化脏值、data size
        // 未设置。必须整体清零并把 RANK<=4（tilingKey 0）原型的数据长度钉死，否则
        // kernel Init()/Process() 会读到垃圾 tile 计数并对空张量下发 VEC 指令
        // （设备 VEC_ERROR → NO_OUTPUT）。
        auto* rawTd = ctx_->GetRawTilingData();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, rawTd);
        OP_CHECK_IF(rawTd->GetData() == nullptr || rawTd->GetCapacity() == 0,
                    OP_LOGE(NodeName(), "empty-tensor tiling data buffer unavailable"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(memset_s(rawTd->GetData(), rawTd->GetCapacity(), 0, rawTd->GetCapacity()) != EOK,
                    OP_LOGE(NodeName(), "Memset empty-tensor tilingdata error"), return ge::GRAPH_FAILED);
        rawTd->SetDataSize(sizeof(MseLossGradV2TilingData<MSE_LOSS_GRAD_V2_RANK_4>));
        OP_CHECK_IF(ctx_->SetTilingKey(GET_TPL_TILING_KEY(MSE_LOSS_GRAD_V2_RANK_4)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(NodeName(), "SetTilingKey failed"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(ctx_->SetBlockDim(1) != ge::GRAPH_SUCCESS, OP_LOGE(NodeName(), "SetBlockDim failed"),
                    return ge::GRAPH_FAILED);
        OP_LOGI(NodeName(), "MseLossGradV2 empty tensor, bypass tiling (zeroed TilingData, key=RANK_4)");
        return ge::GRAPH_SUCCESS;
    }

    // ===== PadAndSqueeze（补1 → 去1 → 归一，§3(1)）+ broadcast 校验（§3(2)）=====
    PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_, normal_input_shapes_, normal_output_shapes_);
    rank_ = static_cast<int64_t>(max_bro_shape_.size());
    OP_CHECK_IF(
        !CheckBroadcastShape(normal_input_shapes_, normal_output_shapes_, rank_),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(NodeName(), "input/output", "incompatible",
                                               "check broadcast shape failed, shapes must be broadcast-compatible"),
        return ge::GRAPH_FAILED);

    // ===== negative-dim 守卫（unknown 标记绝不进切分公式）=====
    ret = CheckNegativeDims();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== 平台零值守卫（§8；aclnn 通路不走 TilingParse，此处兜底）=====
    ret = CheckPlatformParams();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    // ===== stride 预计算（§3(3)）+ cof 折叠（§3(5)）=====
    PrecomputeStrides(normal_input_shapes_, max_bro_shape_, input_strides_);
    PrecomputeStrides(normal_output_shapes_, max_bro_shape_, output_strides_);
    FoldCof();

    // ===== 按 effective rank 分叉（§7/§9；TilingKey.md TPL_SEL 两组合）=====
    const int64_t mapped = (rank_ <= MSE_LOSS_GRAD_V2_RANK_4) ? MSE_LOSS_GRAD_V2_RANK_4 : MSE_LOSS_GRAD_V2_RANK_8;
    if (mapped == MSE_LOSS_GRAD_V2_RANK_4) {
        ret = DoTilingAndSet<MSE_LOSS_GRAD_V2_RANK_4>();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        // UINT 枚举列表 [4, 8] 下标 0 → tilingKey 0（sub-kernel _0）
        OP_CHECK_IF(ctx_->SetTilingKey(GET_TPL_TILING_KEY(MSE_LOSS_GRAD_V2_RANK_4)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(NodeName(), "SetTilingKey failed"), return ge::GRAPH_FAILED);
    } else {
        ret = DoTilingAndSet<MSE_LOSS_GRAD_V2_RANK_8>();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        // UINT 枚举列表 [4, 8] 下标 1 → tilingKey 1（sub-kernel _1）
        OP_CHECK_IF(ctx_->SetTilingKey(GET_TPL_TILING_KEY(MSE_LOSS_GRAD_V2_RANK_8)) != ge::GRAPH_SUCCESS,
                    OP_LOGE(NodeName(), "SetTilingKey failed"), return ge::GRAPH_FAILED);
    }
    OP_LOGI(NodeName(), "SetTilingKey: RANK=%ld", mapped);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// FillSlots — shape/stride slot arrays (HostTiling.md §7): used slots carry
// delta front padding (shape=1 / stride=0), then the normalized values;
// unused slots are cleared to shape=1 / stride=0. ⛔ 前补=1，禁止后补 0。
// ---------------------------------------------------------------------------
template <int64_t R, int64_t MaxSlots>
void MseLossGradV2Tiling::FillSlots(int64_t (&shapes)[MaxSlots][R], int64_t (&strides)[MaxSlots][R],
                                    const std::vector<std::vector<int64_t>>& norm,
                                    const std::vector<std::vector<int64_t>>& normStrides, int64_t delta)
{
    const int64_t num = static_cast<int64_t>(norm.size());
    for (int64_t i = 0; i < num; i++) {
        for (int64_t d = 0; d < delta; d++) {
            shapes[i][d] = 1;
            strides[i][d] = 0;
        }
        for (int64_t d = 0; d < static_cast<int64_t>(norm[static_cast<size_t>(i)].size()); d++) {
            shapes[i][d + delta] = norm[static_cast<size_t>(i)][static_cast<size_t>(d)];
            strides[i][d + delta] = normStrides[static_cast<size_t>(i)][static_cast<size_t>(d)];
        }
    }
    for (int64_t i = num; i < MaxSlots; i++) {
        for (int64_t d = 0; d < R; d++) {
            shapes[i][d] = 1;
            strides[i][d] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// DoTilingAndSet<R> — 全字段赋值 + 维测日志 (HostTiling.md §7):
//   memset 清零 -> FindSplitAxis -> MultiCoreSplit -> common fields (rank/
//   perBufBytes/numInputs/numOutputs/cof) -> front-padded maxBroShape ->
//   FillSlots -> 维测日志（从 struct 打印，R 维全量含 padding）-> SetBlockDim.
//   split.axis stays in the NORMALIZED effective-rank frame (NOT shifted by
//   delta; consumers in the R-frame add delta themselves).
// ---------------------------------------------------------------------------
template <int64_t R>
ge::graphStatus MseLossGradV2Tiling::DoTilingAndSet()
{
    auto* td = ctx_->GetTilingData<MseLossGradV2TilingData<R>>();
    OP_CHECK_NULL_WITH_CONTEXT(ctx_, td);
    OP_CHECK_IF(memset_s(td, sizeof(MseLossGradV2TilingData<R>), 0, sizeof(MseLossGradV2TilingData<R>)) != EOK,
                OP_LOGE(NodeName(), "Memset tilingdata error"), return ge::GRAPH_FAILED);
    // —— 切分（§4 / §5）——
    if (!FindSplitAxis(max_bro_shape_, FP32_SIZE, static_cast<int64_t>(ub_size_), PHYS_NODES, td->split)) {
        return ge::GRAPH_FAILED;
    }
    if (!MultiCoreSplit(max_bro_shape_, td->split, static_cast<int64_t>(core_num_), td->multicore)) {
        return ge::GRAPH_FAILED;
    }
    // —— 公共字段（TilingData.md §2 字段表逐项）——
    td->rank = rank_; // 实际有效 rank（1~8）
    td->perBufBytes = (static_cast<int64_t>(ub_size_) / PHYS_NODES) & UB_ALIGN_MASK;
    td->numInputs = NUM_INPUTS;
    td->numOutputs = NUM_OUTPUTS;
    td->cof = cof_;
    const int64_t delta = R - rank_;
    // maxBroShape：前补 delta 个 1（⛔ 前补=1，禁止后补 0）
    for (int64_t d = 0; d < R; d++) {
        td->maxBroShape[d] = (d < delta) ? 1 : max_bro_shape_[static_cast<size_t>(d - delta)];
    }
    // 各输入/输出 shape/stride：前补 delta 个 shape=1/stride=0；未用槽位清 shape=1/stride=0
    FillSlots<R>(td->inputShapes, td->inputStrides, normal_input_shapes_, input_strides_, delta);
    FillSlots<R>(td->outputShapes, td->outputStrides, normal_output_shapes_, output_strides_, delta);
    // —— 维测日志（TilingData 全量，从 struct 打印）——
    OP_LOGI(NodeName(), "MseLossGradV2 tiling: rank=%ld R=%ld cof=%f perBufBytes=%ld dtypeSize=4(FP32)", td->rank, R,
            td->cof, td->perBufBytes);
    OP_LOGI(NodeName(), "  split: axis=%ld aI=%ld aO=%ld aITail=%ld", td->split.axis, td->split.aI, td->split.aO,
            td->split.aITail);
    OP_LOGI(NodeName(), "  multicore: numCores=%ld totalTiles=%ld tilesMain=%ld coresTail=%ld", td->multicore.numCores,
            td->multicore.totalTiles, td->multicore.tilesMain, td->multicore.coresTail);
    OP_LOGI(NodeName(), "  maxBroShape=[%s]", Arr2String(td->maxBroShape, R).c_str());
    for (int64_t i = 0; i < NUM_INPUTS; i++) {
        OP_LOGI(NodeName(), "  input[%ld] shape=[%s] stride=[%s]", i, Arr2String(td->inputShapes[i], R).c_str(),
                Arr2String(td->inputStrides[i], R).c_str());
    }
    for (int64_t i = 0; i < NUM_OUTPUTS; i++) {
        OP_LOGI(NodeName(), "  output[%ld] shape=[%s] stride=[%s]", i, Arr2String(td->outputShapes[i], R).c_str(),
                Arr2String(td->outputStrides[i], R).c_str());
    }
    // —— BlockDim：≤ 物理核数、≥ 1（严禁 0）——
    OP_CHECK_IF(
        ctx_->SetBlockDim(static_cast<uint32_t>(std::max<int64_t>(td->multicore.numCores, 1))) != ge::GRAPH_SUCCESS,
        OP_LOGE(NodeName(), "SetBlockDim failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// TilingPrepareForMseLossGradV2(context) — compile-time preparation
// (HostTiling.md §8): fills MseLossGradV2CompileInfo with platform hardware
// info (AIV core count, UB size) during graph compilation. Zero-value guards
// per the design (== 0 would divide by zero in FindSplitAxis/MultiCoreSplit).
// Not exercised by tests/tiling (pattern B calls the registered TilingFunc
// directly); retained so the registration chain and the package build stay
// intact.
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForMseLossGradV2(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<MseLossGradV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    OP_CHECK_IF(compileInfo->coreNum == 0, OP_LOGE(context->GetNodeName(), "coreNum is 0"), return ge::GRAPH_FAILED);
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    OP_CHECK_IF(compileInfo->ubSize == 0, OP_LOGE(context->GetNodeName(), "ubSize is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// TilingFuncMseLossGradV2(context) — tiling entry point (HostTiling.md §8):
// creates the tiling orchestrator, runs the pipeline, and on every SUCCESS
// path (including the empty-tensor short-circuit) explicitly sets
// workspaces[0] = 0 (Broadcast needs no workspace; ⛔ 必须显式设置).
// ---------------------------------------------------------------------------
static ge::graphStatus TilingFuncMseLossGradV2(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE("MseLossGradV2", "tiling context is nullptr");
        return ge::GRAPH_FAILED;
    }
    MseLossGradV2Tiling tiling(context);
    auto ret = tiling.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    return ge::GRAPH_SUCCESS;
}

// ---------------------------------------------------------------------------
// IMPL_OP_OPTILING(MseLossGradV2) — register tiling functions with CANN.
// One line, three things: Tiling function / TilingParse callback / CompileInfo
// type. Must match the PascalCase operator type name used by OP_ADD in
// op_host/mse_loss_grad_v2_def.cpp and by the UT's kOpType lookup. No
// .TilingInputsDataDependency: no value-dependent inputs (HostTiling.md §2).
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(MseLossGradV2)
    .Tiling(TilingFuncMseLossGradV2)
    .TilingParse<MseLossGradV2CompileInfo>(TilingPrepareForMseLossGradV2);

} // namespace optiling
