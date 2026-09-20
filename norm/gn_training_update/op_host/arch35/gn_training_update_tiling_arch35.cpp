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
// gn_training_update_package/op_host/arch35/gn_training_update_tiling_arch35.cpp
// =============================================================================
//
// ROLE: Host-side tiling for GnTrainingUpdate on arch35 (Ascend 950).
//
//   Real implementation (TDD green phase), transcribed from:
//     design/HostTiling.md            (shared tiling pipeline + validation exits
//                                      §3.3.5/§3.3.6/§3.3.7/§3.3.8/§3.3.10)
//     design/TilingData.md            (TilingData layout §1, field semantics §2/§3)
//     design/TilingKey.md             (RANK_4/RANK_8 tilingKey bit encoding §1:
//                                      RANK_4 -> key 0, RANK_8 -> key 1)
//     design/BranchRoute.md           (branch routing: single criterion
//                                      effective rank; rank<=4 -> key 0)
//     design/branches/DESIGN-BRANCH-0.md §2 (per-branch constants: P=3,
//                                      perBufBytes=(ubSize/3)&~31, perBufElems
//                                      =/4 fixed fp32, FindSplitAxis /
//                                      MultiCoreSplit formulas, empty-batch
//                                      fallback)
//
//   Validation order (HostTiling.md §3.3.5): rank -> dtype -> format -> attr ->
//   layout -> shape (divisibility/dim positivity/overflow/broadcast). Any
//   failure returns GRAPH_FAILED with an OP_LOGE anchor.
//
// OPERATOR NAME VARIANTS:
//   PascalCase   : GnTrainingUpdate   — IMPL_OP_OPTILING, function suffixes
//   snake_case   : gn_training_update  — filename
//   UPPER_SNAKE  : GN_TRAINING_UPDATE  — GET_TPL_TILING_KEY macro prefix
// =============================================================================

#include "register/op_def_registry.h"                                // IMPL_OP_OPTILING
#include "op_common/log/log.h"                                       // OP_LOGE, OP_LOGI, OP_CHECK_NULL_WITH_CONTEXT
#include "op_common/op_host/util/platform_util.h"                    // platform_ascendc::PlatformAscendC
#include "../../op_kernel/arch35/gn_training_update_tiling_struct.h" // GnTrainingUpdateTilingData, SplitResult, MultiCoreResult
#include "../../op_kernel/arch35/gn_training_update_struct.h" // GN_TRAINING_UPDATE_RANK_4/8, GET_TPL_TILING_KEY
#include "gn_training_update_tiling_arch35.h"                 // GnTrainingUpdateCompileInfo

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace optiling {
namespace {

constexpr const char* kOpName = "GnTrainingUpdate";
constexpr int64_t kRankX = 4;     // spec.yaml: x rank fixed [4,4]
constexpr int64_t kRankStats = 5; // spec.yaml: statistics/affine inputs rank fixed [5,5]
constexpr int64_t kFp32Bytes = 4; // perBufElems always counted in fp32 (DESIGN-BRANCH-0 §2)
// IR input slot indexes (op_host/gn_training_update_def.cpp declaration order)
constexpr size_t kIrInScale = 3;    // optional scale
constexpr size_t kIrInOffset = 4;   // optional offset
constexpr int64_t kUbAlignBits = 5; // 32B alignment
constexpr int64_t kUbAlignMask = ~31LL;

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

const char* NodeName(gert::TilingContext* ctx)
{
    const char* name = ctx->GetNodeName();
    return (name != nullptr) ? name : kOpName;
}

// ---------------------------------------------------------------------------
// GnTrainingUpdateTiling — tiling pipeline instance (HostTiling.md §3.3.10)
// ---------------------------------------------------------------------------
class GnTrainingUpdateTiling {
public:
    explicit GnTrainingUpdateTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

    ge::graphStatus RunTiling()
    {
        ge::graphStatus ret = GetShapeInfo();
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        // ===== 异常值校验（任一失败 → GRAPH_FAILED）=====
        if (!CheckMaxDimensions()) { // rank(x)==4；统计量/仿射 rank==5
            return ge::GRAPH_FAILED;
        }
        if (!CheckDtypeSupportAndCombination()) { // x∈{fp16,fp32}，其余输入恒 fp32，组合一致
            return ge::GRAPH_FAILED;
        }
        if (!CheckFormatSupport()) { // 全 FORMAT_ND
            return ge::GRAPH_FAILED;
        }
        if (!CheckAttrValueRange()) { // num_groups >= 1；epsilon > 0
            return ge::GRAPH_FAILED;
        }
        // ===== 算子特有预处理 =====
        if (!DetectLayout()) { // 布局判定（sum 的 G 维位置）
            return ge::GRAPH_FAILED;
        }
        if (!CheckShapeSemantics()) { // C%num_groups==0；C/H/W>0；numel 溢出守卫
            return ge::GRAPH_FAILED;
        }
        GroupViewReshapeAndFold();    // 组维视图 + 连续维折叠；M / invM / hasAffine
        PadAndSqueeze();              // 补 1 → 去 1 → 归一
        if (!CheckBroadcastShape()) { // 广播兼容校验
            return ge::GRAPH_FAILED;
        }
        MergeAxes(); // 合轴（校验通过后）
        rank_ = static_cast<int64_t>(maxBroShape_.size());
        if (rank_ > GN_TRAINING_UPDATE_RANK_8) { // 超出范式能力（防御，不可达）
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(NodeName(ctx_), "x", std::to_string(rank_).c_str(),
                                                  "effective rank exceeds tiling paradigm limit 8");
            return ge::GRAPH_FAILED;
        }
        // ===== 正常 tiling =====
        int64_t numel = 1;
        for (int64_t d : maxBroShape_) {
            numel *= d;
        }
        perBufBytes_ = (static_cast<int64_t>(ubSize_) / PHYS_NODES) & kUbAlignMask;
        // N=0 空 batch 判据必须用 dimN_（x 的真实 N 维）：maxBroShape_ 经
        // PadAndSqueeze 取逐维 max 后，N=0 会被 scale/offset 的 N=1 抬成 1
        // （case00008 实测：maxBroShape_=[1,2,8] numel=16≠0，空 batch 短路
        // 失效导致 kernel 按幻影 shape 越界写 0 元素输出，oob sentinel FAIL）
        if (dimN_ == 0 || numel == 0) {
            // N=0 空 batch（合法退化，HostTiling.md §3.3.7/§3.3.10,
            // Kernel.md §9.3, DESIGN-BRANCH-0 §2）:
            // split/multicore 全 0，host SetBlockDim(1)（严禁 SetBlockDim(0)），
            // kernel 经 multicore.totalTiles==0 零循环短路
            return DoEmptyTilingAndSet();
        }
        ge::graphStatus platRet = ReadPlatformAndCheck();
        if (platRet != ge::GRAPH_SUCCESS) {
            return platRet;
        }
        FindSplitAxis();
        MultiCoreSplit();
        ComputeStrides();
        const int64_t mapped = (rank_ <= GN_TRAINING_UPDATE_RANK_4) ? GN_TRAINING_UPDATE_RANK_4 :
                                                                      GN_TRAINING_UPDATE_RANK_8;
        if (mapped == GN_TRAINING_UPDATE_RANK_4) {
            ret = DoTilingAndSet<GN_TRAINING_UPDATE_RANK_4>();
        } else {
            ret = DoTilingAndSet<GN_TRAINING_UPDATE_RANK_8>();
        }
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(mapped));
        OP_LOGI(NodeName(ctx_), "tilingKey RANK=%ld", mapped);
        return ge::GRAPH_SUCCESS;
    }

private:
    // -----------------------------------------------------------------------
    // GetShapeInfo — read input/output shapes, dtypes, formats and attrs
    // -----------------------------------------------------------------------
    ge::graphStatus GetShapeInfo()
    {
        numIn_ = static_cast<int64_t>(ctx_->GetComputeNodeInputNum());
        if (numIn_ < 3) { // x/sum/square_sum are REQUIRED
            OP_LOGE_FOR_INVALID_TENSORNUM(NodeName(ctx_), "inputs", numIn_, "3");
            return ge::GRAPH_FAILED;
        }
        inDims_.clear();
        inDtypes_.clear();
        inFormats_.clear();
        for (int64_t i = 0; i < numIn_; i++) {
            const gert::StorageShape* shape = ctx_->GetInputShape(static_cast<size_t>(i));
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
            const gert::Shape& s = shape->GetStorageShape();
            std::vector<int64_t> dims;
            for (size_t d = 0; d < s.GetDimNum(); d++) {
                dims.push_back(s.GetDim(d));
            }
            inDims_.push_back(std::move(dims));
            const gert::CompileTimeTensorDesc* desc = ctx_->GetInputDesc(static_cast<size_t>(i));
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, desc);
            inDtypes_.push_back(desc->GetDataType());
            inFormats_.push_back(desc->GetOriginFormat());
        }
        outDims_.clear();
        outDtypes_.clear();
        outFormats_.clear();
        for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            const gert::StorageShape* shape = ctx_->GetOutputShape(static_cast<size_t>(i));
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
            const gert::Shape& s = shape->GetStorageShape();
            std::vector<int64_t> dims;
            for (size_t d = 0; d < s.GetDimNum(); d++) {
                dims.push_back(s.GetDim(d));
            }
            outDims_.push_back(std::move(dims));
            const gert::CompileTimeTensorDesc* desc = ctx_->GetOutputDesc(static_cast<size_t>(i));
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, desc);
            outDtypes_.push_back(desc->GetDataType());
            outFormats_.push_back(desc->GetOriginFormat());
        }
        // attrs in OpDef order: num_groups (int), epsilon (float);
        // defaults per Interface.md OpDef contract (num_groups=2, epsilon=1e-4)
        numGroups_ = 2;
        epsilon_ = 1e-4f;
        const gert::RuntimeAttrs* attrs = ctx_->GetAttrs();
        if (attrs != nullptr) {
            const int64_t* g = attrs->GetInt(0);
            if (g != nullptr) {
                numGroups_ = *g;
            }
            const float* e = attrs->GetFloat(1);
            if (e != nullptr) {
                epsilon_ = *e;
            }
        }
        // Optional-input presence MUST be queried by IR slot: on the aclnn
        // direct-launch path the executor compacts absent optional inputs out
        // of the input list, so inDims_ indexes no longer line up with IR
        // slots 3/4 (smoke L0_001 evidence: scale absent + offset present ->
        // numIn=5, inDims_[3]=offset shape, inDims_[4]=variance shape; the old
        // hasAffine=(numIn>=5) then built DMA plans from the wrong tensors and
        // the kernel crashed with 507035). GetOptionalInputShape returns
        // nullptr for a non-instantiated OPTIONAL_INPUT.
        scaleDims_.clear();
        offsetDims_.clear();
        const gert::StorageShape* scaleShape = ctx_->GetOptionalInputShape(kIrInScale);
        if (scaleShape != nullptr) {
            const gert::Shape& s = scaleShape->GetStorageShape();
            for (size_t d = 0; d < s.GetDimNum(); d++) {
                scaleDims_.push_back(s.GetDim(d));
            }
        }
        const gert::StorageShape* offsetShape = ctx_->GetOptionalInputShape(kIrInOffset);
        if (offsetShape != nullptr) {
            const gert::Shape& s = offsetShape->GetStorageShape();
            for (size_t d = 0; d < s.GetDimNum(); d++) {
                offsetDims_.push_back(s.GetDim(d));
            }
        }
        OP_LOGI(NodeName(ctx_), "optional inputs: hasScale=%d hasOffset=%d", static_cast<int>(!scaleDims_.empty()),
                static_cast<int>(!offsetDims_.empty()));
        return ge::GRAPH_SUCCESS;
    }

    // -----------------------------------------------------------------------
    // 校验（HostTiling.md §3.3.5；顺序 rank -> dtype -> format -> attr）
    // -----------------------------------------------------------------------
    bool CheckMaxDimensions()
    {
        if (static_cast<int64_t>(inDims_[0].size()) != kRankX) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(NodeName(ctx_), "x", std::to_string(inDims_[0].size()).c_str(),
                                                     "x rank must be 4");
            return false;
        }
        for (int64_t i = 1; i < numIn_; i++) {
            if (static_cast<int64_t>(inDims_[i].size()) != kRankStats) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(NodeName(ctx_), "sum/square_sum/...",
                                                         std::to_string(inDims_[i].size()).c_str(),
                                                         "statistics/affine input rank must be 5");
                return false;
            }
        }
        // outputs: y rank 4, batch_mean/batch_variance rank 5 (InferShape 保证，防御性校验)
        if (static_cast<int64_t>(outDims_[0].size()) != kRankX ||
            static_cast<int64_t>(outDims_[1].size()) != kRankStats ||
            static_cast<int64_t>(outDims_[2].size()) != kRankStats) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(NodeName(ctx_), "y/batch_mean/batch_variance", "-",
                                                     "output ranks must be y=4, batch_mean/batch_variance=5");
            return false;
        }
        return true;
    }

    bool CheckDtypeSupportAndCombination()
    {
        const ge::DataType xdt = inDtypes_[0];
        if (!(xdt == ge::DT_FLOAT16 || xdt == ge::DT_FLOAT)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(NodeName(ctx_), "x", std::to_string(static_cast<int>(xdt)).c_str(),
                                                  "x dtype must be float16 or float32");
            return false;
        }
        for (int64_t i = 1; i < numIn_; i++) {
            if (inDtypes_[i] != ge::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(NodeName(ctx_), "sum/square_sum/scale/offset/mean/variance",
                                                      std::to_string(static_cast<int>(inDtypes_[i])).c_str(),
                                                      "statistics/affine input dtype must be float32");
                return false;
            }
        }
        if (outDtypes_[0] != xdt) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(NodeName(ctx_), "y",
                                                  std::to_string(static_cast<int>(outDtypes_[0])).c_str(),
                                                  "y dtype must equal x dtype");
            return false;
        }
        if (outDtypes_[1] != ge::DT_FLOAT || outDtypes_[2] != ge::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(NodeName(ctx_), "batch_mean/batch_variance", "-",
                                                  "statistics output dtype must be float32");
            return false;
        }
        return true;
    }

    bool CheckFormatSupport()
    {
        // 仅拒绝存储物理结构不同的 format（FRACTAL_NZ 等）；ND/NCHW/NHWC 均为
        // 连续排布标签，存储物理相同，均可接受。GE 图模式会把 origin format 默认
        // 打成 NCHW（含 5D 统计量），属正常路径；布局语义只看 x 的标签。
        auto isContiguousFmt = [](ge::Format f) {
            return f == ge::FORMAT_ND || f == ge::FORMAT_NCHW || f == ge::FORMAT_NHWC;
        };
        for (int64_t i = 0; i < numIn_; i++) {
            if (!isContiguousFmt(inFormats_[i])) {
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(NodeName(ctx_), "inputs",
                                                       std::to_string(static_cast<int>(inFormats_[i])).c_str(),
                                                       "only ND/NCHW/NHWC (contiguous) formats are supported");
                return false;
            }
        }
        for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            if (!isContiguousFmt(outFormats_[i])) {
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(NodeName(ctx_), "outputs",
                                                       std::to_string(static_cast<int>(outFormats_[i])).c_str(),
                                                       "only ND/NCHW/NHWC (contiguous) formats are supported");
                return false;
            }
        }
        return true;
    }

    bool CheckAttrValueRange()
    {
        if (numGroups_ < 1) {
            OP_LOGE(NodeName(ctx_), "num_groups check failed: num_groups %ld must be >= 1", numGroups_);
            return false;
        }
        if (!(epsilon_ > 0.0f)) { // also rejects NaN (spec: lower_exclusive 0.0)
            OP_LOGE(NodeName(ctx_), "epsilon check failed: epsilon must be > 0");
            return false;
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // 布局判定（HostTiling.md §3.3.5 step 1）：x 带显式 NCHW/NHWC 标签时以
    // 标签为准（A2 语义），并要求与 sum 的 G 维位置一致（不一致即 A2 的
    // "布局混用"拒绝）；x 为 ND 时由 sum 的 G 维位置推断；G==1 时两种
    // 判定等价，取 NCHW
    // -----------------------------------------------------------------------
    bool DetectLayout()
    {
        const std::vector<int64_t>& sumDims = inDims_[1];
        ge::Format xFmt = inFormats_[0];
        if (xFmt == ge::FORMAT_NCHW || xFmt == ge::FORMAT_NHWC) {
            nchw_ = (xFmt == ge::FORMAT_NCHW);
            size_t gAxis = nchw_ ? 1 : 3;
            if (numGroups_ > 1 && sumDims[gAxis] != numGroups_) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    NodeName(ctx_), "sum", ("G-axis=" + std::to_string(gAxis)).c_str(),
                    "x format tag conflicts with sum shape G-position (num_groups mismatch)");
                return false;
            }
            return true;
        }
        if (sumDims[1] == numGroups_) {
            nchw_ = true;
        } else if (sumDims[3] == numGroups_) {
            nchw_ = false;
        } else {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                NodeName(ctx_), "sum", std::to_string(numGroups_).c_str(),
                "sum shape G-position matches neither NCHW (axis 1) nor NHWC (axis 3) convention for num_groups");
            return false;
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // shape 语义校验：C%num_groups==0（attr 域）、C/H/W>0、numel 溢出守卫
    // -----------------------------------------------------------------------
    bool CheckShapeSemantics()
    {
        const std::vector<int64_t>& x = inDims_[0];
        const int64_t n = x[0];
        if (nchw_) {
            dimC_ = x[1];
            dimH_ = x[2];
            dimW_ = x[3];
        } else {
            dimH_ = x[1];
            dimW_ = x[2];
            dimC_ = x[3];
        }
        dimN_ = n;
        if (dimC_ % numGroups_ != 0) {
            OP_LOGE(NodeName(ctx_), "num_groups check failed: C %% num_groups != 0 (C=%ld, num_groups=%ld)", dimC_,
                    numGroups_);
            return false;
        }
        if (dimC_ <= 0 || dimH_ <= 0 || dimW_ <= 0) { // N=0 合法（空 batch 短路）
            OP_LOGE(NodeName(ctx_), "C/H/W/G check failed: C/H/W must be > 0 (C=%ld, H=%ld, W=%ld)", dimC_, dimH_,
                    dimW_);
            return false;
        }
        // numel 溢出守卫（int64 元素计数域；spec.yaml boundary_conditions）
        const __int128 numel = static_cast<__int128>(n) * dimC_ * dimH_ * dimW_;
        if (numel > std::numeric_limits<int64_t>::max()) {
            OP_LOGE(NodeName(ctx_), "overflow check failed: numel N*C*H*W exceeds int64 max");
            return false;
        }
        const __int128 m = static_cast<__int128>(dimC_ / numGroups_) * dimH_ * dimW_;
        if (m > std::numeric_limits<int64_t>::max()) {
            OP_LOGE(NodeName(ctx_), "overflow check failed: group size M=(C/G)*H*W exceeds int64 max");
            return false;
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // 组维视图 + 连续维折叠（HostTiling.md §3.3.5 step 2）
    //   NCHW: x [N,C,H,W] -> [N,G,M]；统计量 [N,G,1,1,1] -> [N,G,1]；
    //         仿射 [1,G,1,1,1] -> [1,G,1]
    //   NHWC: x [N,H,W,C] -> [N,H*W,G,C/G]；统计量 [N,1,1,G,1] -> [N,1,G,1]；
    //         仿射 [1,1,1,G,1] -> [1,1,G,1]
    //   mean/variance 保留槽位以 {1} 占位（TilingData.md §3）
    // -----------------------------------------------------------------------
    void GroupViewReshapeAndFold()
    {
        // golden semantics (golden.py _golden_impl): affine iff scale present;
        // offset absent -> identity 0; scale absent -> offset ignored.
        hasAffine_ = (!scaleDims_.empty()) ? 1 : 0;
        hasOffset_ = (hasAffine_ == 1 && !offsetDims_.empty()) ? 1 : 0;
        dimM_ = (dimC_ / numGroups_) * dimH_ * dimW_;
        invM_ = 1.0f / static_cast<float>(dimM_);

        foldedIn_.assign(MAX_INPUT_SLOTS, std::vector<int64_t>());
        foldedOut_.assign(MAX_OUTPUT_SLOTS, std::vector<int64_t>());
        auto foldNchw = [](const std::vector<int64_t>& d) {
            return std::vector<int64_t>{d[0], d[1], d[2] * d[3] * d[4]};
        };
        auto foldNhwc = [](const std::vector<int64_t>& d) {
            return std::vector<int64_t>{d[0], d[1] * d[2], d[3], d[4]};
        };
        const std::vector<int64_t>& y = outDims_[0];
        if (nchw_) {
            foldedIn_[0] = {dimN_, numGroups_, dimM_};
            foldedIn_[1] = foldNchw(inDims_[1]);
            foldedIn_[2] = foldNchw(inDims_[2]);
            foldedIn_[3] = (hasAffine_ == 1) ? foldNchw(scaleDims_) : std::vector<int64_t>{1};
            foldedIn_[4] = (hasOffset_ == 1) ? foldNchw(offsetDims_) : std::vector<int64_t>{1};
            foldedOut_[0] = {y[0], numGroups_, (y[1] / numGroups_) * y[2] * y[3]};
            foldedOut_[1] = foldNchw(outDims_[1]);
            foldedOut_[2] = foldNchw(outDims_[2]);
        } else {
            foldedIn_[0] = {dimN_, dimH_ * dimW_, numGroups_, dimC_ / numGroups_};
            foldedIn_[1] = foldNhwc(inDims_[1]);
            foldedIn_[2] = foldNhwc(inDims_[2]);
            foldedIn_[3] = (hasAffine_ == 1) ? foldNhwc(scaleDims_) : std::vector<int64_t>{1};
            foldedIn_[4] = (hasOffset_ == 1) ? foldNhwc(offsetDims_) : std::vector<int64_t>{1};
            foldedOut_[0] = {y[0], y[1] * y[2], numGroups_, y[3] / numGroups_};
            foldedOut_[1] = foldNhwc(outDims_[1]);
            foldedOut_[2] = foldNhwc(outDims_[2]);
        }
        foldedIn_[5] = {1}; // mean: reserved IR slot, placeholder only
        foldedIn_[6] = {1}; // variance: reserved IR slot, placeholder only
    }

    // -----------------------------------------------------------------------
    // PadAndSqueeze（HostTiling.md §3.3.5 附源码）：前补 1 → 去全 1 维 → 归一
    // -----------------------------------------------------------------------
    void PadAndSqueeze()
    {
        int64_t maxRank = 0;
        for (const auto& s : foldedIn_) {
            maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
        }
        for (const auto& s : foldedOut_) {
            maxRank = std::max(maxRank, static_cast<int64_t>(s.size()));
        }
        auto padFront = [maxRank](std::vector<std::vector<int64_t>>& shapes) {
            for (auto& s : shapes) {
                s.insert(s.begin(), maxRank - static_cast<int64_t>(s.size()), 1);
            }
        };
        padFront(foldedIn_);
        padFront(foldedOut_);
        maxBroShape_.clear();
        normalInputShapes_.assign(foldedIn_.size(), std::vector<int64_t>());
        normalOutputShapes_.assign(foldedOut_.size(), std::vector<int64_t>());
        for (int64_t d = 0; d < maxRank; d++) {
            bool allOne = true;
            int64_t maxDim = 0;
            for (const auto& s : foldedIn_) {
                if (s[d] != 1) {
                    allOne = false;
                }
                maxDim = std::max(maxDim, s[d]);
            }
            for (const auto& s : foldedOut_) {
                if (s[d] != 1) {
                    allOne = false;
                }
                maxDim = std::max(maxDim, s[d]);
            }
            if (allOne) {
                continue; // squeeze
            }
            maxBroShape_.push_back(maxDim);
            for (size_t i = 0; i < foldedIn_.size(); i++) {
                normalInputShapes_[i].push_back(foldedIn_[i][d]);
            }
            for (size_t i = 0; i < foldedOut_.size(); i++) {
                normalOutputShapes_[i].push_back(foldedOut_[i][d]);
            }
        }
        if (maxBroShape_.empty()) { // 全标量兜底（本算子不可达，防御保留）
            maxBroShape_.push_back(1);
            for (auto& s : normalInputShapes_) {
                s.push_back(1);
            }
            for (auto& s : normalOutputShapes_) {
                s.push_back(1);
            }
        }
    }

    // -----------------------------------------------------------------------
    // CheckBroadcastShape（HostTiling.md §3.3.5 附源码）：
    // 逐维所有非 1 尺寸必须一致（输入/输出组共享 ref）
    // -----------------------------------------------------------------------
    bool CheckBroadcastShape()
    {
        const int64_t rank = static_cast<int64_t>(maxBroShape_.size());
        for (int64_t d = 0; d < rank; d++) {
            int64_t ref = -1;
            for (const auto& s : normalInputShapes_) {
                if (s[d] == 1) {
                    continue;
                }
                if (ref == -1) {
                    ref = s[d];
                } else if (s[d] != ref) {
                    OP_LOGE(NodeName(ctx_),
                            "broadcast check failed: dim %ld input sizes incompatible (%ld vs %ld), "
                            "sizes must be equal or 1",
                            d, s[d], ref);
                    return false;
                }
            }
            for (const auto& s : normalOutputShapes_) {
                if (s[d] == 1) {
                    continue;
                }
                if (ref == -1) {
                    ref = s[d];
                } else if (s[d] != ref) {
                    OP_LOGE(NodeName(ctx_),
                            "broadcast check failed: dim %ld output sizes incompatible (%ld vs %ld), "
                            "sizes must be equal or 1",
                            d, s[d], ref);
                    return false;
                }
            }
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // MergeAxes（HostTiling.md §3.3.5 step 3；Broadcast 范式贪心合轴，
    // 从末维向前：[d..hi] 可合并 iff 每个 tensor 在该区间的尺寸积为 1 或
    // 等于 maxBroShape 区间积）
    // -----------------------------------------------------------------------
    void MergeAxes()
    {
        const int64_t oldRank = static_cast<int64_t>(maxBroShape_.size());
        if (oldRank <= 1) {
            return;
        }
        std::vector<std::vector<int64_t>> groups;
        int64_t lo = oldRank - 1;
        int64_t hi = oldRank - 1;
        for (int64_t d = oldRank - 2; d >= 0; d--) {
            int64_t maxProd = 1;
            for (int64_t k = d; k <= hi; k++) {
                maxProd *= maxBroShape_[k];
            }
            bool mergeable = true;
            for (const auto& s : normalInputShapes_) {
                int64_t prod = 1;
                for (int64_t k = d; k <= hi; k++) {
                    prod *= s[k];
                }
                if (prod != 1 && prod != maxProd) {
                    mergeable = false;
                    break;
                }
            }
            if (mergeable) {
                for (const auto& s : normalOutputShapes_) {
                    int64_t prod = 1;
                    for (int64_t k = d; k <= hi; k++) {
                        prod *= s[k];
                    }
                    if (prod != 1 && prod != maxProd) {
                        mergeable = false;
                        break;
                    }
                }
            }
            if (mergeable) {
                lo = d;
            } else {
                groups.push_back({lo, hi});
                lo = d;
                hi = d;
            }
        }
        groups.push_back({lo, hi});
        std::reverse(groups.begin(), groups.end());
        auto mergeByGroups = [&groups](const std::vector<int64_t>& s) {
            std::vector<int64_t> merged;
            for (const auto& g : groups) {
                int64_t prod = 1;
                for (int64_t k = g[0]; k <= g[1]; k++) {
                    prod *= s[k];
                }
                merged.push_back(prod);
            }
            return merged;
        };
        maxBroShape_ = mergeByGroups(maxBroShape_);
        for (auto& s : normalInputShapes_) {
            s = mergeByGroups(s);
        }
        for (auto& s : normalOutputShapes_) {
            s = mergeByGroups(s);
        }
    }

    // -----------------------------------------------------------------------
    // 平台信息（运行期查询，禁止硬编码）+ 零值守卫（HostTiling.md §3.3.9）
    // -----------------------------------------------------------------------
    ge::graphStatus ReadPlatformAndCheck()
    {
        fe::PlatFormInfos* platformInfo = ctx_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, platformInfo);
        auto ap = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ap.GetCoreNumAiv();
        if (coreNum_ < 1) {
            OP_LOGE(NodeName(ctx_), "coreNum is 0 (coreNum %lu must be >= 1)", static_cast<unsigned long>(coreNum_));
            return ge::GRAPH_FAILED;
        }
        ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        perBufBytes_ = (static_cast<int64_t>(ubSize_) / PHYS_NODES) & kUbAlignMask;
        if (perBufBytes_ < kFp32Bytes) {
            // perBufElems < 1：UB 低于一个最小 32B block，任何 tile 都装不下
            OP_LOGE(NodeName(ctx_), "ubSize check failed: ubSize %lu too small, perBufBytes %ld < 4",
                    static_cast<unsigned long>(ubSize_), perBufBytes_);
            return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    // -----------------------------------------------------------------------
    // FindSplitAxis（HostTiling.md §3.3.6 / DESIGN-BRANCH-0 §2）：
    // 以 maxBroShape 为坐标系从内轴向外扫描，取首个 maxBro[k]*inner >
    // perBufElems 的轴；dtypeSize 固定按 fp32=4B
    // -----------------------------------------------------------------------
    void FindSplitAxis()
    {
        const int64_t perBufElems = perBufBytes_ / kFp32Bytes;
        const int64_t rank = static_cast<int64_t>(maxBroShape_.size());
        // UB 最内维占用按 32B 行距粒度上取整计入预算（kernel 侧 PadV：
        // fp32=8 元素 / fp16=16 元素；NDDMA dim0 粒度 + MTE3 UB 地址 32B
        // 对齐，errcode 80 实测）：tile 的 UB 占用 =
        // aISeg × Π（最内维按粒度上取整，其余真实），必须 ≤ perBufElems
        const int64_t padUnit = (inDtypes_[0] == ge::DT_FLOAT16) ? (32 / 2) : (32 / 4);
        auto padV = [padUnit](int64_t x) { return (x + padUnit - 1) / padUnit * padUnit; };
        int64_t inner = 1;
        for (int64_t k = rank - 1; k >= 0; k--) {
            const int64_t dim = (k == rank - 1) ? padV(maxBroShape_[k]) : maxBroShape_[k];
            if (dim * inner > perBufElems) {
                split_.aI = perBufElems / inner;
                if (k == rank - 1) {
                    // 切分轴即最内维：aI 按粒度对齐，尾块 PadV 后不越预算
                    split_.aI = (split_.aI / padUnit) * padUnit;
                }
                if (split_.aI < 1) {
                    split_.aI = 1;
                }
                split_.aO = CeilDiv(maxBroShape_[k], split_.aI);
                const int64_t rem = maxBroShape_[k] % split_.aI;
                split_.aITail = (rem == 0) ? split_.aI : rem;
                split_.axis = k;
                return;
            }
            if (k == 0) {
                // 全量装得下（GN 典型小 shape：M <= perBufElems）
                split_.axis = 0;
                split_.aI = maxBroShape_[0];
                split_.aO = 1;
                split_.aITail = split_.aI;
                return;
            }
            inner *= dim;
        }
    }

    // -----------------------------------------------------------------------
    // MultiCoreSplit（HostTiling.md §3.3.7 / DESIGN-BRANCH-0 §2）：
    // totalTiles = aO × Π_{j<axis} maxBroShape[j]；一维均衡，前 coresTail
    // 个核各多处理 1 个 tile
    // -----------------------------------------------------------------------
    void MultiCoreSplit()
    {
        int64_t outerProd = 1;
        for (int64_t j = 0; j < split_.axis; j++) {
            outerProd *= maxBroShape_[j];
        }
        multicore_.totalTiles = outerProd * split_.aO;
        if (multicore_.totalTiles == 0) {
            multicore_.numCores = 0;
            multicore_.tilesMain = 0;
            multicore_.coresTail = 0;
            return;
        }
        multicore_.numCores = (multicore_.totalTiles < static_cast<int64_t>(coreNum_)) ? multicore_.totalTiles :
                                                                                         static_cast<int64_t>(coreNum_);
        multicore_.tilesMain = multicore_.totalTiles / multicore_.numCores;
        multicore_.coresTail = multicore_.totalTiles % multicore_.numCores;
    }

    // -----------------------------------------------------------------------
    // (S4) strides：按各 tensor 自身合轴后 shape 行主序；size-1（broadcast）
    // 维 stride 置 0（NDDMA 随路展开，TilingData.md §3）
    // -----------------------------------------------------------------------
    static std::vector<int64_t> CalcStrides(const std::vector<int64_t>& s)
    {
        std::vector<int64_t> strides(s.size(), 0);
        for (int64_t d = static_cast<int64_t>(s.size()) - 1; d >= 0; d--) {
            if (s[d] == 1) {
                strides[d] = 0;
                continue;
            }
            int64_t prod = 1;
            for (size_t j = d + 1; j < s.size(); j++) {
                prod *= s[j];
            }
            strides[d] = prod;
        }
        return strides;
    }

    void ComputeStrides()
    {
        inStrides_.clear();
        outStrides_.clear();
        for (const auto& s : normalInputShapes_) {
            inStrides_.push_back(CalcStrides(s));
        }
        for (const auto& s : normalOutputShapes_) {
            outStrides_.push_back(CalcStrides(s));
        }
    }

    // -----------------------------------------------------------------------
    // DoTilingAndSet<RANK> — 填 TilingData 全字段（HostTiling.md §3.3.8，
    // 前补 1 对齐到 RANK 维：padding 维 shape=1、stride=0；mean/variance
    // 槽 5/6 行全 0）+ SetBlockDim（严禁 SetBlockDim(0)）
    // -----------------------------------------------------------------------
    template <int64_t RANK>
    ge::graphStatus DoTilingAndSet()
    {
        auto* td = ctx_->GetTilingData<GnTrainingUpdateTilingData<RANK>>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, td);
        const int64_t delta = RANK - rank_;

        td->split = split_;
        td->split.axis += delta; // axis 以补 1 后的 RANK 维坐标系记录
        td->multicore = multicore_;
        td->rank = rank_;
        td->perBufBytes = perBufBytes_;
        td->numInputs = MAX_INPUT_SLOTS;
        td->numOutputs = MAX_OUTPUT_SLOTS;
        td->hasAffine = hasAffine_;
        td->hasOffset = hasOffset_;
        td->epsilon = epsilon_;
        td->invM = invM_;

        for (int64_t d = 0; d < delta; d++) {
            td->maxBroShape[d] = 1;
        }
        for (int64_t d = 0; d < rank_; d++) {
            td->maxBroShape[d + delta] = maxBroShape_[d];
        }
        for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
            for (int64_t d = 0; d < RANK; d++) {
                if (i >= 5) {
                    // mean/variance 保留槽位：占位行全 0，kernel 不消费
                    td->inputShapes[i][d] = 0;
                    td->inputStrides[i][d] = 0;
                    continue;
                }
                if (d < delta) {
                    td->inputShapes[i][d] = 1;
                    td->inputStrides[i][d] = 0;
                } else {
                    td->inputShapes[i][d] = normalInputShapes_[i][d - delta];
                    td->inputStrides[i][d] = inStrides_[i][d - delta];
                }
            }
        }
        for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
            for (int64_t d = 0; d < RANK; d++) {
                if (d < delta) {
                    td->outputShapes[i][d] = 1;
                    td->outputStrides[i][d] = 0;
                } else {
                    td->outputShapes[i][d] = normalOutputShapes_[i][d - delta];
                    td->outputStrides[i][d] = outStrides_[i][d - delta];
                }
            }
        }
        LogTilingData<RANK>(td);
        // SetBlockDim：严禁 SetBlockDim(0)，空 batch（numCores=0）也设 1，
        // kernel 经 multicore.totalTiles==0 零循环短路
        ctx_->SetBlockDim(static_cast<uint32_t>(std::max<int64_t>(multicore_.numCores, 1)));
        return ge::GRAPH_SUCCESS;
    }

    // -----------------------------------------------------------------------
    // DoEmptyTilingAndSet — N=0 空 batch 短路（DESIGN-BRANCH-0 §2：
    // split={0,0,0,0}、multicore={0,0,0,0}、SetBlockDim(1)），仍按 rank_
    // 分发 RANK 档并设置 tilingKey
    // -----------------------------------------------------------------------
    ge::graphStatus DoEmptyTilingAndSet()
    {
        split_ = {0, 0, 0, 0};
        multicore_ = {0, 0, 0, 0};
        ComputeStrides();
        const int64_t mapped = (rank_ <= GN_TRAINING_UPDATE_RANK_4) ? GN_TRAINING_UPDATE_RANK_4 :
                                                                      GN_TRAINING_UPDATE_RANK_8;
        ge::graphStatus ret;
        if (mapped == GN_TRAINING_UPDATE_RANK_4) {
            ret = DoTilingAndSet<GN_TRAINING_UPDATE_RANK_4>();
        } else {
            ret = DoTilingAndSet<GN_TRAINING_UPDATE_RANK_8>();
        }
        if (ret != ge::GRAPH_SUCCESS) {
            return ret;
        }
        ctx_->SetTilingKey(GET_TPL_TILING_KEY(mapped));
        return ge::GRAPH_SUCCESS;
    }

    template <int64_t RANK>
    void LogTilingData(const GnTrainingUpdateTilingData<RANK>* td) const
    {
        OP_LOGI(NodeName(ctx_), "[gn_training_update] split: axis=%ld, aI=%ld, aO=%ld, tail=%ld", td->split.axis,
                td->split.aI, td->split.aO, td->split.aITail);
        OP_LOGI(NodeName(ctx_), "[gn_training_update] multicore: cores=%ld, tiles=%ld, main=%ld, tail=%ld",
                td->multicore.numCores, td->multicore.totalTiles, td->multicore.tilesMain, td->multicore.coresTail);
        OP_LOGI(NodeName(ctx_), "[gn_training_update] rank=%ld, perBufBytes=%ld, hasAffine=%ld, epsilon=%f, invM=%f",
                td->rank, td->perBufBytes, td->hasAffine, td->epsilon, td->invM);
    }

    gert::TilingContext* ctx_;
    int64_t numIn_ = 0;
    std::vector<std::vector<int64_t>> inDims_;
    std::vector<ge::DataType> inDtypes_;
    std::vector<ge::Format> inFormats_;
    std::vector<std::vector<int64_t>> outDims_;
    std::vector<ge::DataType> outDtypes_;
    std::vector<ge::Format> outFormats_;
    int64_t numGroups_ = 2;
    float epsilon_ = 1e-4f;
    bool nchw_ = true;
    int64_t dimN_ = 0;
    int64_t dimC_ = 0;
    int64_t dimH_ = 0;
    int64_t dimW_ = 0;
    int64_t dimM_ = 0;
    float invM_ = 0.0f;
    int64_t hasAffine_ = 0;
    int64_t hasOffset_ = 0;
    std::vector<int64_t> scaleDims_;  // IR slot 3 shape; empty = scale absent
    std::vector<int64_t> offsetDims_; // IR slot 4 shape; empty = offset absent
    std::vector<std::vector<int64_t>> foldedIn_;
    std::vector<std::vector<int64_t>> foldedOut_;
    std::vector<int64_t> maxBroShape_;
    std::vector<std::vector<int64_t>> normalInputShapes_;
    std::vector<std::vector<int64_t>> normalOutputShapes_;
    std::vector<std::vector<int64_t>> inStrides_;
    std::vector<std::vector<int64_t>> outStrides_;
    int64_t rank_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t perBufBytes_ = 0;
    SplitResult split_{};
    MultiCoreResult multicore_{};
};

// ---------------------------------------------------------------------------
// TilingFuncGnTrainingUpdate(context) — tiling entry point called by CANN
// (HostTiling.md §3.3.10 入口函数；Broadcast 不需要 workspace，显式置 0)
// ---------------------------------------------------------------------------
ge::graphStatus TilingFuncGnTrainingUpdate(gert::TilingContext* context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "context is nullptr");
        return ge::GRAPH_FAILED;
    }
    gert::TilingData* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    if (rawTilingData->GetData() == nullptr) {
        // Host-side UT harness contract (tiling-ut-harness harness-layout.md):
        // the harness zero-fills the whole TilingData buffer (header included)
        // before invoking, wiping capacity_/data_. Rebuild the header the way
        // TilingData::CreateCap laid it out (data area immediately after the
        // header, capacity = the RANK_8 instance size). In the real runtime
        // the framework always delivers a valid data pointer, so this repair
        // only ever triggers under the UT harness.
        rawTilingData->Init(sizeof(GnTrainingUpdateTilingData<GN_TRAINING_UPDATE_RANK_8>),
                            reinterpret_cast<uint8_t*>(rawTilingData) + sizeof(gert::TilingData));
    }
    GnTrainingUpdateTiling tiling(context);
    ge::graphStatus ret = tiling.RunTiling();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    size_t* workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        OP_LOGE(kOpName, "workspace sizes is nullptr");
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = 0; // Broadcast 无 workspace，显式置 0
    return ge::GRAPH_SUCCESS;
}

} // namespace

// ---------------------------------------------------------------------------
// TilingPrepareForGnTrainingUpdate(context) — compile-time preparation
//
// Populates GnTrainingUpdateCompileInfo with platform hardware info
// (core count, UB size), with zero-value guards (design/HostTiling.md
// §3.3.9: coreNum/ubSize of 0 would cause downstream division by zero).
// ---------------------------------------------------------------------------
ge::graphStatus TilingPrepareForGnTrainingUpdate(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<GnTrainingUpdateCompileInfo>();
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
// IMPL_OP_OPTILING(GNTrainingUpdate) — 官方 IR 图类型名注册。
// ---------------------------------------------------------------------------
IMPL_OP_OPTILING(GNTrainingUpdate)
    .Tiling(TilingFuncGnTrainingUpdate)
    .TilingParse<GnTrainingUpdateCompileInfo>(TilingPrepareForGnTrainingUpdate);
IMPL_OP_OPTILING(GnTrainingUpdate)
    .Tiling(TilingFuncGnTrainingUpdate)
    .TilingParse<GnTrainingUpdateCompileInfo>(TilingPrepareForGnTrainingUpdate);

} // namespace optiling
