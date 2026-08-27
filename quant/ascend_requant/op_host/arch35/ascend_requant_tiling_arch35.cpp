/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#include "graph/types.h"

#include "ascend_requant_tiling_arch35.h"

#ifndef ASCEND_REQUANT_TILING_UT
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/ascend_requant_struct.h"
#include <sstream>
#endif

namespace optiling {

namespace ascend_requant {
namespace public_api {

int64_t PadAndSqueeze(const std::vector<std::vector<int64_t>>& inputShapes,
                      const std::vector<std::vector<int64_t>>& outputShapes, std::vector<int64_t>& maximumBroShape,
                      std::vector<std::vector<int64_t>>& normalInputShapes,
                      std::vector<std::vector<int64_t>>& normalOutputShapes)
{
    int64_t numInputs = (int64_t)inputShapes.size();
    int64_t numOutputs = (int64_t)outputShapes.size();

    int64_t maxRank = 0;
    for (auto& s : inputShapes)
        maxRank = std::max(maxRank, (int64_t)s.size());
    for (auto& s : outputShapes)
        maxRank = std::max(maxRank, (int64_t)s.size());

    auto pad = [&](const std::vector<int64_t>& s) {
        std::vector<int64_t> p;
        p.assign(maxRank - (int64_t)s.size(), 1);
        p.insert(p.end(), s.begin(), s.end());
        return p;
    };
    std::vector<std::vector<int64_t>> paddedIn(numInputs), paddedOut(numOutputs);
    for (int64_t i = 0; i < numInputs; i++)
        paddedIn[i] = pad(inputShapes[i]);
    for (int64_t i = 0; i < numOutputs; i++)
        paddedOut[i] = pad(outputShapes[i]);

    int64_t bcastRet = CheckBroadcastShape(paddedIn, paddedOut, maxRank);
    if (bcastRet != kOk) {
        return kErrShapeMismatch;
    }

    maximumBroShape.clear();
    normalInputShapes.assign(numInputs, std::vector<int64_t>());
    normalOutputShapes.assign(numOutputs, std::vector<int64_t>());
    for (int64_t d = 0; d < maxRank; d++) {
        bool allOne = true;
        int64_t maxDim = 0;
        for (int64_t i = 0; i < numInputs; i++) {
            if (paddedIn[i][d] != 1)
                allOne = false;
            maxDim = std::max(maxDim, paddedIn[i][d]);
        }
        for (int64_t i = 0; i < numOutputs; i++) {
            if (paddedOut[i][d] != 1)
                allOne = false;
            maxDim = std::max(maxDim, paddedOut[i][d]);
        }
        for (int64_t i = 0; i < numInputs; i++)
            if (paddedIn[i][d] == 0)
                maxDim = 0;
        for (int64_t i = 0; i < numOutputs; i++)
            if (paddedOut[i][d] == 0)
                maxDim = 0;
        if (!allOne) {
            maximumBroShape.push_back(maxDim);
            for (int64_t i = 0; i < numInputs; i++)
                normalInputShapes[i].push_back(paddedIn[i][d]);
            for (int64_t i = 0; i < numOutputs; i++)
                normalOutputShapes[i].push_back(paddedOut[i][d]);
        }
    }

    if (maximumBroShape.empty()) {
        maximumBroShape.push_back(1);
        for (int64_t i = 0; i < numInputs; i++)
            normalInputShapes[i].push_back(1);
        for (int64_t i = 0; i < numOutputs; i++)
            normalOutputShapes[i].push_back(1);
    }
    return kOk;
}

int64_t CheckBroadcastShape(const std::vector<std::vector<int64_t>>& paddedIn,
                            const std::vector<std::vector<int64_t>>& paddedOut, int64_t maxRank)
{
    for (int64_t d = 0; d < maxRank; d++) {
        int64_t ref = -1;
        for (size_t i = 0; i < paddedIn.size(); i++) {
            if (paddedIn[i][d] != 1) {
                if (ref == -1)
                    ref = paddedIn[i][d];
                else if (paddedIn[i][d] != ref)
                    return kErrShapeMismatch;
            }
        }
        for (size_t i = 0; i < paddedOut.size(); i++) {
            if (paddedOut[i][d] != 1) {
                if (ref == -1)
                    ref = paddedOut[i][d];
                else if (paddedOut[i][d] != ref)
                    return kErrShapeMismatch;
            }
        }
    }
    return kOk;
}

int64_t FindSplitAxis(const std::vector<int64_t>& maxBroShape, int64_t maxDtypeSize, int64_t ubPerCore,
                      int64_t physNodes, SplitResult& out)
{
    for (int64_t d = 0; d < (int64_t)maxBroShape.size(); d++) {
        if (maxBroShape[d] == 0) {
            out.axis = 0;
            out.aI = 0;
            out.aO = 0;
            out.aITail = 0;
            return kOk;
        }
    }

    constexpr int64_t UB_ALIGN_MASK = ~31LL;
    int64_t perBufBytes = (ubPerCore / physNodes) & UB_ALIGN_MASK;
    int64_t perBufElems = perBufBytes / maxDtypeSize;
    int64_t rank = (int64_t)maxBroShape.size();
    int64_t inner = 1;
    for (int64_t k = rank - 1; k >= 0; k--) {
        if (maxBroShape[k] * inner > perBufElems) {
            out.aI = perBufElems / inner;
            if (out.aI < 1)
                out.aI = 1;
            out.aO = (maxBroShape[k] + out.aI - 1) / out.aI;
            int64_t rem = maxBroShape[k] % out.aI;
            out.aITail = (rem == 0) ? out.aI : rem;
            out.axis = k;
            return kOk;
        }
        if (k == 0) {
            out.axis = 0;
            out.aI = maxBroShape[0];
            out.aO = 1;
            out.aITail = maxBroShape[0];
            return kOk;
        }
        inner *= maxBroShape[k];
    }
    return kOk;
}

int64_t MultiCoreSplit(const std::vector<int64_t>& maxBroShape, const SplitResult& ubSplit, int64_t maxCores,
                       MultiCoreResult& out)
{
    int64_t k = ubSplit.axis, outerProd = 1;
    for (int64_t j = 0; j < k; j++)
        outerProd *= maxBroShape[j];
    out.totalTiles = outerProd * ubSplit.aO;

    if (out.totalTiles == 0) {
        out.numCores = 0;
        out.tilesMain = 0;
        out.coresTail = 0;
        return kOk;
    }

    out.numCores = (out.totalTiles < maxCores) ? out.totalTiles : maxCores;
    if (out.numCores < 1)
        out.numCores = 1;
    out.tilesMain = out.totalTiles / out.numCores;
    out.coresTail = out.totalTiles % out.numCores;
    return kOk;
}

int64_t MapRankToTemplate(int64_t rank) { return (rank <= 4) ? 4 : 8; }

int64_t ValidateDtype(ge::DataType xDtype, ge::DataType scaleDtype)
{
    if (xDtype == ge::DT_INT32 && scaleDtype == ge::DT_UINT64)
        return kOk;
    return kErrDtypeNotSupported;
}

int64_t ValidateFormat(ge::Format xFmt, ge::Format scaleFmt, ge::Format yFmt)
{
    if (xFmt == ge::FORMAT_ND && scaleFmt == ge::FORMAT_ND && yFmt == ge::FORMAT_ND)
        return kOk;
    return kErrFormatNotSupported;
}

int64_t ValidateDimensions(int64_t xRank, int64_t scaleRank)
{
    if (xRank < 1 || xRank > 8)
        return kErrDimOutOfRange;
    if (scaleRank < 0 || scaleRank > 8)
        return kErrDimOutOfRange;
    if (scaleRank > xRank)
        return kErrRankExceedsX;
    return kOk;
}

int64_t ValidateAttr(bool /*reluFlag*/) { return kOk; }

} // namespace public_api
} // namespace ascend_requant

namespace ascend_requant {
namespace branch_api {

template <int64_t RANK>
int64_t ComputeBranchTiling(const BranchInputs<RANK>& in, AscendRequantTilingData<RANK>& out)
{
    std::memset(&out, 0, sizeof(out));

    constexpr int64_t MAX_DTYPE_SIZE = 8;
    constexpr int64_t UB_ALIGN_MASK = ~31LL;

    int64_t rank = (int64_t)in.maxBroShape.size();
    int64_t delta = RANK - rank;

    int64_t perBufBytes = (in.ubPerCore / PHYS_NODES) & UB_ALIGN_MASK;
    int64_t perBufElems = perBufBytes / MAX_DTYPE_SIZE;

    if constexpr (RANK > 4) {
    }

    SplitResult split{};
    {
        int64_t inner = 1;
        for (int64_t k = rank - 1; k >= 0; k--) {
            if (in.maxBroShape[k] * inner > perBufElems) {
                split.aI = perBufElems / inner;
                if (split.aI < 1)
                    split.aI = 1;
                split.aO = (in.maxBroShape[k] + split.aI - 1) / split.aI;
                int64_t rem = in.maxBroShape[k] % split.aI;
                split.aITail = (rem == 0) ? split.aI : rem;
                split.axis = k;
                break;
            }
            if (k == 0) {
                split.axis = 0;
                split.aI = in.maxBroShape[0];
                split.aO = 1;
                split.aITail = in.maxBroShape[0];
                break;
            }
            inner *= in.maxBroShape[k];
        }
    }

    MultiCoreResult mc{};
    {
        int64_t outerProd = 1;
        for (int64_t j = 0; j < split.axis; j++)
            outerProd *= in.maxBroShape[j];
        mc.totalTiles = outerProd * split.aO;
        if (mc.totalTiles == 0) {
            mc.numCores = 0;
            mc.tilesMain = 0;
            mc.coresTail = 0;
        } else {
            mc.numCores = (mc.totalTiles < in.maxCores) ? mc.totalTiles : in.maxCores;
            if (mc.numCores < 1)
                mc.numCores = 1;
            mc.tilesMain = mc.totalTiles / mc.numCores;
            mc.coresTail = mc.totalTiles % mc.numCores;
        }
    }

    auto computeStrides = [](const std::vector<int64_t>& s) {
        int64_t r = (int64_t)s.size();
        std::vector<int64_t> st(r, 0);
        for (int64_t d = r - 1; d >= 0; d--) {
            if (s[d] == 1) {
                st[d] = 0;
                continue;
            }
            int64_t prod = 1;
            for (int64_t j = d + 1; j < r; j++)
                prod *= s[j];
            st[d] = prod;
        }
        return st;
    };

    out.split.axis = split.axis + delta;
    out.split.aI = split.aI;
    out.split.aO = split.aO;
    out.split.aITail = split.aITail;
    out.multicore = mc;
    out.rank = rank;
    out.perBufBytes = perBufBytes;
    out.numInputs = (int64_t)in.normalInputShapes.size();
    out.numOutputs = (int64_t)in.normalOutputShapes.size();
    out.reluFlag = in.reluFlag;

    for (int64_t d = 0; d < delta; d++)
        out.maxBroShape[d] = 1;
    for (int64_t d = 0; d < rank; d++)
        out.maxBroShape[d + delta] = in.maxBroShape[d];

    for (int64_t i = 0; i < MAX_INPUT_SLOTS; i++) {
        for (int64_t d = 0; d < RANK; d++) {
            out.inputShapes[i][d] = 1;
            out.inputStrides[i][d] = 0;
        }
    }
    for (int64_t i = 0; i < (int64_t)in.normalInputShapes.size(); i++) {
        auto st = computeStrides(in.normalInputShapes[i]);
        for (int64_t d = 0; d < delta; d++) {
            out.inputShapes[i][d] = 1;
            out.inputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.inputShapes[i][d + delta] = in.normalInputShapes[i][d];
            out.inputStrides[i][d + delta] = st[d];
        }
    }

    for (int64_t i = 0; i < MAX_OUTPUT_SLOTS; i++) {
        for (int64_t d = 0; d < RANK; d++) {
            out.outputShapes[i][d] = 1;
            out.outputStrides[i][d] = 0;
        }
    }
    for (int64_t i = 0; i < (int64_t)in.normalOutputShapes.size(); i++) {
        auto st = computeStrides(in.normalOutputShapes[i]);
        for (int64_t d = 0; d < delta; d++) {
            out.outputShapes[i][d] = 1;
            out.outputStrides[i][d] = 0;
        }
        for (int64_t d = 0; d < rank; d++) {
            out.outputShapes[i][d + delta] = in.normalOutputShapes[i][d];
            out.outputStrides[i][d + delta] = st[d];
        }
    }

    return 0;
}

} // namespace branch_api
} // namespace ascend_requant

#ifndef ASCEND_REQUANT_TILING_UT

static std::string Arr2String(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    if (n > 0) {
        for (int64_t i = 0; i < n - 1; ++i) {
            oss << arr[i] << ",";
        }
        oss << arr[n - 1];
    }
    oss << "]";
    return oss.str();
}

static bool PrecomputeStrides(const std::vector<int64_t>& s, std::vector<int64_t>& strides)
{
    int64_t rank = (int64_t)s.size();
    strides.assign(rank, 0);
    for (int64_t d = rank - 1; d >= 0; d--) {
        if (s[d] == 1) {
            strides[d] = 0;
            continue;
        }
        int64_t prod = 1;
        for (int64_t j = d + 1; j < rank; j++)
            prod *= s[j];
        strides[d] = prod;
    }
    return true;
}

class AscendRequantTiling {
public:
    explicit AscendRequantTiling(gert::TilingContext* ctx) : ctx_(ctx) {}

    ge::graphStatus RunTiling()
    {
        ge::graphStatus ret = GetShapeInfo();
        if (ret != ge::GRAPH_SUCCESS)
            return ret;

        int64_t mapped = (rank_ <= 4) ? 4 : 8;
        if (mapped == 4) {
            ret = DoTilingAndSet<4>();
            ctx_->SetTilingKey(GET_TPL_TILING_KEY(ASCEND_REQUANT_RANK_4, reluFlag_));
        } else {
            ret = DoTilingAndSet<8>();
            ctx_->SetTilingKey(GET_TPL_TILING_KEY(ASCEND_REQUANT_RANK_8, reluFlag_));
        }
        return ret;
    }

private:
    ge::graphStatus ReadPlatform()
    {
        fe::PlatFormInfos* platformInfo = ctx_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, platformInfo);
        auto ap = platform_ascendc::PlatformAscendC(platformInfo);
        coreNum_ = ap.GetCoreNumAiv();
        ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ReadShapes()
    {
        auto readInto = [&](const gert::StorageShape* shape, std::vector<std::vector<int64_t>>& dst) {
            std::vector<int64_t> dims;
            gert::Shape s = shape->GetStorageShape();
            for (size_t d = 0; d < s.GetDimNum(); ++d)
                dims.push_back(s.GetDim(d));
            dst.push_back(dims);
        };
        auto xShape = ctx_->GetInputShape(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, xShape);
        readInto(xShape, raw_input_shapes_);
        auto scShape = ctx_->GetInputShape(1);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, scShape);
        readInto(scShape, raw_input_shapes_);
        auto* computeNodeInfo = ctx_->GetComputeNodeInfo();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, computeNodeInfo);
        for (size_t i = 0; i < computeNodeInfo->GetOutputsNum(); ++i) {
            auto shape = ctx_->GetOutputShape(i);
            OP_CHECK_NULL_WITH_CONTEXT(ctx_, shape);
            readInto(shape, raw_output_shapes_);
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ValidateInputs()
    {
        auto xDesc = ctx_->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, xDesc);
        auto scaleDesc = ctx_->GetInputDesc(1);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, scaleDesc);
        auto yDesc = ctx_->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, yDesc);

        if (ascend_requant::public_api::ValidateDtype(xDesc->GetDataType(), scaleDesc->GetDataType()) !=
            ascend_requant::public_api::kOk) {
            OP_LOGE(ctx_->GetNodeName(), "dtype not supported, require x=INT32 and req_scale=UINT64");
            return ge::GRAPH_FAILED;
        }

        if (ascend_requant::public_api::ValidateFormat(xDesc->GetStorageFormat(), scaleDesc->GetStorageFormat(),
                                                       yDesc->GetStorageFormat()) != ascend_requant::public_api::kOk) {
            OP_LOGE(ctx_->GetNodeName(), "format not supported, require ND");
            return ge::GRAPH_FAILED;
        }

        int64_t xRank = static_cast<int64_t>(raw_input_shapes_[0].size());
        int64_t scaleRank = static_cast<int64_t>(raw_input_shapes_[1].size());
        if (ascend_requant::public_api::ValidateDimensions(xRank, scaleRank) != ascend_requant::public_api::kOk) {
            OP_LOGE(ctx_->GetNodeName(), "dimensions invalid: xRank=%lld, scaleRank=%lld", xRank, scaleRank);
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus ReadDtypeSize()
    {
        auto inputDesc = ctx_->GetInputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, inputDesc);
        ge::DataType dtype = inputDesc->GetDataType();
        if (dtype != ge::DT_INT32) {
            OP_LOGE(ctx_->GetNodeName(), "Unsupported dtype");
            return ge::GRAPH_FAILED;
        }
        dtype_size_ = 4;
        return ge::GRAPH_SUCCESS;
    }

    void ReadAttrs()
    {
        reluFlag_ = 0;
        auto* attrs = ctx_->GetAttrs();
        if (attrs != nullptr) {
            const bool* reluFlagPtr = attrs->GetBool(0);
            if (reluFlagPtr != nullptr && *reluFlagPtr) {
                reluFlag_ = 1;
            }
        }
    }

    void PrePadScaleBias() { return; }

    ge::graphStatus GetShapeInfo()
    {
        if (ReadPlatform() != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;
        if (ReadShapes() != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;
        if (ValidateInputs() != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;
        if (ReadDtypeSize() != ge::GRAPH_SUCCESS)
            return ge::GRAPH_FAILED;
        ReadAttrs();
        PrePadScaleBias();

        if (ascend_requant::public_api::PadAndSqueeze(raw_input_shapes_, raw_output_shapes_, max_bro_shape_,
                                                      normal_input_shapes_,
                                                      normal_output_shapes_) != ascend_requant::public_api::kOk) {
            OP_LOGE(ctx_->GetNodeName(), "scale/bias shape cannot broadcast to x");
            return ge::GRAPH_FAILED;
        }
        rank_ = (int64_t)max_bro_shape_.size();

        if (rank_ > 8) {
            OP_LOGE(ctx_->GetNodeName(), "rank(%lld) exceeds max supported dim 8", rank_);
            return ge::GRAPH_FAILED;
        }

        OP_LOGI(ctx_->GetNodeName(), "GetShapeInfo done rank %lld dtype %lld ub %llu core %llu relu_flag %lld", rank_,
                dtype_size_, ubSize_, coreNum_, reluFlag_);

        return ge::GRAPH_SUCCESS;
    }

    template <int64_t R, typename ShapeArr, typename StrideArr>
    static void FillSlots(ShapeArr shapes, StrideArr strides, const std::vector<std::vector<int64_t>>& norm,
                          const std::vector<std::vector<int64_t>>& norm_strides, int64_t num, int64_t max_slots,
                          int64_t delta)
    {
        int64_t rank = R - delta;
        for (int64_t i = 0; i < num; i++) {
            for (int64_t d = 0; d < delta; d++) {
                shapes[i][d] = 1;
                strides[i][d] = 0;
            }
            for (int64_t d = 0; d < rank; d++) {
                shapes[i][d + delta] = norm[i][d];
                strides[i][d + delta] = norm_strides[i][d];
            }
        }
        for (int64_t i = num; i < max_slots; i++)
            for (int64_t d = 0; d < R; d++) {
                shapes[i][d] = 1;
                strides[i][d] = 0;
            }
    }

    template <int64_t R>
    ge::graphStatus DoTilingAndSet()
    {
        auto* tiling = ctx_->GetTilingData<AscendRequantTilingData<R>>();
        OP_CHECK_NULL_WITH_CONTEXT(ctx_, tiling);

        int64_t ub_per_core = (int64_t)ubSize_;
        int64_t per_buf_bytes = (ub_per_core / PHYS_NODES) & ~31LL;

        constexpr int64_t MAX_DTYPE_SIZE = 8;
        ascend_requant::public_api::FindSplitAxis(max_bro_shape_, MAX_DTYPE_SIZE, ub_per_core, PHYS_NODES,
                                                  tiling->split);
        ascend_requant::public_api::MultiCoreSplit(max_bro_shape_, tiling->split, (int64_t)coreNum_, tiling->multicore);
        if (tiling->multicore.numCores < 1) {
            tiling->multicore.numCores = 1;
        }
        tiling->perBufBytes = per_buf_bytes;

        int64_t num_in = (int64_t)normal_input_shapes_.size();
        int64_t num_out = (int64_t)normal_output_shapes_.size();
        std::vector<std::vector<int64_t>> in_strides(num_in), out_strides(num_out);
        for (int64_t i = 0; i < num_in; i++)
            PrecomputeStrides(normal_input_shapes_[i], in_strides[i]);
        for (int64_t i = 0; i < num_out; i++)
            PrecomputeStrides(normal_output_shapes_[i], out_strides[i]);

        tiling->rank = rank_;
        tiling->reluFlag = reluFlag_;
        int64_t delta = R - rank_;

        for (int64_t d = 0; d < delta; d++)
            tiling->maxBroShape[d] = 1;
        for (int64_t d = 0; d < rank_; d++)
            tiling->maxBroShape[d + delta] = max_bro_shape_[d];

        tiling->split.axis += delta;

        tiling->numInputs = num_in;
        tiling->numOutputs = num_out;

        FillSlots<R>(tiling->inputShapes, tiling->inputStrides, normal_input_shapes_, in_strides, num_in,
                     MAX_INPUT_SLOTS, delta);
        FillSlots<R>(tiling->outputShapes, tiling->outputStrides, normal_output_shapes_, out_strides, num_out,
                     MAX_OUTPUT_SLOTS, delta);

        ctx_->SetBlockDim(tiling->multicore.numCores);
        LogTilingData<R>(tiling, num_in, num_out);
        return ge::GRAPH_SUCCESS;
    }

    template <int64_t R>
    void LogTilingData(AscendRequantTilingData<R>* tiling, int64_t num_in, int64_t num_out)
    {
        OP_LOGI(ctx_->GetNodeName(),
                "TilingData: perBufBytes=%lld rank=%lld->R=%d "
                "maxBroShape=%s "
                "split(axis=%lld aI=%lld aO=%lld aITail=%lld) "
                "multi(cores=%lld tiles=%lld main=%lld coresTail=%lld) num_in=%lld num_out=%lld reluFlag=%lld",
                tiling->perBufBytes, rank_, (int)R, Arr2String(tiling->maxBroShape, R).c_str(), tiling->split.axis,
                tiling->split.aI, tiling->split.aO, tiling->split.aITail, tiling->multicore.numCores,
                tiling->multicore.totalTiles, tiling->multicore.tilesMain, tiling->multicore.coresTail, num_in, num_out,
                reluFlag_);
    }

    gert::TilingContext* ctx_;
    std::vector<std::vector<int64_t>> raw_input_shapes_;
    std::vector<std::vector<int64_t>> raw_output_shapes_;
    std::vector<int64_t> max_bro_shape_;
    std::vector<std::vector<int64_t>> normal_input_shapes_;
    std::vector<std::vector<int64_t>> normal_output_shapes_;
    int64_t dtype_size_ = 0;
    int64_t rank_ = 0;
    int64_t reluFlag_ = 0;
    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
};

static ge::graphStatus TilingFuncAscendRequant(gert::TilingContext* context)
{
    AscendRequantTiling scaleTiling(context);
    auto ret = scaleTiling.RunTiling();
    if (ret != ge::GRAPH_SUCCESS)
        return ret;
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForAscendRequant(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    auto compileInfo = context->GetCompiledInfo<AscendRequantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto ap = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ap.GetCoreNumAiv();
    ap.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AscendRequant)
    .Tiling(TilingFuncAscendRequant)
    .TilingParse<AscendRequantCompileInfo>(TilingPrepareForAscendRequant);

#endif // !ASCEND_REQUANT_TILING_UT

} // namespace optiling
