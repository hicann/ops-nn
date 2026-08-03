/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file adaptive_avg_pool2d_base_tiling.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_POOL2D_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_ADAPTIVE_POOL2D_TILING_H_

#include <array>
#include <sstream>
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../op_kernel/arch35/adaptive_avg_pool2d_struct.h"

namespace optiling {
using Ops::NN::Optiling::TilingBaseClass;
using namespace AdaptiveAvgPool2dOp;
constexpr int64_t MAX_INT32 = 2147483647;
constexpr int64_t MAX_THREAD = 1024;
constexpr int64_t MIN_THREAD = 512;
constexpr uint64_t DCACHE_SIZE = 128 * 1024UL;
constexpr uint64_t DIM_N = 0;
constexpr uint64_t DIM_C = 1;
constexpr uint64_t DIM_H = 2;
constexpr uint64_t DIM_W = 3;
constexpr uint64_t OUTPUTSIZE_DIMW = 2;
constexpr uint64_t OUTPUT_DIM_MAX = 2;
constexpr uint64_t DIM_NUM_THREE = 3;
constexpr uint64_t DIM_NUM_FOUR = 4;
constexpr int64_t DTYPE_INT32 = 3;
constexpr int64_t DTYPE_INT64 = 9;
constexpr int64_t ONE_DIM = 1;
constexpr int64_t NONE_DIM = 0;
constexpr int64_t OUTPUT_IDX_SHAPE = 0;
constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr uint64_t KERNEL_CALC_COUNT_THERSHOLD = 10000;
constexpr uint64_t TILING_DOUBLE = 2;
constexpr uint64_t TILING_TRANS_ADDR_LEN = 16;
constexpr uint64_t TILING_INT32_MAX_VALUE = 2147483647UL;
// Window area (kH*kW) above which a b16 shape is retried with half ncFactor: a larger window
// needs more UB per nc, so trading vector width for hoFactor usually pays off. Distinct from
// SPLIT_C_KERNEL_W_LINE, which is a single-dimension kW threshold that happens to share the value.
constexpr uint64_t TILING_LARGE_KERNEL_AREA = 32;
// Upper bound on the hoFactor values tried when H is upsampled.
constexpr uint64_t TILING_HUPSAMPLE_MAX_HO_TRY = 8;

struct CommonComputeInfo {
    uint64_t xDtypeSize{0};
    uint64_t useCoreNum{0};
    uint64_t totalOuter{0};
    uint64_t blockFactor{0};
    uint64_t blockTail{0};
    uint64_t ncFactor{0};
    uint64_t hoFactor{0};
    uint64_t hiFactor{0};
    uint64_t ncOuter{0};
    uint64_t hoOuter{0};
    uint64_t ncTail{0};
    uint64_t hoTail{0};
    uint64_t kernelHMax{0};
    uint64_t kernelWMax{0};
    uint64_t vfLen{0};
    uint64_t alignNum{0};
    uint64_t availableUbSize{0};
    uint64_t inputQueSize{0};
    uint64_t resQue1Size{0};
    uint64_t resQue2Size{0};
};

struct BaseInput {
    uint64_t coreNum{0};
    uint64_t ubSize{0};
    ge::DataType xDtype{ge::DT_FLOAT};
    ge::DataType indicesDtype{ge::DT_INT32};
    ge::Format dataFormat{ge::Format::FORMAT_NDHWC};
    uint64_t nIn{0};
    uint64_t cIn{0};
    uint64_t hIn{0};
    uint64_t wIn{0};
    uint64_t hOut{0};
    uint64_t wOut{0};
};

struct AdaptivePool2dCompileInfo {
    uint64_t coreNum;
    uint64_t ubSizePlatForm;
};

class AdaptivePool2dBaseTiling : public TilingBaseClass {
public:
    explicit AdaptivePool2dBaseTiling(gert::TilingContext* context) : TilingBaseClass(context) {}
    ~AdaptivePool2dBaseTiling() override {}

    BaseInput input_;
    std::string nodeName = "";

protected:
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    uint64_t GetTilingKey() const override;
    void DumpTilingInfo() override;
    ge::graphStatus GetAndCheckDataFormat();
    ge::graphStatus GetRealOutDims(const int64_t* outputSize, const gert::Shape& xShape, size_t output_size_len,
                                   size_t input_dim_num);
    ge::graphStatus CheckNpuArch();
    uint64_t CalKernelSizeOneDimMax(uint64_t inSize, uint64_t outSize);
    ge::graphStatus CheckOutDims();

    template <typename ComputeT>
    void CalUbBlockFactor(ComputeT& computeInfo)
    {
        computeInfo.ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, computeInfo.ncFactor);
        computeInfo.ncTail = input_.nIn * input_.cIn - (computeInfo.ncOuter - 1) * computeInfo.ncFactor;
        computeInfo.hoOuter = Ops::Base::CeilDiv(input_.hOut, computeInfo.hoFactor);
        computeInfo.hoTail = input_.hOut - (computeInfo.hoOuter - 1) * computeInfo.hoFactor;

        computeInfo.totalOuter = computeInfo.ncOuter * computeInfo.hoOuter;
        computeInfo.blockFactor = Ops::Base::CeilDiv(computeInfo.totalOuter, input_.coreNum);
        computeInfo.useCoreNum = Ops::Base::CeilDiv(computeInfo.totalOuter, computeInfo.blockFactor);
        computeInfo.blockTail = computeInfo.totalOuter - (computeInfo.useCoreNum - 1) * computeInfo.blockFactor;
    }

    template <typename ComputeT>
    bool InitComputeBase(ComputeT& computeInfo)
    {
        computeInfo.xDtypeSize = ge::GetSizeByDataType(input_.xDtype);
        if (computeInfo.xDtypeSize == 0) {
            OP_LOGE(context_->GetNodeName(), "Get xDtype size is 0, not support");
            return false;
        }
        computeInfo.vfLen = Ops::Base::GetVRegSize(context_) / computeInfo.xDtypeSize;
        computeInfo.alignNum = Ops::Base::GetUbBlockSize(context_) / computeInfo.xDtypeSize;
        computeInfo.availableUbSize = input_.ubSize;
        computeInfo.ncFactor = computeInfo.vfLen;
        computeInfo.hoFactor = 1;
        computeInfo.hiFactor = 1;
        computeInfo.kernelHMax = CalKernelSizeOneDimMax(input_.hIn, input_.hOut);
        computeInfo.kernelWMax = CalKernelSizeOneDimMax(input_.wIn, input_.wOut);
        return true;
    }

    template <typename ComputeT>
    void OptimizeHoFactorForCores(ComputeT& computeInfo)
    {
        CalUbBlockFactor(computeInfo);
        while (computeInfo.useCoreNum < input_.coreNum && computeInfo.hoFactor > 1) {
            uint64_t lastBlockFactor = computeInfo.blockFactor;
            computeInfo.hoFactor--;
            CalUbBlockFactor(computeInfo);
            if (computeInfo.blockFactor > lastBlockFactor) {
                computeInfo.hoFactor++;
                CalUbBlockFactor(computeInfo);
                break;
            }
        }
    }

    // wInVal: input_.wIn for SplitH/W/UpsampleH, computeInfo_.wInFactor for SplitC.
    template <typename ComputeT>
    void CalCommonUbSplitSize(ComputeT& ci, uint64_t wInVal)
    {
        uint64_t wInAlign = Ops::Base::CeilAlign(wInVal, ci.alignNum);
        uint64_t wOutAlign = Ops::Base::CeilAlign(input_.wOut, ci.alignNum);
        uint64_t hoNum = ci.hoFactor;
        uint64_t hiNum = ci.hiFactor;
        uint64_t vlNum = ci.ncFactor;
        ci.inputQueSize = vlNum * hiNum * wInAlign * ci.xDtypeSize;
        uint64_t outTransAlign = Ops::Base::CeilAlign(hoNum * wOutAlign, TILING_TRANS_ADDR_LEN);
        ci.resQue1Size = outTransAlign * vlNum * sizeof(float);
        ci.resQue2Size = (ci.xDtypeSize < sizeof(float)) ? outTransAlign * vlNum * sizeof(float) : 0;
    }

    // Returns total excluding strategy-specific extras (e.g. SplitH's wiBuf).
    template <typename ComputeT>
    uint64_t CalCommonUbOccupy(const ComputeT& ci, uint64_t wInVal) const
    {
        uint64_t dataBlock = Ops::Base::GetUbBlockSize(context_);
        uint64_t wInAlign = Ops::Base::CeilAlign(wInVal, ci.alignNum);
        uint64_t wOutAlign = Ops::Base::CeilAlign(input_.wOut, ci.alignNum);
        uint64_t vlNum = ci.ncFactor;
        uint64_t hoNum = ci.hoFactor;
        uint64_t hiNum = ci.hiFactor;
        uint64_t outTransAlign = Ops::Base::CeilAlign(hoNum * wOutAlign, TILING_TRANS_ADDR_LEN);
        uint64_t transRowAlign = Ops::Base::CeilAlign(hiNum * wInAlign, TILING_TRANS_ADDR_LEN);
        uint64_t transBufSize = transRowAlign * vlNum * ci.xDtypeSize;
        uint64_t sumBufSize = wOutAlign * vlNum * sizeof(float);
        uint64_t outBufSize = outTransAlign * vlNum * sizeof(float);
        uint64_t wBufSize = Ops::Base::CeilAlign(input_.wOut * sizeof(int32_t), dataBlock) * TILING_DOUBLE;
        return ci.inputQueSize + ci.resQue1Size + ci.resQue2Size + transBufSize + sumBufSize + outBufSize + wBufSize;
    }

    uint64_t CalCommonTilingKey(uint64_t kernelType, const CommonComputeInfo& ci) const
    {
        int64_t maxIdxValue = std::max(input_.hIn * input_.hOut, input_.wIn * input_.wOut);
        uint64_t idxTypeMode = static_cast<uint64_t>(maxIdxValue) < TILING_INT32_MAX_VALUE ? TPL_INT32_UINT32 :
                                                                                             TPL_INT64_UINT64;
        uint64_t ncFactor = ci.ncFactor == Ops::Base::GetVRegSize(context_) / sizeof(float) ? TPL_NC_FACTOR_64 :
                                                                                              TPL_NC_FACTOR_128;
        return GET_TPL_TILING_KEY(kernelType, idxTypeMode, ncFactor, TPL_BIG_KERNEL_NDDMA);
    }

    // Pipeline: ShrinkHiFactor → BinarySearchMaxHoFactor → TryHalfNcFactor → OptimizeHoFactorForCores.
    template <typename ComputeT, typename MeetUbFunc>
    void StandardUbOptimization(ComputeT& ci, MeetUbFunc meetUb)
    {
        ci.ncFactor = ci.vfLen;
        ShrinkHiFactor(ci, meetUb);
        BinarySearchMaxHoFactor(ci, input_.hOut, meetUb);
        if (ci.xDtypeSize == TILING_DOUBLE && ci.kernelHMax * ci.kernelWMax > TILING_LARGE_KERNEL_AREA) {
            TryHalfNcFactor(ci, input_.hOut, meetUb);
        }
        OptimizeHoFactorForCores(ci);
    }

    template <typename ComputeT, typename MeetUbFunc>
    void BinarySearchMaxWInFactor(ComputeT& ci, uint64_t maxWIn, MeetUbFunc meetUb)
    {
        ci.wInFactor = maxWIn;
        if (!meetUb()) {
            uint64_t left = 1;
            uint64_t right = maxWIn;
            uint64_t best = 1;
            while (left <= right) {
                uint64_t mid = left + (right - left) / TILING_DOUBLE;
                ci.wInFactor = mid;
                if (meetUb()) {
                    best = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            ci.wInFactor = best;
        }
    }

    template <typename ComputeT, typename MeetUbFunc>
    void TryHalfNcFactorSplitC(ComputeT& ci, MeetUbFunc meetUb)
    {
        uint64_t origNcFactor = ci.ncFactor;
        uint64_t origHoFactor = ci.hoFactor;
        uint64_t origWInFactor = ci.wInFactor;
        ci.ncFactor = ci.vfLen / TILING_DOUBLE;
        ci.hoFactor = 1;
        BinarySearchMaxWInFactor(ci, input_.wIn, meetUb);
        BinarySearchMaxHoFactor(ci, input_.hOut, meetUb);
        if (ci.hoFactor <= origHoFactor) {
            ci.ncFactor = origNcFactor;
            ci.hoFactor = origHoFactor;
            ci.wInFactor = origWInFactor;
        }
    }

    template <typename ComputeT>
    uint64_t EstimateHUpsamplePerCoreCost(const ComputeT& ci, uint64_t hoFactor, uint64_t wInFactor) const
    {
        uint64_t ncOuter = Ops::Base::CeilDiv(input_.nIn * input_.cIn, ci.ncFactor);
        uint64_t hoOuter = Ops::Base::CeilDiv(input_.hOut, hoFactor);
        uint64_t totalOuter = ncOuter * hoOuter;
        uint64_t blockFactor = Ops::Base::CeilDiv(totalOuter, input_.coreNum);
        uint64_t numChunks = Ops::Base::CeilDiv(input_.wIn, wInFactor);
        return blockFactor * (ci.kernelWMax * input_.wOut + hoFactor * input_.wOut * numChunks);
    }

    template <typename ComputeT, typename MeetUbFunc>
    void OptimizeHoForHUpsample(ComputeT& ci, MeetUbFunc meetUb)
    {
        uint64_t origNcFactor = ci.ncFactor;
        uint64_t origHoFactor = ci.hoFactor;
        uint64_t origWInFactor = ci.wInFactor;
        uint64_t bestNcFactor = origNcFactor;
        uint64_t bestHoFactor = origHoFactor;
        uint64_t bestWInFactor = origWInFactor;
        uint64_t bestCost = EstimateHUpsamplePerCoreCost(ci, origHoFactor, origWInFactor);

        uint64_t maxTry = std::min(input_.hOut, TILING_HUPSAMPLE_MAX_HO_TRY);
        uint64_t ncTrials = (ci.xDtypeSize == TILING_DOUBLE) ? TILING_DOUBLE : 1;
        for (uint64_t ncIdx = 0; ncIdx < ncTrials; ncIdx++) {
            ci.ncFactor = (ncIdx == 0) ? ci.vfLen : ci.vfLen / TILING_DOUBLE;
            for (uint64_t tryHo = 2; tryHo <= maxTry; tryHo++) {
                ci.hoFactor = tryHo;
                ci.wInFactor = 1;
                BinarySearchMaxWInFactor(ci, input_.wIn, meetUb);
                if (ci.wInFactor < ci.alignNum) {
                    break;
                }
                uint64_t cost = EstimateHUpsamplePerCoreCost(ci, tryHo, ci.wInFactor);
                if (cost < bestCost) {
                    bestCost = cost;
                    bestNcFactor = ci.ncFactor;
                    bestHoFactor = tryHo;
                    bestWInFactor = ci.wInFactor;
                }
            }
        }
        ci.ncFactor = bestNcFactor;
        ci.hoFactor = bestHoFactor;
        ci.wInFactor = bestWInFactor;
        CalUbBlockFactor(ci);
    }
};

template <typename TilingDataT>
ge::graphStatus FillCommonTilingData(gert::TilingContext* context, const BaseInput& input,
                                     const CommonComputeInfo& computeInfo)
{
    TilingDataT* tilingData = context->GetTilingData<TilingDataT>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tilingData);
    tilingData->hIn = static_cast<int64_t>(input.hIn);
    tilingData->wIn = static_cast<int64_t>(input.wIn);
    tilingData->hOut = static_cast<int64_t>(input.hOut);
    tilingData->wOut = static_cast<int64_t>(input.wOut);
    tilingData->useCoreNum = static_cast<int64_t>(computeInfo.useCoreNum);
    tilingData->blockFactor = static_cast<int64_t>(computeInfo.blockFactor);
    tilingData->blockTail = static_cast<int64_t>(computeInfo.blockTail);
    tilingData->ncFactor = static_cast<int64_t>(computeInfo.ncFactor);
    tilingData->hoFactor = static_cast<int64_t>(computeInfo.hoFactor);
    tilingData->hiFactor = static_cast<int64_t>(computeInfo.hiFactor);
    tilingData->ncOuter = static_cast<int64_t>(computeInfo.ncOuter);
    tilingData->hoOuter = static_cast<int64_t>(computeInfo.hoOuter);
    tilingData->ncTail = static_cast<int64_t>(computeInfo.ncTail);
    tilingData->hoTail = static_cast<int64_t>(computeInfo.hoTail);
    tilingData->inputQueSize = static_cast<int64_t>(computeInfo.inputQueSize);
    tilingData->resQue1Size = static_cast<int64_t>(computeInfo.resQue1Size);
    tilingData->resQue2Size = static_cast<int64_t>(computeInfo.resQue2Size);
    return ge::GRAPH_SUCCESS;
}

inline void PrintCommonTilingData(gert::TilingContext* context, const BaseInput& input,
                                  const CommonComputeInfo& computeInfo)
{
    std::ostringstream info;
    info << "nc: " << input.nIn * input.cIn;
    info << ", hIn: " << input.hIn;
    info << ", wIn: " << input.wIn;
    info << ", hOut: " << input.hOut;
    info << ", wOut: " << input.wOut;
    info << ", useCoreNum: " << computeInfo.useCoreNum;
    info << ", blockFactor: " << computeInfo.blockFactor;
    info << ", blockTail: " << computeInfo.blockTail;
    info << ", ncFactor: " << computeInfo.ncFactor;
    info << ", hoFactor: " << computeInfo.hoFactor;
    info << ", hiFactor: " << computeInfo.hiFactor;
    info << ", ncOuter: " << computeInfo.ncOuter;
    info << ", hoOuter: " << computeInfo.hoOuter;
    info << ", inputQueSize: " << computeInfo.inputQueSize;
    info << ", resQue1Size: " << computeInfo.resQue1Size;
    info << ", resQue2Size: " << computeInfo.resQue2Size;
    OP_LOGI(context->GetNodeName(), "%s", info.str().c_str());
}

template <typename MeetUbFunc>
void ShrinkHiFactor(CommonComputeInfo& ci, MeetUbFunc meetUb)
{
    ci.hoFactor = 1;
    ci.hiFactor = ci.kernelHMax;
    while (ci.hiFactor > 1 && !meetUb()) {
        ci.hiFactor--;
    }
}

template <typename MeetUbFunc>
void BinarySearchMaxHoFactor(CommonComputeInfo& ci, uint64_t maxHo, MeetUbFunc meetUb)
{
    ci.hoFactor = maxHo;
    if (!meetUb()) {
        uint64_t left = 1;
        uint64_t right = maxHo;
        uint64_t best = 1;
        while (left <= right) {
            uint64_t mid = left + (right - left) / TILING_DOUBLE;
            ci.hoFactor = mid;
            if (meetUb()) {
                best = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        ci.hoFactor = best;
    }
}

template <typename MeetUbFunc>
void TryHalfNcFactor(CommonComputeInfo& ci, uint64_t maxHo, MeetUbFunc meetUb)
{
    uint64_t origNcFactor = ci.ncFactor;
    uint64_t origHiFactor = ci.hiFactor;
    uint64_t origHoFactor = ci.hoFactor;
    ci.ncFactor = ci.vfLen / TILING_DOUBLE;
    ShrinkHiFactor(ci, meetUb);
    BinarySearchMaxHoFactor(ci, maxHo, meetUb);
    if (ci.hoFactor <= origHoFactor) {
        ci.ncFactor = origNcFactor;
        ci.hiFactor = origHiFactor;
        ci.hoFactor = origHoFactor;
    }
}

#define DECLARE_SPLIT_TILING_CLASS(ClassName, ComputeInfoType)                                  \
    class ClassName : public AdaptivePool2dBaseTiling {                                         \
    public:                                                                                     \
        explicit ClassName(gert::TilingContext* context) : AdaptivePool2dBaseTiling(context) {} \
        ~ClassName() override {}                                                                \
        bool IsCapable() override;                                                              \
        ge::graphStatus DoOpTiling() override;                                                  \
        uint64_t GetTilingKey() const override;                                                 \
        ge::graphStatus PostTiling() override;                                                  \
                                                                                                \
    private:                                                                                    \
        bool IsMeetUbSize();                                                                    \
        void CalUbSplitSize();                                                                  \
        ge::graphStatus SetTilingData();                                                        \
        void PrintTilingData() const;                                                           \
        ComputeInfoType computeInfo_;                                                           \
    };

} // namespace optiling

#endif
